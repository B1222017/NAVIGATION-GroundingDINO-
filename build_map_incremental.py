"""Incremental topological map builder — processes photos one by one.

Each photo is processed individually with GroundingDINO, added to the
growing detection list, and the topological map is regenerated after
every photo. Outputs:
  - detections_incremental.json  (all accumulated detections so far)
  - topomap_step/step_NNN.png   (map snapshot after each photo)

Usage:
    python build_map_incremental.py
    python build_map_incremental.py --photo-dir ~/Desktop/output_jpg/系辦
    python build_map_incremental.py --photo-dir ~/Desktop/output_jpg/系辦 --goal "find refrigerator"
    python build_map_incremental.py --resume    # skip already-detected photos
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

log = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_PHOTO_DIR = os.path.expanduser("~/Desktop/output_jpg/系辦")
DEFAULT_GOAL      = "電腦"
DEFAULT_CLASSES   = [
    "door", "cabinet", "chair", "table", "desk", "sofa",
    "sign", "fire extinguisher", "plant", "printer",
    "whiteboard", "monitor", "computer", "bookshelf", "window",
    "counter", "reception", "bulletin board", "notice board",
    # environment additions
    "refrigerator", "water dispenser", "shelf", "trash can", "poster",
]
OUTPUT_JSON    = "detections_incremental.json"
OUTPUT_MAP_DIR = "topomap_step"
SIM_THRESHOLD  = 0.25

# Goal → extra detection classes (Chinese & English variants)
GOAL_CLASS_MAP: dict[str, list[str]] = {
    "電腦": [
        "電腦", "桌上型電腦", "筆記型電腦", "螢幕", "鍵盤", "滑鼠",
        "computer", "laptop", "desktop computer", "monitor", "keyboard", "mouse", "screen",
    ],
    "辦公室": [
        "辦公室", "教師辦公室", "辦公區域", "房間門", "房間標牌",
        "指示牌", "接待區", "辦公桌", "走廊", "nameplate", "office door",
    ],
    "冰箱": ["冰箱", "refrigerator", "fridge", "冷藏"],
    "印表機": ["印表機", "printer", "laser printer"],
    "飲水機": ["飲水機", "water dispenser", "water cooler"],
}


def goal_to_classes(goal: str) -> list[str]:
    """Return extra detection classes relevant to the given goal string."""
    extra: list[str] = []
    goal_lower = goal.lower()
    for key, classes in GOAL_CLASS_MAP.items():
        if key in goal or key.lower() in goal_lower:
            extra.extend(classes)
    # Always include the raw goal text itself as a class
    if goal not in extra:
        extra.append(goal)
    return extra


# ── Photo discovery ─────────────────────────────────────────────────────

def find_photos(photo_dir: str) -> list[Path]:
    root = Path(photo_dir)
    seen, files = set(), []
    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG"):
        for f in root.glob(pattern):
            if f.name.lower() not in seen:
                seen.add(f.name.lower())
                files.append(f)
    files.sort(key=lambda p: p.name)
    return files


# ── Load / save incremental JSON ────────────────────────────────────────

def load_state(json_path: str) -> dict:
    if Path(json_path).exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"goal": "", "photos": []}


def save_state(state: dict, json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ── GroundingDINO detection ─────────────────────────────────────────────

def init_perception():
    """Load GroundingDINO model once."""
    sys.path.insert(0, str(Path(__file__).parent))
    from server.perception import Perception
    p = Perception()
    p.load()
    return p


def detect_photo(perception, photo_path: Path, classes: list[str]) -> list[dict]:
    detections = perception.detect(str(photo_path), classes)
    return [
        {"label": d.label, "score": round(d.score, 3),
         "box": [round(x, 1) for x in d.box]}
        for d in detections
    ]


# ── Topomap generation (reuses generate_topomap.py) ─────────────────────

def build_and_render_map(state: dict, step: int, goal: str, output_dir: Path) -> None:
    """Re-cluster all photos so far and render a map snapshot."""
    from generate_topomap import (
        cluster_photos_into_zones, build_zone_info,
        generate_zone_label, build_edges,
        find_goal_zones, parse_goals,
        find_navigation_path, render_map,
    )
    import networkx as nx

    photos = state["photos"]
    if len(photos) < 2:
        log.info("  Skipping map render — need at least 2 photos")
        return

    goals = parse_goals(goal)
    zones = cluster_photos_into_zones(photos, sim_threshold=SIM_THRESHOLD)
    zone_info = build_zone_info(zones)
    goal_zones = find_goal_zones(zone_info, goals)
    start_zone = 0

    G = nx.Graph()
    used_names: set = set()
    for zid, info in zone_info.items():
        label = generate_zone_label(zid, info, used_names)
        G.add_node(zid, label=label)
    for u, v, shared in build_edges(zones, zone_info):
        G.add_edge(u, v, shared=shared)

    path_edges = find_navigation_path(G, start_zone, goal_zones)

    prefix = str(output_dir / f"step_{step:03d}")
    render_map(G, zone_info, goal_zones, start_zone, path_edges, goals, prefix)
    # rename _hires copy to something consistent
    hires = Path(f"{prefix}_hires.png")
    if hires.exists():
        hires.rename(output_dir / f"step_{step:03d}_hires.png")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build topological map one photo at a time"
    )
    parser.add_argument("--photo-dir", default=DEFAULT_PHOTO_DIR,
                        help=f"Folder with photos (default: {DEFAULT_PHOTO_DIR})")
    parser.add_argument("--photos", default="",
                        help="Comma-separated photo filenames to process (e.g. IMG_1433.jpg,IMG_1435.jpg). "
                             "If omitted, all photos in --photo-dir are used.")
    parser.add_argument("--goal", default=DEFAULT_GOAL,
                        help=f"Navigation goal (default: '{DEFAULT_GOAL}')")
    parser.add_argument("--output-json", default=OUTPUT_JSON,
                        help=f"Accumulated detections JSON (default: {OUTPUT_JSON})")
    parser.add_argument("--map-dir", default=OUTPUT_MAP_DIR,
                        help=f"Output folder for step maps (default: {OUTPUT_MAP_DIR})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip photos already present in the output JSON")
    parser.add_argument("--no-map", action="store_true",
                        help="Only run detection, skip map rendering")
    parser.add_argument("--map-every", type=int, default=1,
                        help="Render map every N photos (default: 1 = every photo)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    photo_dir = Path(os.path.expanduser(args.photo_dir))
    if not photo_dir.exists():
        log.error("Photo directory not found: %s", photo_dir)
        sys.exit(1)

    map_dir = Path(args.map_dir)
    map_dir.mkdir(exist_ok=True)

    # Build combined class list: base + goal-specific
    goal_classes = goal_to_classes(args.goal)
    detect_classes = DEFAULT_CLASSES + [c for c in goal_classes if c not in DEFAULT_CLASSES]
    log.info("Detection classes (%d): %s", len(detect_classes), ", ".join(detect_classes))

    # Discover photos
    photos = find_photos(str(photo_dir))
    if not photos:
        log.error("No images found in %s", photo_dir)
        sys.exit(1)

    # Filter to specific filenames if --photos is given
    if args.photos:
        names = {n.strip() for n in args.photos.split(",") if n.strip()}
        # Accept with or without .jpg extension
        def _match(p: Path) -> bool:
            return p.name in names or p.stem in names
        photos = [p for p in photos if _match(p)]
        if not photos:
            log.error("None of the specified photos found in %s: %s", photo_dir, args.photos)
            sys.exit(1)
        log.info("Processing %d specified photos: %s", len(photos), [p.name for p in photos])
    else:
        log.info("Found %d photos in %s", len(photos), photo_dir)

    # Load existing state (for --resume)
    state = load_state(args.output_json)
    state["goal"] = args.goal
    already_done = {p["filename"] for p in state["photos"]} if args.resume else set()
    if already_done:
        log.info("Resuming — %d photos already processed", len(already_done))

    # Load model
    log.info("Loading GroundingDINO...")
    perception = init_perception()

    # Process photos one by one
    processed = len(state["photos"]) if args.resume else 0
    for i, photo_path in enumerate(photos, start=1):
        filename = photo_path.name

        if filename in already_done:
            log.info("[%d/%d] SKIP %s (already processed)", i, len(photos), filename)
            continue

        log.info("[%d/%d] ── Detecting: %s", i, len(photos), filename)
        objects = detect_photo(perception, photo_path, detect_classes)

        if objects:
            labels = ", ".join(f"{o['label']}({o['score']})" for o in objects)
            log.info("  Found: %s", labels)
        else:
            log.info("  No detections")

        # Accumulate
        state["photos"].append({
            "filename": filename,
            "objects": objects,
            "summary": ", ".join(f"{o['label']}({o['score']})" for o in objects),
        })
        processed += 1
        save_state(state, args.output_json)

        # Render map snapshot
        if not args.no_map and processed % args.map_every == 0:
            log.info("  Building map after %d photos...", processed)
            try:
                build_and_render_map(state, processed, args.goal, map_dir)
                log.info("  Map saved: %s/step_%03d.png", map_dir, processed)
            except Exception as e:
                log.warning("  Map render failed: %s", e)

    log.info("\n=== Done! ===")
    log.info("Total photos processed: %d", processed)
    log.info("Detections saved to: %s", args.output_json)
    if not args.no_map:
        log.info("Map snapshots in: %s/", map_dir)
        log.info("Final map: %s/step_%03d.png", map_dir, processed)


if __name__ == "__main__":
    main()
