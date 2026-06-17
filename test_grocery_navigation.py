"""
Grocery Store Navigation Test Script
=====================================
Two tests for in-store navigation:
  Test 1: Full-photo navigation using all 192 grocery store photos
  Test 2: Blind navigation using only the initial position photo

Requirements:
  - Server running: python -m server.run_server
  - Ollama running with llama3.2-vision
  - Grocery photos accessible (C1-C6 camera paths)

Usage:
  python test_grocery_navigation.py --test 1 --photo-dir /path/to/grocery/photos
  python test_grocery_navigation.py --test 2 --initial-photo /path/to/aisle1.jpg
  python test_grocery_navigation.py --analyze  # Analyze existing detections only (no server needed)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Server URL
BASE_URL = "http://localhost:8000"

# Navigation goal
GOAL = "find cheese, then find frozen dumplings, then go to checkout cashier"

# Expected targets in order
TARGETS = ["cheese", "frozen dumplings", "checkout"]


@dataclass
class TestResult:
    test_name: str
    photos_used: int
    targets_found: List[str] = field(default_factory=list)
    targets_missed: List[str] = field(default_factory=list)
    total_steps: int = 0
    total_time_s: float = 0.0
    detections: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    vlm_responses: List[dict] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)


def analyze_existing_detections():
    """Analyze the existing detections_new_env.json without needing the server."""
    project_root = Path(__file__).parent
    det_file = project_root / "detections_new_env.json"

    if not det_file.exists():
        print("ERROR: detections_new_env.json not found")
        return

    data = json.loads(det_file.read_text(encoding="utf-8"))
    photos = data["photos"]
    obj_index = data["object_index"]

    print("=" * 70)
    print("GROCERY STORE DETECTION ANALYSIS (192 Photos)")
    print("=" * 70)

    # Overall stats
    all_scores = []
    label_stats = {}
    for p in photos:
        for obj in p["objects"]:
            all_scores.append(obj["score"])
            lbl = obj["label"]
            if lbl not in label_stats:
                label_stats[lbl] = {"count": 0, "scores": [], "photos": set()}
            label_stats[lbl]["count"] += 1
            label_stats[lbl]["scores"].append(obj["score"])
            label_stats[lbl]["photos"].add(p["filename"])

    print(f"\nTotal photos: {len(photos)}")
    print(f"Total detections: {len(all_scores)}")
    print(f"Unique labels: {len(label_stats)}")
    print(f"Avg confidence: {sum(all_scores)/len(all_scores):.1%}")
    print(f"Confidence range: [{min(all_scores):.1%}, {max(all_scores):.1%}]")

    # Camera paths
    print("\n--- Camera Path Coverage ---")
    cameras = {}
    for p in photos:
        cam = p["filename"].split("_")[0]
        cameras.setdefault(cam, []).append(p["filename"])
    for cam in sorted(cameras):
        print(f"  {cam}: {len(cameras[cam])} photos ({cameras[cam][0]} ... {cameras[cam][-1]})")

    # Per-label breakdown
    print("\n--- Detection Labels (sorted by count) ---")
    for lbl in sorted(label_stats, key=lambda x: -label_stats[x]["count"]):
        s = label_stats[lbl]
        avg = sum(s["scores"]) / len(s["scores"])
        mx = max(s["scores"])
        print(f"  {lbl:40s} count={s['count']:3d}  photos={len(s['photos']):3d}  "
              f"avg={avg:.1%}  best={mx:.1%}")

    # Navigation target analysis
    print("\n--- Navigation Target Detection ---")
    grocery_targets = {
        "cheese": [],
        "dairy": [],
        "frozen": [],
        "dumpling": [],
        "checkout": [],
        "cashier": [],
        "cash register": [],
    }
    for lbl in label_stats:
        for target in grocery_targets:
            if target in lbl.lower():
                grocery_targets[target].append(lbl)

    for target, found_labels in grocery_targets.items():
        if found_labels:
            print(f"  {target}: FOUND as {found_labels}")
        else:
            print(f"  {target}: NOT DETECTED")

    # Proxy targets
    print("\n--- Proxy Targets (indirect detection) ---")
    fridge_photos = [p for p in photos
                     if any("refrigerator" in o["label"] and o["score"] > 0.4
                            for o in p["objects"])]
    print(f"  Refrigerator (>0.4 conf): {len(fridge_photos)} photos "
          f"(proxy for dairy/frozen sections)")
    if fridge_photos:
        best = max(fridge_photos,
                   key=lambda p: max(o["score"] for o in p["objects"]
                                     if "refrigerator" in o["label"]))
        best_score = max(o["score"] for o in best["objects"]
                         if "refrigerator" in o["label"])
        print(f"    Best: {best['filename']} at {best_score:.1%}")

    sign_photos = [p for p in photos
                   if any("sign" in o["label"] and o["score"] > 0.4
                          for o in p["objects"])]
    print(f"  Sign (>0.4 conf): {len(sign_photos)} photos "
          f"(proxy for aisle/checkout markers)")

    # Error analysis
    print("\n--- Error Analysis ---")
    bathroom_count = sum(1 for p in photos
                         if any("bathroom" in o["label"] for o in p["objects"]))
    print(f"  'bathroom restroom' false positives: {bathroom_count}/{len(photos)} photos "
          f"({bathroom_count/len(photos):.0%})")

    merged_labels = [lbl for lbl in label_stats if " " in lbl and len(lbl.split()) > 2]
    print(f"  Ambiguous merged labels: {len(merged_labels)}")
    for lbl in merged_labels:
        print(f"    '{lbl}' ({label_stats[lbl]['count']} detections)")

    low_conf = sum(1 for s in all_scores if s < 0.45)
    print(f"  Low confidence (<45%): {low_conf}/{len(all_scores)} "
          f"({low_conf/len(all_scores):.0%})")

    # Confidence distribution
    print("\n--- Confidence Distribution ---")
    brackets = [(0.3, 0.35), (0.35, 0.4), (0.4, 0.45), (0.45, 0.5),
                (0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7)]
    for lo, hi in brackets:
        count = sum(1 for s in all_scores if lo <= s < hi)
        bar = "#" * (count // 5)
        print(f"  {lo:.0%}-{hi:.0%}: {count:3d} {bar}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
  1. NO direct detection of cheese, frozen dumplings, or checkout.
  2. 'refrigerator' (67.5% best) is the best proxy for dairy/frozen areas.
  3. 'bathroom restroom' appears in ALL 192 photos (systematic false positive).
  4. 86% of detections are in the LOW confidence range (30-45%).
  5. The VLM (LLaMA 3.2 Vision) is essential for actual target identification.
  6. Grocery-specific detection prompts would significantly improve results.
""")


def run_test1(photo_dir: str):
    """Test 1: Full-photo navigation using all 192 grocery store photos."""
    import requests

    print("=" * 70)
    print("TEST 1: Full-Photo Navigation (192 photos)")
    print("=" * 70)

    result = TestResult(test_name="Test 1: Full Photo", photos_used=0)
    start_time = time.time()

    # 1. Create session
    print(f"\n[1] Creating session with goal: {GOAL}")
    try:
        r = requests.post(f"{BASE_URL}/session",
                          json={"goal": GOAL}, timeout=60)
        r.raise_for_status()
        session = r.json()
        sid = session["session_id"]
        print(f"    Session ID: {sid}")
        print(f"    Goal objects: {session.get('goal_objects', [])}")
        print(f"    Guidance: {session.get('guidance', '')}")
    except Exception as e:
        result.errors.append(f"Session creation failed: {e}")
        print(f"    ERROR: {e}")
        return result

    # 2. Upload all photos
    photo_dir = Path(photo_dir)
    photos = sorted(photo_dir.glob("*.jpg")) + sorted(photo_dir.glob("*.JPG"))
    if not photos:
        result.errors.append(f"No photos found in {photo_dir}")
        print(f"    ERROR: No photos found in {photo_dir}")
        return result

    print(f"\n[2] Uploading {len(photos)} photos...")
    for i, photo_path in enumerate(photos):
        print(f"    [{i+1}/{len(photos)}] {photo_path.name}...", end=" ", flush=True)
        try:
            with open(photo_path, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/session/{sid}/photo",
                    files={"photo": (photo_path.name, f, "image/jpeg")},
                    timeout=300,
                )
            r.raise_for_status()
            resp = r.json()
            action = resp.get("action", "?")
            guidance = resp.get("guidance", "")[:80]
            node_id = resp.get("node_id", "?")
            print(f"action={action} node={node_id} | {guidance}")

            result.vlm_responses.append(resp)
            result.photos_used += 1
            result.total_steps += 1

            if action == "ARRIVED":
                result.targets_found.append(guidance[:50])
                print(f"    >>> TARGET REACHED: {guidance[:80]}")

            if action == "ASK" and resp.get("question"):
                print(f"    >>> QUESTION: {resp['question']}")
                # Auto-answer YES for testing
                ans_r = requests.post(
                    f"{BASE_URL}/session/{sid}/answer",
                    json={"answer": "yes"},
                    timeout=60,
                )
                if ans_r.ok:
                    ans_resp = ans_r.json()
                    print(f"    >>> ANSWER: yes -> {ans_resp.get('action', '?')}: "
                          f"{ans_resp.get('guidance', '')[:60]}")

        except Exception as e:
            result.errors.append(f"Photo {photo_path.name}: {e}")
            print(f"ERROR: {e}")

    # 3. Get final map
    print(f"\n[3] Retrieving topological map...")
    try:
        r = requests.get(f"{BASE_URL}/session/{sid}/map", timeout=30)
        if r.ok:
            map_data = r.json()
            print(f"    Nodes: {len(map_data.get('nodes', []))}")
            print(f"    Edges: {len(map_data.get('edges', []))}")
    except Exception as e:
        print(f"    Map retrieval error: {e}")

    result.total_time_s = time.time() - start_time
    result.targets_missed = [t for t in TARGETS if t not in str(result.targets_found)]

    # Summary
    print(f"\n{'=' * 70}")
    print(f"TEST 1 RESULTS")
    print(f"{'=' * 70}")
    print(f"  Photos processed: {result.photos_used}")
    print(f"  Total steps: {result.total_steps}")
    print(f"  Targets found: {len(result.targets_found)}/3")
    print(f"  Targets missed: {result.targets_missed}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Total time: {result.total_time_s:.1f}s")

    return result


def run_test2(initial_photo: str):
    """Test 2: Blind navigation with only initial position photo."""
    import requests

    print("=" * 70)
    print("TEST 2: Blind Navigation (1 initial photo)")
    print("=" * 70)

    result = TestResult(test_name="Test 2: Blind Navigation", photos_used=0)
    start_time = time.time()

    # 1. Create session
    print(f"\n[1] Creating session with goal: {GOAL}")
    try:
        r = requests.post(f"{BASE_URL}/session",
                          json={"goal": GOAL}, timeout=60)
        r.raise_for_status()
        session = r.json()
        sid = session["session_id"]
        print(f"    Session ID: {sid}")
        print(f"    Goal objects: {session.get('goal_objects', [])}")
        print(f"    Initial guidance: {session.get('guidance', '')}")
    except Exception as e:
        result.errors.append(f"Session creation failed: {e}")
        print(f"    ERROR: {e}")
        return result

    # 2. Upload initial photo
    initial = Path(initial_photo)
    if not initial.exists():
        result.errors.append(f"Initial photo not found: {initial}")
        print(f"    ERROR: Photo not found: {initial}")
        return result

    print(f"\n[2] Uploading initial position photo: {initial.name}")
    try:
        with open(initial, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/session/{sid}/photo",
                files={"photo": (initial.name, f, "image/jpeg")},
                timeout=300,
            )
        r.raise_for_status()
        resp = r.json()
        result.vlm_responses.append(resp)
        result.photos_used += 1
        result.total_steps += 1
        print(f"    Action: {resp.get('action', '?')}")
        print(f"    Guidance: {resp.get('guidance', '')}")
        if resp.get("question"):
            print(f"    Question: {resp['question']}")
    except Exception as e:
        result.errors.append(f"Initial photo upload failed: {e}")
        print(f"    ERROR: {e}")
        return result

    # 3. Interactive navigation loop
    print(f"\n[3] Starting interactive navigation...")
    print("    The system will now give you directions.")
    print("    At each step, take a photo in the direction indicated")
    print("    and upload it to continue navigation.\n")

    step = 1
    arrived_count = 0
    max_steps = 15

    while arrived_count < 3 and step < max_steps:
        last_resp = result.vlm_responses[-1]
        action = last_resp.get("action", "MOVE")
        guidance = last_resp.get("guidance", "")

        print(f"\n  --- Step {step} ---")
        print(f"  Action: {action}")
        print(f"  Guidance: {guidance}")

        if action == "ARRIVED":
            arrived_count += 1
            result.targets_found.append(f"Target {arrived_count}: {guidance[:50]}")
            print(f"  >>> TARGET {arrived_count}/3 REACHED!")
            if arrived_count >= 3:
                break

        if action == "ASK" and last_resp.get("question"):
            print(f"  Question: {last_resp['question']}")
            answer = input("  Your answer (or 'yes'/'no'): ").strip() or "yes"
            try:
                r = requests.post(
                    f"{BASE_URL}/session/{sid}/answer",
                    json={"answer": answer},
                    timeout=60,
                )
                if r.ok:
                    resp = r.json()
                    result.vlm_responses.append(resp)
                    result.total_steps += 1
                    continue
            except Exception as e:
                result.errors.append(f"Answer failed: {e}")

        # Prompt user for next photo
        photo_path = input(f"\n  Enter path to next photo (or 'quit'): ").strip()
        if photo_path.lower() in ("quit", "q", "exit"):
            break

        if not Path(photo_path).exists():
            print(f"  ERROR: File not found: {photo_path}")
            continue

        try:
            with open(photo_path, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/session/{sid}/photo",
                    files={"photo": (Path(photo_path).name, f, "image/jpeg")},
                    timeout=300,
                )
            r.raise_for_status()
            resp = r.json()
            result.vlm_responses.append(resp)
            result.photos_used += 1
            result.total_steps += 1
        except Exception as e:
            result.errors.append(f"Photo upload failed: {e}")
            print(f"  ERROR: {e}")

        step += 1

    result.total_time_s = time.time() - start_time
    result.targets_missed = [t for t in TARGETS
                             if t not in str(result.targets_found)]

    # Summary
    print(f"\n{'=' * 70}")
    print(f"TEST 2 RESULTS")
    print(f"{'=' * 70}")
    print(f"  Photos used: {result.photos_used}")
    print(f"  Total steps: {result.total_steps}")
    print(f"  Targets found: {len(result.targets_found)}/3")
    print(f"  Targets missed: {result.targets_missed}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Total time: {result.total_time_s:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="Grocery Store Navigation Tests")
    parser.add_argument("--test", type=int, choices=[1, 2],
                        help="Which test to run (1=full photo, 2=blind)")
    parser.add_argument("--photo-dir", type=str,
                        help="Directory with all grocery photos (Test 1)")
    parser.add_argument("--initial-photo", type=str,
                        help="Path to initial position photo (Test 2)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing detections only (no server needed)")
    args = parser.parse_args()

    if args.analyze:
        analyze_existing_detections()
    elif args.test == 1:
        if not args.photo_dir:
            print("ERROR: --photo-dir required for Test 1")
            sys.exit(1)
        run_test1(args.photo_dir)
    elif args.test == 2:
        if not args.initial_photo:
            print("ERROR: --initial-photo required for Test 2")
            sys.exit(1)
        run_test2(args.initial_photo)
    else:
        # Default: run analysis
        analyze_existing_detections()


if __name__ == "__main__":
    main()
