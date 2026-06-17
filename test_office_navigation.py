"""
Office Navigation Test Script (CSIE Department)
=================================================
Two tests for office navigation:
  Test 1: Full-photo navigation using all 56 office photos
  Test 2: Blind navigation using only the initial sofa photo

Goal: Find refrigerator -> Find fire extinguisher -> Exit left-side door

Usage:
  python test_office_navigation.py --analyze
  python test_office_navigation.py --test 1 --photo-dir /path/to/office/photos
  python test_office_navigation.py --test 2 --initial-photo /path/to/sofa_photo.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

BASE_URL = "http://localhost:8000"
GOAL = "find the refrigerator, then find the fire extinguisher, then exit through the left door (not the right side door)"
TARGETS = ["refrigerator", "fire extinguisher", "left door"]


@dataclass
class TestResult:
    test_name: str
    photos_used: int = 0
    targets_found: List[str] = field(default_factory=list)
    targets_missed: List[str] = field(default_factory=list)
    total_steps: int = 0
    total_time_s: float = 0.0
    errors: List[str] = field(default_factory=list)
    vlm_responses: List[dict] = field(default_factory=list)


def analyze_existing_detections():
    """Analyze detections_groundingdino.json for the office environment."""
    project_root = Path(__file__).parent
    det_file = project_root / "detections_groundingdino.json"

    if not det_file.exists():
        print("ERROR: detections_groundingdino.json not found")
        return

    data = json.loads(det_file.read_text(encoding="utf-8"))
    photos = data["photos"]

    print("=" * 70)
    print("CSIE OFFICE DETECTION ANALYSIS (56 Photos)")
    print("=" * 70)
    print(f"Goal: {data['goal']}")

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

    print("\n--- Detection Labels (sorted by count) ---")
    for lbl in sorted(label_stats, key=lambda x: -label_stats[x]["count"]):
        s = label_stats[lbl]
        avg = sum(s["scores"]) / len(s["scores"])
        mx = max(s["scores"])
        print(f"  {lbl:40s} count={s['count']:3d}  photos={len(s['photos']):3d}  "
              f"avg={avg:.1%}  best={mx:.1%}")

    # Target analysis
    print("\n--- TARGET 1: REFRIGERATOR ---")
    fridge_hits = []
    for p in photos:
        for o in p["objects"]:
            if "refrigerator" in o["label"]:
                fridge_hits.append((p["filename"], o["score"]))
    fridge_hits.sort(key=lambda x: -x[1])
    for fn, sc in fridge_hits[:10]:
        print(f"  {fn}: {sc:.1%}")
    print(f"  Total: {len(fridge_hits)} detections in {len(set(f for f,_ in fridge_hits))} photos")

    print("\n--- TARGET 2: FIRE EXTINGUISHER ---")
    fire_hits = []
    for p in photos:
        for o in p["objects"]:
            if "extinguisher" in o["label"]:
                fire_hits.append((p["filename"], o["score"]))
    fire_hits.sort(key=lambda x: -x[1])
    for fn, sc in fire_hits[:10]:
        print(f"  {fn}: {sc:.1%}")
    print(f"  Total: {len(fire_hits)} detections in {len(set(f for f,_ in fire_hits))} photos")

    print("\n--- TARGET 3: DOOR (left side exit) ---")
    door_hits = []
    for p in photos:
        for o in p["objects"]:
            if o["label"] == "door" and o["score"] > 0.4:
                door_hits.append((p["filename"], o["score"],
                                  o["box"][0] if o["box"] else None))
    door_hits.sort(key=lambda x: -x[1])
    for fn, sc, x1 in door_hits[:10]:
        side = "LEFT?" if x1 and x1 < 1500 else "RIGHT?" if x1 and x1 > 2500 else "CENTER"
        print(f"  {fn}: {sc:.1%}  x1={x1:.0f}  ({side})" if x1 else f"  {fn}: {sc:.1%}")
    print(f"  Total: {len(door_hits)} high-conf door detections")

    # Zone analysis
    print("\n--- ZONE MAPPING ---")
    zones = {
        "Zone 2 - Sofa Area (START)": range(1432, 1438),
        "Zone 1 - Fire Ext Area": [1438, 1439],
        "Zone 3 - Open Area": range(1440, 1444),
        "Zone 4 - Door Area": [1444, 1445],
        "Zone 5 - Refrigerator Area": range(1446, 1449),
        "Zone 6 - Open Area 2": range(1449, 1454),
        "Zone 7 - Sign Area": range(1454, 1461),
        "Zone 8 - Chair Sofa Area": range(1461, 1467),
        "Zone 9 - Cabinet Area": range(1467, 1474),
        "Zone 10 - Table Desk Area": range(1474, 1484),
        "Zone 11 - Corridor": [1484, 1485],
        "Zone 12 - Printer Area": [1486, 1487],
    }

    photo_map = {p["filename"]: p for p in photos}
    for zone_name, nums in zones.items():
        targets = {"fridge": 0, "fire": 0, "door": 0, "sofa": 0}
        for num in nums:
            fn = f"IMG_{num}.jpg"
            if fn in photo_map:
                for o in photo_map[fn]["objects"]:
                    if "refrigerator" in o["label"]:
                        targets["fridge"] = max(targets["fridge"], o["score"])
                    if "extinguisher" in o["label"]:
                        targets["fire"] = max(targets["fire"], o["score"])
                    if o["label"] == "door":
                        targets["door"] = max(targets["door"], o["score"])
                    if "sofa" in o["label"]:
                        targets["sofa"] = max(targets["sofa"], o["score"])

        parts = []
        if targets["sofa"]: parts.append(f"SOFA({targets['sofa']:.0%})")
        if targets["fridge"]: parts.append(f"FRIDGE({targets['fridge']:.0%})")
        if targets["fire"]: parts.append(f"FIRE({targets['fire']:.0%})")
        if targets["door"]: parts.append(f"DOOR({targets['door']:.0%})")
        print(f"  {zone_name} ({len(list(nums))} photos)")
        print(f"    {' | '.join(parts) if parts else 'no targets'}")

    # Confidence distribution
    print("\n--- Confidence Distribution ---")
    brackets = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7),
                (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for lo, hi in brackets:
        count = sum(1 for s in all_scores if lo <= s < hi)
        bar = "#" * (count // 2)
        print(f"  {lo:.0%}-{hi:.0%}: {count:3d} {bar}")

    # Error analysis
    print("\n--- Error Analysis ---")
    cabinet_count = label_stats.get("cabinet", {}).get("count", 0)
    print(f"  Cabinet overdetection: {cabinet_count} detections (noise)")

    fire_at_start = any("extinguisher" in o["label"] and o["score"] > 0.5
                        for p in photos if p["filename"] in ["IMG_1434.jpg", "IMG_1435.jpg"]
                        for o in p["objects"])
    print(f"  Fire ext visible at START: {'YES (may cause premature ARRIVED)' if fire_at_start else 'No'}")

    fridge_at_start = any("refrigerator" in o["label"] and o["score"] > 0.5
                          for p in photos if p["filename"] == "IMG_1436.jpg"
                          for o in p["objects"])
    print(f"  Fridge visible at START: {'YES (0.62 from sofa area)' if fridge_at_start else 'No'}")

    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print("=" * 70)
    print("""
  1. ALL 3 targets directly detectable by GroundingDINO:
     - Refrigerator: 84.9% best confidence (IMG_1472)
     - Fire Extinguisher: 82.2% best confidence (IMG_1454)
     - Door: 77.0% best confidence (IMG_1450)
  2. Average confidence 52% is MUCH better than grocery store (41.2%).
  3. LEFT vs RIGHT door is the main challenge (GroundingDINO cannot tell).
  4. Fire extinguisher visible from START may cause premature ARRIVED.
  5. Office environment has distinct, well-separated objects = easier detection.
  6. Recommended path: Zone 2 (sofa) -> Zone 5/9 (fridge) -> Zone 7 (fire ext) -> Zone 8 -> Left Door.
""")


def run_test1(photo_dir: str):
    """Test 1: Full-photo navigation using all 56 office photos."""
    import requests

    print("=" * 70)
    print("TEST 1: Full-Photo Office Navigation (56 photos)")
    print("=" * 70)

    result = TestResult(test_name="Test 1: Full Photo Office")
    start_time = time.time()

    print(f"\n[1] Creating session with goal: {GOAL}")
    try:
        r = requests.post(f"{BASE_URL}/session", json={"goal": GOAL}, timeout=60)
        r.raise_for_status()
        session = r.json()
        sid = session["session_id"]
        print(f"    Session ID: {sid}")
        print(f"    Goal objects: {session.get('goal_objects', [])}")
    except Exception as e:
        result.errors.append(f"Session creation failed: {e}")
        print(f"    ERROR: {e}")
        return result

    photos = sorted(Path(photo_dir).glob("IMG_*.jpg"))
    if not photos:
        print(f"    ERROR: No IMG_*.jpg photos found in {photo_dir}")
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
            print(f"action={action} | {guidance}")

            result.vlm_responses.append(resp)
            result.photos_used += 1
            result.total_steps += 1

            if action == "ARRIVED":
                result.targets_found.append(guidance[:50])
                print(f"    >>> TARGET REACHED: {guidance}")

            if action == "ASK" and resp.get("question"):
                print(f"    >>> QUESTION: {resp['question']}")
                ans_r = requests.post(
                    f"{BASE_URL}/session/{sid}/answer",
                    json={"answer": "yes, it is on the left side"},
                    timeout=60,
                )
                if ans_r.ok:
                    ans_resp = ans_r.json()
                    print(f"    >>> ANSWER -> {ans_resp.get('action')}: {ans_resp.get('guidance', '')[:60]}")

        except Exception as e:
            result.errors.append(f"Photo {photo_path.name}: {e}")
            print(f"ERROR: {e}")

    result.total_time_s = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("TEST 1 RESULTS")
    print(f"{'=' * 70}")
    print(f"  Photos processed: {result.photos_used}")
    print(f"  Targets found: {len(result.targets_found)}/3")
    for t in result.targets_found:
        print(f"    - {t}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Total time: {result.total_time_s:.1f}s")
    return result


def run_test2(initial_photo: str):
    """Test 2: Blind navigation with only initial sofa photo."""
    import requests

    print("=" * 70)
    print("TEST 2: Blind Office Navigation (1 initial photo)")
    print("=" * 70)

    result = TestResult(test_name="Test 2: Blind Office")
    start_time = time.time()

    print(f"\n[1] Creating session: {GOAL}")
    try:
        r = requests.post(f"{BASE_URL}/session", json={"goal": GOAL}, timeout=60)
        r.raise_for_status()
        session = r.json()
        sid = session["session_id"]
        print(f"    Session ID: {sid}")
        print(f"    Goal objects: {session.get('goal_objects', [])}")
    except Exception as e:
        result.errors.append(str(e))
        print(f"    ERROR: {e}")
        return result

    initial = Path(initial_photo)
    print(f"\n[2] Uploading initial photo: {initial.name}")
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
        print(f"    Action: {resp.get('action')}")
        print(f"    Guidance: {resp.get('guidance')}")
    except Exception as e:
        result.errors.append(str(e))
        print(f"    ERROR: {e}")
        return result

    print("\n[3] Interactive navigation (take photos as directed)...")
    step = 1
    arrived_count = 0

    while arrived_count < 3 and step < 15:
        last = result.vlm_responses[-1]
        action = last.get("action", "MOVE")

        print(f"\n  --- Step {step} ---")
        print(f"  Action: {action}")
        print(f"  Guidance: {last.get('guidance', '')}")

        if action == "ARRIVED":
            arrived_count += 1
            result.targets_found.append(f"Target {arrived_count}: {last.get('guidance', '')[:50]}")
            print(f"  >>> TARGET {arrived_count}/3 REACHED!")
            if arrived_count >= 3:
                break

        if action == "ASK" and last.get("question"):
            print(f"  Question: {last['question']}")
            answer = input("  Your answer: ").strip() or "yes"
            try:
                r = requests.post(f"{BASE_URL}/session/{sid}/answer",
                                  json={"answer": answer}, timeout=60)
                if r.ok:
                    result.vlm_responses.append(r.json())
                    result.total_steps += 1
                    step += 1
                    continue
            except Exception as e:
                result.errors.append(str(e))

        photo_path = input(f"\n  Next photo path (or 'quit'): ").strip()
        if photo_path.lower() in ("quit", "q"):
            break
        if not Path(photo_path).exists():
            print(f"  File not found: {photo_path}")
            continue

        try:
            with open(photo_path, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/session/{sid}/photo",
                    files={"photo": (Path(photo_path).name, f, "image/jpeg")},
                    timeout=300,
                )
            r.raise_for_status()
            result.vlm_responses.append(r.json())
            result.photos_used += 1
            result.total_steps += 1
        except Exception as e:
            result.errors.append(str(e))
            print(f"  ERROR: {e}")

        step += 1

    result.total_time_s = time.time() - start_time
    print(f"\n{'=' * 70}")
    print("TEST 2 RESULTS")
    print(f"{'=' * 70}")
    print(f"  Photos used: {result.photos_used}")
    print(f"  Targets found: {len(result.targets_found)}/3")
    print(f"  Total time: {result.total_time_s:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Office Navigation Tests")
    parser.add_argument("--test", type=int, choices=[1, 2])
    parser.add_argument("--photo-dir", type=str)
    parser.add_argument("--initial-photo", type=str)
    parser.add_argument("--analyze", action="store_true")
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
        analyze_existing_detections()


if __name__ == "__main__":
    main()
