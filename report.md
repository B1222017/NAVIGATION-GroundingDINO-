# Navigation Model Report

**UniGoal vs WMNav — How They Work, How They Differ, and How to Fix the Topological Map**

---

## Table of Contents

1. [What is WMNav and How Does It Work](#1-what-is-wmnav-and-how-does-it-work)
2. [How WMNav Differs from UniGoal](#2-how-wmnav-differs-from-unigoal)
3. [Why the Topological Map is Inaccurate](#3-why-the-topological-map-is-inaccurate)
4. [How to Fix the Topological Map](#4-how-to-fix-the-topological-map)

---

## 1. What is WMNav and How Does It Work

### What It Is

WMNav (World Model Navigation) is a zero-shot object navigation system — meaning it can find any object in any unknown indoor environment without prior training on that environment. It was built for embodied AI robots, tested in simulated 3D environments (HM3D and MP3D benchmarks), and achieves **58.1% success rate** on HM3D.

The core idea is: **instead of just reacting to what the camera sees right now, WMNav first imagines what might be in each direction before moving there.** It uses a Vision-Language Model (VLM) as a "world model" — the same way a human mentally pictures what's likely around a corner before walking to it.

---

### How It Works — Step by Step

#### Step 1: Panoramic Capture

At every navigation step, the agent rotates and takes **6 photos at fixed angles** (30°, 90°, 150°, 210°, 270°, 330°). These 6 photos are stitched into one panoramic image so the agent has full 360° awareness — nothing is hidden by the camera angle.

```
30°   90°   150°   210°   270°   330°
 ↓     ↓      ↓      ↓      ↓      ↓
[img] [img]  [img]  [img]  [img]  [img]  →  panoramic image
```

---

#### Step 2: PredictVLM — Score Every Direction (0–10)

The panoramic image is sent to **PredictVLM** with this prompt:

> "The agent is looking for a BED. Score each direction (0–10) by how likely the bed is there."

The VLM returns a score for each of the 6 angles:

```
Angle 30:  2  — "A hallway, could lead somewhere"
Angle 90:  0  — "A wall, no path"
Angle 330: 10 — "A bed is clearly visible"
```

This is the **world model step** — the VLM is predicting future states without physically going there.

---

#### Step 3: Curiosity Value Map — Build Spatial Memory

These scores are projected from the egocentric camera view down onto a **top-down 2D grid map** (like a bird's eye view of the floor). Each cell in the grid stores a curiosity score (0–10):

- **10** = never visited, could have the goal
- **0** = visited and confirmed empty
- **Between** = VLM estimated some likelihood

The map is updated at every step using: `new_score = min(old_score, predicted_score)`

This means **once a region is confirmed empty, its score can only go down — never back up**. This prevents the agent from revisiting areas it already ruled out.

```
Initial map:   All cells = 10 (total uncertainty)
After step 1:  Direction 90° and 150° cells → 0 (walls, no path)
After step 2:  Direction 270° cells → 2 (door seen but unclear)
After step 3:  Direction 330° cells → 10 (bed found)
```

---

#### Step 4: PlanVLM — Choose Direction and Set Subtask

The curiosity scores are projected back onto the panoramic image. The direction with the **highest curiosity score** is selected.

The selected direction's image is sent to **PlanVLM**, which:
1. Checks if the previous subtask was completed
2. Decides whether the goal object is now visible
3. Generates the **next subtask** (e.g., "go down the hallway" → "enter the bedroom" → "approach the bed")

The previous subtask is always passed as context — this is the **cost feedback loop** that prevents hallucination from compounding across steps.

```
Step 1 subtask: "Find the corridor"
Step 2 subtask: "Go down the hallway" (previous subtask done)
Step 3 subtask: "Enter the bedroom" (hallway traversed)
Step 4 subtask: "Approach the bed" (bedroom found)
```

---

#### Step 5: Two-Stage Action Proposer — Move Precisely

**Stage 1 — Exploration:**
The navigable area in the selected direction image is sampled into candidate polar coordinate vectors (direction + distance). Vectors pointing toward already-explored regions are filtered out. **ActionVLM** picks the best action from the labeled candidates.

**Stage 2 — Goal Approaching (triggered when goal is visible):**
When PlanVLM sets `goal_visible = True`, the system switches to dense sampling near the goal. Length constraints are removed. **GoalVLM** picks the vector pointing most precisely at the goal's ground location.

Stopping is determined by **Euclidean distance** (not VLM judgment):
```
Stop if: distance_to_goal < 1.0 meters
```

---

### WMNav Full Pipeline Summary

```
[Rotate 6 angles] → [Panoramic Image]
        ↓
[PredictVLM] → scores per direction (0-10)
        ↓
[Project to top-down map] → update Curiosity Value Map
        ↓
[Project back to image] → highest-score direction selected
        ↓
[PlanVLM] → new subtask + goal_visible flag
        ↓
[Two-Stage Action Proposer]
  → Exploration: ActionVLM picks best polar action
  → Approaching: GoalVLM picks goal-pinpointing action
        ↓
[Execute action] → repeat until distance < threshold
```

---

## 2. How WMNav Differs from UniGoal

### Fundamental Difference

| | UniGoal | WMNav |
|---|---|---|
| **Core idea** | Build a map from photos, then navigate using that map | Predict future states before moving, update memory as you go |
| **When does it "think"** | Reacts to current photo only | Predicts what's ahead before moving |
| **Map type** | Topological graph (nodes = zones, edges = shared objects) | Curiosity Value Map (2D grid, each cell = predicted goal likelihood) |
| **Input** | Single smartphone photo per step | 6-angle panoramic RGB-D per step |
| **Perception** | GroundingDINO (object detector) + EasyOCR | Pure VLM reasoning — no separate detector |
| **Decision making** | Single VLM call → ARRIVED / MOVE / ASK | Three VLM calls: PredictVLM → PlanVLM → ActionVLM/GoalVLM |
| **Hallucination handling** | None — trusts VLM output | Previous subtask fed back as cost → self-correcting loop |
| **Stopping condition** | VLM says "ARRIVED" | Euclidean distance < 1.0m threshold |
| **Environment** | Real-world photos, any smartphone | Simulated 3D environments, needs RGB-D sensor |
| **OCR** | Yes — reads aisle signs, shelf labels | No OCR capability |
| **Map reuse** | Yes — build once, reuse for all sessions | No — rebuilds from scratch each session |
| **Privacy** | Fully local (Ollama) | Cloud API (Gemini) |

### Key Conceptual Difference Explained Simply

**UniGoal** works like a tourist with a printed map:
- Someone else walked the building, took photos, and built a map
- The tourist follows the map and reacts to what they see

**WMNav** works like a local who knows buildings well:
- They imagine "a bedroom is probably at the end of this corridor"
- They move toward their prediction, update their mental model, and self-correct if wrong
- No pre-built map needed — they reason from experience

### Where Each is Better

**WMNav is better at:**
- Navigating efficiently in completely unknown environments (no pre-built map needed)
- Not revisiting already-ruled-out areas (Curiosity Value Map memory)
- Self-correcting hallucinations (subtask cost feedback)
- Precise goal localization (two-stage approach)
- Reliable stopping (distance-based, not VLM-based)

**UniGoal is better at:**
- Real-world smartphone photos (no depth sensor needed)
- Reading signs and shelf labels (EasyOCR)
- Reusing a pre-built store map across many sessions
- Keeping customer data private (fully local inference)
- Letting users ask for clarification (ASK action)
- Human-readable map output (labeled graph PNG)

---

## 3. Why the Topological Map is Inaccurate

The topological map in UniGoal is built purely from **object co-occurrence** — zones are defined by which objects appear together in consecutive photos. This causes several accuracy problems in a real store:

### Problem 1: Jaccard Clustering Ignores Physical Location

The clustering algorithm in `generate_topomap.py:61` only looks at **which objects appear together**, not **where in the store those photos were taken**. Two physically distant locations can be clustered into the same zone if they have similar objects (e.g., two separate dairy aisles both contain "shelf", "refrigerator", "milk bottle").

```python
# generate_topomap.py:53-56
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)  # only object overlap — no spatial info
```

**Result**: The map shows Zone A connected to Zone C even if they are on opposite sides of the store, because they share common objects.

### Problem 2: Photos Must Be in Physical Walk Order

The clustering algorithm at `generate_topomap.py:68-75` assumes **consecutive photos are physically adjacent**:

```python
zones = [[photos[0]]]
for i in range(1, len(photos)):
    prev_set = photo_object_set(photos[i - 1])
    curr_set = photo_object_set(photos[i])
    sim = jaccard(prev_set, curr_set)
    if sim < sim_threshold:
        zones.append([photos[i]])   # new zone
    else:
        zones[-1].append(photos[i]) # same zone
```

Photos are sorted **alphabetically by filename** (`photos.sort(key=lambda p: p['filename'])`). If filenames don't reflect physical walking order (e.g., IMG_1001, IMG_1002... but you walked back and forth), the clusters will be completely wrong.

**Result**: A zone might contain photos from two different aisles because they happen to be alphabetically adjacent.

### Problem 3: Threshold Too Low — Zones Over-Merge

The default `sim_threshold=0.25` means: if two consecutive photos share even 25% of their detected objects, they are merged into the same zone. In a store where nearly every photo contains "shelf" and "product", this threshold is too low — the entire store collapses into a few large zones.

```python
# generate_topomap.py:61
def cluster_photos_into_zones(photos, sim_threshold=0.25, ...):
```

**Result**: Entire aisles are merged into one giant zone, making the map useless for navigation.

### Problem 4: Generic Objects Dominate Clustering

Objects like "shelf", "sign", "door", "product" appear in almost every store photo. These dominate the Jaccard similarity calculation and make every photo look similar to every other photo — again causing over-merging.

The `ubiquitous` filter in `build_edges()` only applies to **edge building**, not to the initial **zone clustering**:

```python
# generate_topomap.py:207-211 — only used for edges, not clustering
ubiquitous = {label for label, count in label_zone_count.items()
              if count > 0.6 * n}
```

**Result**: Ubiquitous objects like "shelf" blur zone boundaries during clustering.

### Problem 5: Graph Layout is Visual, Not Spatial

The graph is rendered using **Kamada-Kawai layout** (`generate_topomap.py:322`), which arranges nodes to minimize visual edge crossings. This layout has **nothing to do with physical geography**. Zone 3 might appear to the left of Zone 1 on the map even if it is physically to the right in the store.

```python
pos = nx.kamada_kawai_layout(G, scale=3.5)  # aesthetic, not geographic
```

**Result**: The map looks clean but the spatial relationships between zones are misleading.

### Problem 6: Non-Adjacent Edges Are Too Aggressive

The algorithm at `generate_topomap.py:214-222` connects zones that are far apart if they share ≥2 distinctive objects with Jaccard ≥ 0.35:

```python
for i in range(n):
    for j in range(i + 2, n):  # non-adjacent zones
        ...
        if len(shared) >= 2 and sim >= 0.35:
            add_edge(i, j, shared)
```

In a store, frozen food aisles and refrigerated dairy both have "refrigerator" and "cooler" — so they get connected with a shortcut edge even if they are physically far apart with no direct path.

**Result**: False shortcut edges that mislead navigation routing.

---

## 4. How to Fix the Topological Map

### Fix 1: Enforce Photo Walk Order with Proper Naming

**This is the most important fix.** Photos must be sorted in the physical order they were taken, not alphabetically.

**Option A — Rename photos before testing:**
Name photos sequentially in the order you physically walked: `001.jpg`, `002.jpg`, `003.jpg` etc. The current alphabetical sort will then match physical walk order.

**Option B — Add a sort by timestamp:**
```python
# generate_topomap.py — replace the sort line
import os
photos.sort(key=lambda p: os.path.getmtime(p['filename']))
# or sort by EXIF timestamp if available
```

**Option C — Add a `sequence` field to the detection JSON** and sort by it in `load_detections()`.

---

### Fix 2: Filter Ubiquitous Objects Before Clustering (Not Just Edge Building)

Remove generic store objects ("shelf", "product", "sign", "wall") from the similarity calculation used for zone clustering.

```python
# generate_topomap.py — modify cluster_photos_into_zones()

STORE_GENERIC_OBJECTS = {
    "shelf", "shelving", "rack", "product", "item", "wall",
    "floor", "ceiling", "sign", "label", "door", "window",
    "light", "tile", "display"
}

def photo_object_set(photo: dict, exclude: set = None) -> set:
    labels = set(obj['label'] for obj in photo['objects'])
    if exclude:
        labels = {l for l in labels if l.lower() not in exclude}
    return labels

def cluster_photos_into_zones(photos, sim_threshold=0.25, min_zone_size=2):
    ...
    for i in range(1, len(photos)):
        prev_set = photo_object_set(photos[i - 1], exclude=STORE_GENERIC_OBJECTS)
        curr_set = photo_object_set(photos[i],     exclude=STORE_GENERIC_OBJECTS)
        sim = jaccard(prev_set, curr_set)
        ...
```

This forces zone boundaries to be defined by **distinctive objects** (milk, bread, produce) rather than objects that appear everywhere (shelf, sign).

---

### Fix 3: Raise the Similarity Threshold for Stores

The default `sim_threshold=0.25` is too low for retail environments. Raise it so zones are more granular:

```python
# generate_topomap.py — change default
def cluster_photos_into_zones(photos, sim_threshold=0.40, min_zone_size=2):
```

Or pass it from the command line:

```bash
python generate_topomap.py --sim-threshold 0.40
```

A higher threshold (0.40–0.50) means the model is stricter about what counts as "the same zone" — producing more, smaller, more accurate zones instead of a few giant ones.

---

### Fix 4: Add Label Normalization Before Clustering

Normalize synonyms before computing Jaccard so "milk bottle" and "milk carton" count as the same object:

```python
# server/config.py
LABEL_SYNONYMS = {
    "milk": ["milk", "milk bottle", "milk carton", "whole milk"],
    "shelf": ["shelf", "shelving", "rack", "display rack"],
    "refrigerator": ["refrigerator", "fridge", "cooler", "freezer"],
    "bread": ["bread", "loaf", "bread loaf"],
    ...
}

def normalize_label(label: str) -> str:
    for canonical, variants in LABEL_SYNONYMS.items():
        if any(v in label.lower() for v in variants):
            return canonical
    return label.lower()
```

Apply in `photo_object_set()`:

```python
def photo_object_set(photo: dict) -> set:
    return set(normalize_label(obj['label']) for obj in photo['objects'])
```

---

### Fix 5: Raise the Non-Adjacent Edge Threshold

The current Jaccard ≥ 0.35 for non-adjacent edges is too low — it creates too many false shortcut edges. Raise to 0.50 and require more shared objects:

```python
# generate_topomap.py:219-222 — tighten conditions
if len(shared) >= 3:       # was 2 — require more shared distinctive objects
    sim = jaccard(labels_i, labels_j)
    if sim >= 0.50:        # was 0.35 — stricter similarity for non-adjacent
        add_edge(i, j, shared)
```

---

### Fix 6: Use Grid/Floor Plan Layout Instead of Kamada-Kawai

If you know the rough physical layout of the store (or can estimate it from photo sequence), assign positions manually or use a sequential layout:

**Option A — Sequential row layout (simplest fix):**
```python
# generate_topomap.py — replace kamada-kawai
def sequential_layout(G, zones_per_row=5):
    pos = {}
    nodes = sorted(G.nodes())
    for i, node in enumerate(nodes):
        row = i // zones_per_row
        col = i % zones_per_row
        # Alternate row direction (snake layout, like store aisles)
        if row % 2 == 1:
            col = zones_per_row - 1 - col
        pos[node] = (col * 2.0, -row * 2.0)
    return pos

pos = sequential_layout(G, zones_per_row=5)  # replace kamada_kawai_layout
```

This places zones in a snake pattern matching how a store is physically walked — far more accurate than force-directed layout.

**Option B — Use spring layout with sequential position hints:**
```python
# Seed positions based on photo sequence before running spring layout
initial_pos = {i: (i % 5, -(i // 5)) for i in G.nodes()}
pos = nx.spring_layout(G, pos=initial_pos, fixed=None, iterations=50, seed=42)
```

---

### Fix 7: Add Zone Name Override from OCR

When EasyOCR detects a clear aisle sign in the photos of a zone (e.g., "DAIRY", "AISLE 7"), use that text as the zone label instead of the auto-generated one:

```python
# generate_topomap.py — in build_zone_info() or generate_zone_label()
def generate_zone_label(zid, info, ocr_results=None, used_names=None):
    # If OCR found a clear aisle sign in this zone's photos, use it
    if ocr_results:
        for photo in info['photos']:
            texts = ocr_results.get(photo, [])
            for text, confidence in texts:
                if confidence > 0.7 and len(text) > 3:
                    clean = text.strip().title()
                    if clean not in used_names:
                        used_names.add(clean)
                        return f"Zone {zid}\n{clean}\n({len(info['photos'])} photos)"
    # Fallback to object-based label
    ...
```

This produces zone names like "Zone 3 — Dairy" instead of "Zone 3 — Refrigerator Area" — far more meaningful for navigation.

---

### Summary of Fixes

| Fix | File | Problem Solved | Priority |
|---|---|---|---|
| Sort photos by walk order | `generate_topomap.py` | Zones contain physically unrelated photos | Critical |
| Filter generic objects from clustering | `generate_topomap.py` | Shelf/sign blur all zone boundaries | High |
| Raise sim threshold to 0.40 | `generate_topomap.py` | Entire store collapses into few zones | High |
| Label normalization | `server/config.py`, `generate_topomap.py` | Same object splits zones | High |
| Tighten non-adjacent edge threshold | `generate_topomap.py` | False shortcut edges | Medium |
| Sequential/snake layout | `generate_topomap.py` | Map positions don't match physical layout | Medium |
| OCR zone naming | `generate_topomap.py` | Zone labels don't reflect real aisle names | Medium |

---

### Recommended Testing Approach

After applying fixes, validate the map accuracy by:

1. **Walk the store in a single continuous path** (don't backtrack) and number photos 001, 002, 003... in order.
2. Run `python generate_topomap.py --sim-threshold 0.40`
3. Check: does the number of zones roughly match the number of distinct areas in the store?
4. Check: do the zone labels match the aisle signs you photographed?
5. Check: are adjacent zones in the map also adjacent in the physical store?

If zones are too few → lower threshold (try 0.35).
If zones are too many → raise threshold (try 0.50) or raise `min_zone_size`.

---

*Report generated: 2026-05-27*
*UniGoal codebase: C:\Users\user\Downloads\Navigation-main\Navigation-main*
*WMNav paper: arXiv:2503.02247v4*
