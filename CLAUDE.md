# UniGoal — AI Instructions

This is **UniGoal**, being evolved into a **hybrid in-store shopping navigation app**.

## App Context

- **Use case**: Customer navigates inside a retail store to find a product (e.g., "find milk", "find bread").
- **Input**: Smartphone photos only — no depth sensor, no RGB-D, no known pose.
- **Testing phase**: All store environment photos are available upfront — used to pre-build the topological map once.
- **Runtime phase**: Customer uploads one photo per step and receives turn-by-turn guidance.
- **Architecture**: UniGoal's infrastructure (map, OCR, real photos, local inference) + WMNav's reasoning (subtask decomposition, curiosity scoring, two-stage localization).

## What to Keep from UniGoal (Do NOT Remove)

| Feature | Why Keep |
|---|---|
| GroundingDINO object detection | Finds specific products on shelves — essential for in-store |
| EasyOCR sign reading | Reads aisle signs ("Aisle 7 — Dairy"), shelf tags — critical for stores |
| Batch mode map building | Pre-build store map from all environment photos once, reuse per session |
| Topological graph (NetworkX) | Store zones (produce, dairy, frozen) map perfectly to node-edge graph |
| Ollama local inference | No customer photos sent to cloud — privacy requirement |
| ASK action | Ask customer "Do you see the frozen foods sign?" when uncertain |

## What to Add from WMNav (Do NOT Add Panoramic/RGB-D)

| Feature | Why Add | WMNav Source |
|---|---|---|
| Subtask decomposition | "Find milk" → "find dairy section" → "find refrigerated aisle" → "find milk shelf" | PlanVLM + cost feedback |
| Curiosity scores on zones | Stop routing to aisles already confirmed empty | Curiosity Value Map |
| VLM zone scoring before routing | Score neighboring zones by goal likelihood before moving | PredictVLM |
| Two-stage goal localization | Stage 1: find aisle. Stage 2: pinpoint shelf position | Two-stage Action Proposer |
| User-confirmed ARRIVED | Don't declare success until customer confirms they see the product | Reliable stop condition |

## What to NOT Add from WMNav

| Feature | Why Skip |
|---|---|
| Panoramic 360° capture | Customers won't rotate 360° in a store aisle |
| RGB-D depth requirement | Smartphone has no depth sensor |
| Pure VLM (no detector) | GroundingDINO is better for finding specific products on shelves |
| Gemini cloud API | Privacy — customer photos must stay local |

---

## Key Files

| File | Purpose |
|------|---------|
| `server/perception.py` | GroundingDINO wrapper — object/product detection |
| `server/config.py` | Thresholds: BOX=0.30, TEXT=0.25, TOP_K=5 |
| `server/batch_mapper.py` | Batch photo processing + Jaccard zone clustering |
| `generate_topomap.py` | Topological map generation (clustering, graph, render) |
| `server/vlm.py` | Ollama llama3.2-vision client — navigation decisions |
| `server/session.py` | Per-session state (goal, history, node, etc.) |
| `server/topomap.py` | Runtime topological map (node-edge graph) |
| `server/goal_decomposer.py` | Goal text → list of GroundingDINO class names |
| `server/ocr.py` | EasyOCR for aisle sign / shelf label extraction |
| `server/annotator.py` | Draw bounding boxes + OCR regions on photos |
| `server/server.py` | FastAPI REST API |

---

## Known Problems (Priority Order)

### 1. Object Detection Errors (BIGGEST ISSUE)
- **Root cause A**: Thresholds too low (`BOX_THRESHOLD=0.30`) → false positives on store shelves.
- **Root cause B**: Only top-5 detections returned → misses relevant products in dense shelf scenes.
- **Root cause C**: Label string mismatch ("milk bottle" vs "milk") breaks Jaccard zone clustering.
- **Root cause D**: No VLM verification — GroundingDINO false positives passed through unchecked.
- **Root cause E**: No feedback loop — if detection fails, keeps routing to the wrong aisle.
- **Root cause F**: Goal decomposer generates too few / too generic class names → low recall.
- **Root cause G**: OCR results not used to confirm or reject detections — wasted signal.

### 2. Hallucination in VLM Decisions
- VLM receives a single prompt with no correction mechanism across steps.
- No subtask decomposition — VLM tries to solve "find milk" in one step instead of breaking it down.

### 3. Inefficient Exploration
- Shortest-path routing ignores whether an aisle was already visited and confirmed empty.
- No curiosity/prediction — customer is sent back to already-searched zones.

### 4. Unreliable ARRIVED Detection
- VLM decides ARRIVED from a photo — bad at judging proximity.
- Customer may stop too far from the actual product shelf.

---

## Improvements to Implement

Implement in this order. Tasks 1–5 fix object detection (biggest issue first). Tasks 6–9 add WMNav reasoning on top.

---

### TASK 1 — Raise & Adaptive Thresholds (Quickest Win)

**Files**: `server/config.py`, `server/perception.py`

**What**: Current thresholds are too low for busy store shelves — too many false positives. Raise primary thresholds and retry with lower fallback only if nothing is detected.

```python
# server/config.py
BOX_THRESHOLD = 0.40          # was 0.30 — fewer false positives on shelves
TEXT_THRESHOLD = 0.30         # was 0.25
BOX_THRESHOLD_FALLBACK = 0.30 # retry if nothing found at primary threshold
TOP_K_DETECTIONS = 10         # was 5 — stores have more objects per scene
```

```python
# server/perception.py
def detect(image, classes):
    results = _run_groundingdino(image, classes, BOX_THRESHOLD, TEXT_THRESHOLD)
    if len(results) == 0:
        results = _run_groundingdino(image, classes, BOX_THRESHOLD_FALLBACK, TEXT_THRESHOLD)
    return results[:TOP_K_DETECTIONS]
```

---

### TASK 2 — Store-Specific Label Normalization for Jaccard Clustering

**Files**: `server/config.py`, `server/batch_mapper.py`, `generate_topomap.py`

**What**: GroundingDINO labels the same object differently across photos ("milk bottle" vs "milk carton" vs "milk"). This breaks Jaccard clustering — same physical zone gets split into two nodes. Normalize all labels before computing Jaccard similarity.

```python
# server/config.py
LABEL_SYNONYMS = {
    # Store sections
    "dairy": ["dairy", "milk section", "refrigerated dairy", "dairy aisle"],
    "produce": ["produce", "vegetables", "fruits", "fresh produce", "greens"],
    "frozen": ["frozen", "frozen foods", "freezer section", "frozen aisle"],
    "bakery": ["bakery", "bread section", "baked goods", "bread aisle"],
    "beverages": ["beverages", "drinks", "sodas", "water aisle", "juice"],
    "checkout": ["checkout", "cashier", "register", "cash register", "till"],
    # Common products
    "milk": ["milk", "milk bottle", "milk carton", "whole milk", "skim milk"],
    "bread": ["bread", "loaf", "bread loaf", "sandwich bread"],
    "refrigerator": ["refrigerator", "fridge", "cooler", "refrigerated case", "freezer"],
    "shelf": ["shelf", "shelving", "rack", "display rack", "store shelf"],
    "cart": ["cart", "shopping cart", "trolley", "basket"],
    "sign": ["sign", "aisle sign", "store sign", "label", "price tag"],
}

def normalize_label(label: str) -> str:
    label_lower = label.lower()
    for canonical, variants in LABEL_SYNONYMS.items():
        if any(v in label_lower for v in variants):
            return canonical
    return label_lower
```

Apply `normalize_label()` to every detected label in `server/batch_mapper.py` before Jaccard computation.

---

### TASK 3 — VLM Verification of Goal Detections (Biggest Accuracy Gain)

**File**: `server/perception.py`

**What**: When GroundingDINO says it found the goal product, crop that bounding box region and ask the VLM to confirm before trusting it. Store shelves are visually dense — false positives are common.

```python
def verify_goal_detection(image_pil, box_xyxy, goal_label: str, vlm_client) -> bool:
    """Crop the detection box and ask VLM to confirm it is the goal product."""
    x1, y1, x2, y2 = [int(c) for c in box_xyxy]
    cropped = image_pil.crop((x1, y1, x2, y2))
    prompt = f"Does this image clearly show '{goal_label}'? Answer only: yes or no."
    response = vlm_client.ask_simple(cropped, prompt)
    return "yes" in response.lower()
```

Call this in `server/session.py` after getting detections. Only set `goal_visible=True` if verification passes.

---

### TASK 4 — Better Goal Decomposition Prompts (Increases Recall)

**File**: `server/goal_decomposer.py`

**What**: The goal decomposer generates class names for GroundingDINO to search. If it generates too few or too generic names, detections miss. Make the prompt store-aware so it generates product variants, section names, and landmark objects.

```python
STORE_DECOMPOSE_PROMPT = """
A customer wants to find: "{goal}" in a retail store.
Generate a list of object labels that GroundingDINO should look for.
Include:
- The exact product name and common variants (e.g. "milk", "milk bottle", "milk carton")
- The store section it belongs to (e.g. "dairy section", "refrigerated aisle")
- Nearby landmark objects (e.g. "refrigerator", "cooler", "freezer door")
- Aisle signage words (e.g. "dairy sign", "aisle 5 sign")
Return as JSON list: ["label1", "label2", ...]
"""
```

Replace the existing prompt in `goal_decomposer.py` with this store-specific version.

---

### TASK 5 — OCR Confirmation of Detections (Store-Specific Signal)

**File**: `server/session.py`

**What**: EasyOCR already runs on every photo. Use its output as a second confirmation signal — if the aisle sign text matches the product's section, boost confidence even if GroundingDINO score is borderline.

```python
SECTION_KEYWORDS = {
    "milk": ["dairy", "milk", "refrigerated"],
    "bread": ["bakery", "bread", "baked"],
    "eggs": ["dairy", "eggs", "refrigerated"],
    "frozen": ["frozen", "freezer"],
    "produce": ["produce", "vegetables", "fruit", "fresh"],
    "beverages": ["drinks", "beverages", "sodas", "juice", "water"],
}

def confirm_zone_by_ocr(ocr_texts: list[str], goal: str) -> bool:
    """Return True if any OCR sign text matches the goal product's store section."""
    relevant_keywords = SECTION_KEYWORDS.get(goal.lower(), [goal.lower()])
    return any(
        any(kw in text.lower() for kw in relevant_keywords)
        for text in ocr_texts
    )
```

If `confirm_zone_by_ocr()` returns True, treat the current zone as a strong candidate even if GroundingDINO confidence is at fallback threshold.

---

### TASK 6 — Subtask Decomposition with Feedback (Highest Impact for Navigation)

**Files**: `server/vlm.py`, `server/session.py`, `server/models.py`

**What**: Break "find {product}" into store-specific subtasks at each step. Feed the previous subtask back as context to prevent hallucination compounding across long multi-aisle journeys.

Add fields to session state:

```python
# server/session.py
@dataclass
class Session:
    goal: str
    subtask: str = ""          # e.g. "find the dairy section"
    goal_visible: bool = False # product is visible in current photo
    stage: str = "explore"     # "explore" | "approach"
    subtask_done: bool = False
    ...
```

Split the VLM call in `server/vlm.py` into two sequential calls:

**Call 1 — PlanVLM** (subtask planning):
```
You are helping a customer navigate a retail store to find: {goal}.
Previous subtask: "{last_subtask}" — completed: {subtask_done}
Detected objects in current photo: {detections}
Aisle signs / shelf labels seen: {ocr_text}

Tasks:
(1) Is the previous subtask completed?
(2) Is the goal product visible in the photo right now?
(3) What is the next subtask? (be store-specific: aisle name, section, shelf)

Return JSON: {"subtask": "...", "goal_visible": true/false, "subtask_done": true/false}
```

**Call 2 — ActionVLM** (movement decision, uses subtask from Call 1):
```
Goal: {goal}. Current subtask: "{subtask}".
Detected: {detections}. Signs: {ocr_text}.
Return JSON: {"action": "ARRIVED|MOVE|ASK", "guidance": "...", "question": "..."}
```

---

### TASK 7 — Curiosity Scores on Store Zone Nodes

**Files**: `server/topomap.py`, `server/session.py`, `generate_topomap.py`

**What**: Track per-zone likelihood that the goal product is there. Once an aisle is searched and confirmed empty, stop routing the customer back there.

```python
# server/topomap.py
@dataclass
class TopoNode:
    node_id: str
    label: str
    objects: list[str]
    photos: list[str]
    curiosity_score: float = 10.0  # 0=confirmed empty, 10=not yet visited
    visited: bool = False
```

When a zone is visited and goal NOT found:

```python
def mark_zone_empty(graph, node_id: str):
    graph.nodes[node_id]['curiosity_score'] = 0.0
    graph.nodes[node_id]['visited'] = True
    for neighbor in graph.neighbors(node_id):
        current = graph.nodes[neighbor].get('curiosity_score', 10.0)
        graph.nodes[neighbor]['curiosity_score'] = current * 0.7
```

Change pathfinding to route toward highest curiosity unvisited zone:

```python
def next_zone_to_visit(graph, current_node) -> str:
    unvisited = [n for n in graph.nodes if not graph.nodes[n].get('visited', False)]
    if not unvisited:
        return None
    return max(unvisited, key=lambda n: graph.nodes[n].get('curiosity_score', 5.0))
```

---

### TASK 8 — VLM Zone Scoring Before Routing

**Files**: `server/vlm.py`, `server/session.py`

**What**: Before routing to the next zone, ask the VLM to score neighboring zones by how likely they are to contain the goal product. Multiply with curiosity scores from TASK 7 for final ranking.

```python
PREDICT_PROMPT = """
A customer is looking for: {goal} in a retail store.
They are currently in zone '{current_zone}' which contains: {current_objects}.
Aisle signs visible: {ocr_text}.

Rate each neighboring zone by likelihood of containing '{goal}' (0-10):
{zone_descriptions}

Return JSON: {"zone_id": score, ...}
"""

def predict_zone_scores(current_zone, neighbor_zones, goal, ocr_text, vlm_client) -> dict:
    zone_desc = "\n".join(
        f"- {z.node_id} ({z.label}): contains {z.objects[:5]}"
        for z in neighbor_zones
    )
    prompt = PREDICT_PROMPT.format(
        goal=goal,
        current_zone=current_zone.label,
        current_objects=current_zone.objects[:5],
        ocr_text=ocr_text,
        zone_descriptions=zone_desc
    )
    response = vlm_client.ask_text_only(prompt)
    return json.loads(response)

def rank_zones(graph, neighbor_zones, vlm_scores: dict) -> list:
    """Combine VLM prediction × curiosity score for final zone ranking."""
    ranked = []
    for zone in neighbor_zones:
        vlm_score = vlm_scores.get(zone.node_id, 5.0)
        curiosity = graph.nodes[zone.node_id].get('curiosity_score', 10.0)
        ranked.append((zone.node_id, vlm_score * curiosity))
    return sorted(ranked, key=lambda x: x[1], reverse=True)
```

---

### TASK 9 — Two-Stage Goal Localization with User Confirmation

**Files**: `server/session.py`, `server/vlm.py`

**What**: When the product is detected (stage 1 complete), switch to approach mode with a focused prompt. Require customer to confirm they are physically next to the product before declaring ARRIVED.

```python
# server/session.py — switch to approach when goal is visible
if vlm_plan_response.get("goal_visible"):
    session.stage = "approach"
```

In `server/vlm.py`, use a different action prompt when `session.stage == "approach"`:

```
The customer can see '{goal}' in their photo.
Instruct them to move DIRECTLY toward it without turning.
Tell them to get as close as possible to the shelf.
Do NOT say ARRIVED yet.
Return JSON: {"action": "MOVE", "guidance": "Walk straight toward the {goal} on your left/right/ahead."}
```

After 2 approach steps, trigger user confirmation instead of VLM-decided ARRIVED:

```python
if session.stage == "approach" and session.approach_steps >= 2:
    return {
        "action": "ASK",
        "question": f"Can you reach the {session.goal} on the shelf in front of you? (yes / no)"
    }
# On yes → ARRIVED. On no → continue approach.
```

---

## Full Runtime Flow (After All Tasks Implemented)

```
TESTING PHASE (run once per store):
  python generate_topomap.py --photos ./store_photos
  → GroundingDINO detects products + shelves (TASK 1: adaptive thresholds)
  → Labels normalized before Jaccard clustering (TASK 2)
  → EasyOCR reads all aisle signs
  → Zones built with curiosity_score=10.0 on all nodes (TASK 7)
  → Map saved and reused for all customer sessions

RUNTIME PHASE (per customer session):
  POST /session  {"goal": "find milk"}
  → goal_decomposer generates store-aware class list (TASK 4)
  → session initialized: subtask="", stage="explore"

  Per step — POST /session/{id}/photo
  → GroundingDINO detects objects (TASK 1: adaptive thresholds)
  → Labels normalized (TASK 2)
  → EasyOCR reads aisle signs
  → OCR confirmation check (TASK 5)
  → Call 1 PlanVLM: determines subtask + goal_visible (TASK 6)
  → If goal_visible → VLM crop verification (TASK 3) → switch to approach stage (TASK 9)
  → VLM scores neighboring zones (TASK 8) × curiosity scores (TASK 7)
  → Route to highest-ranked unvisited zone
  → Call 2 ActionVLM: returns MOVE guidance with subtask context (TASK 6)
  → If approach stage → focused movement prompt → user confirmation (TASK 9)
  → On confirmed yes → ARRIVED → session complete
```

---

## Topological Map Problems & Fixes

The tested topological map is **inaccurate in physical location** — zones don't match real store areas. Root causes and fixes below. Do NOT change code yet — implement these fixes when continuing on another device.

### Root Causes of Map Inaccuracy

| # | Problem | Location in Code |
|---|---|---|
| 1 | Photos sorted alphabetically, not by walk order | `generate_topomap.py:43` — `photos.sort(key=lambda p: p['filename'])` |
| 2 | Generic objects (shelf, sign, wall) dominate clustering — blurs zone boundaries | `generate_topomap.py:48-56` — `photo_object_set()` includes all labels |
| 3 | `sim_threshold=0.25` too low — entire store collapses into few large zones | `generate_topomap.py:61` — default threshold |
| 4 | Same object labeled differently ("milk bottle" vs "milk carton") splits zones | `generate_topomap.py:53` — no normalization before Jaccard |
| 5 | Non-adjacent edge threshold too loose (Jaccard ≥ 0.35, shared ≥ 2) — creates false shortcut edges | `generate_topomap.py:219-222` |
| 6 | Kamada-Kawai layout is aesthetic only — node positions don't match physical store layout | `generate_topomap.py:322` |

### TASK 10 — Fix Photo Walk Order (MOST CRITICAL)

**File**: `generate_topomap.py:43`

**What**: Photos must be sorted in the physical order they were taken, not alphabetically. If filenames don't reflect walk order, zones will contain physically unrelated photos.

**Option A** — Rename photos before running: name them `001.jpg`, `002.jpg`, `003.jpg` in the exact order walked. Alphabetical sort will then match physical order.

**Option B** — Sort by file modification timestamp instead of filename:
```python
# generate_topomap.py — replace line 43
import os
photos.sort(key=lambda p: os.path.getmtime(p['filename']))
```

**Option C** — Add a `sequence` field to `detections.json` during batch mapping and sort by it in `load_detections()`.

---

### TASK 11 — Filter Generic Objects Before Clustering

**File**: `generate_topomap.py` — `photo_object_set()` function

**What**: Objects like "shelf", "sign", "wall", "product" appear in almost every store photo. Exclude them from the Jaccard similarity used for zone clustering so boundaries are defined by distinctive objects (milk, bread, produce) instead.

```python
# generate_topomap.py
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

# In cluster_photos_into_zones(), pass exclude=STORE_GENERIC_OBJECTS
# to both photo_object_set() calls
```

---

### TASK 12 — Raise Similarity Threshold for Stores

**File**: `generate_topomap.py:61`

**What**: Default `sim_threshold=0.25` is too low for retail — produces too few, too-large zones. Raise to 0.40.

```python
# generate_topomap.py
def cluster_photos_into_zones(photos, sim_threshold=0.40, min_zone_size=2):
```

Or run from command line:
```bash
python generate_topomap.py --sim-threshold 0.40
```

If zones are still too few → try 0.50. If too many → drop back to 0.35.

---

### TASK 13 — Apply Label Normalization in Topomap Clustering

**File**: `generate_topomap.py` — `photo_object_set()`

**What**: Same normalization from TASK 2 must also be applied inside `generate_topomap.py` so the clustering uses canonical labels. Import `normalize_label` from `server/config.py` or duplicate the function locally.

```python
# generate_topomap.py
def photo_object_set(photo: dict, exclude: set = None) -> set:
    labels = set(normalize_label(obj['label']) for obj in photo['objects'])
    if exclude:
        labels = {l for l in labels if l not in exclude}
    return labels
```

---

### TASK 14 — Tighten Non-Adjacent Edge Threshold

**File**: `generate_topomap.py:219-222`

**What**: Current settings (shared ≥ 2 objects, Jaccard ≥ 0.35) create too many false shortcut edges between physically distant zones. Tighten both conditions.

```python
# generate_topomap.py — in build_edges()
if len(shared) >= 3:    # was 2
    sim = jaccard(labels_i, labels_j)
    if sim >= 0.50:     # was 0.35
        add_edge(i, j, shared)
```

---

### TASK 15 — Replace Kamada-Kawai with Snake/Sequential Layout

**File**: `generate_topomap.py:322`

**What**: Kamada-Kawai arranges nodes to minimize visual edge crossings — not to match physical geography. Replace with a snake layout that mirrors how store aisles are physically walked.

```python
# generate_topomap.py — replace kamada_kawai_layout
def sequential_layout(G, zones_per_row=5):
    pos = {}
    nodes = sorted(G.nodes())
    for i, node in enumerate(nodes):
        row = i // zones_per_row
        col = i % zones_per_row
        if row % 2 == 1:           # alternate direction per row (snake)
            col = zones_per_row - 1 - col
        pos[node] = (col * 2.0, -row * 2.0)
    return pos

pos = sequential_layout(G, zones_per_row=5)  # replace kamada_kawai_layout line
```

Adjust `zones_per_row` to match the number of aisles in the store.

---

### TASK 16 — Use OCR Text as Zone Label

**File**: `generate_topomap.py` — `generate_zone_label()`

**What**: If EasyOCR detected a clear aisle sign in a zone's photos (e.g., "DAIRY", "AISLE 7"), use that as the zone label instead of the auto-generated object-based name. Requires passing OCR results into the map generation pipeline.

```python
# generate_topomap.py — in generate_zone_label()
def generate_zone_label(zid, info, ocr_results=None, used_names=None):
    if ocr_results:
        for photo in info['photos']:
            texts = ocr_results.get(photo, [])
            for text, confidence in texts:
                if confidence > 0.7 and len(text) > 3:
                    clean = text.strip().title()
                    if clean not in used_names:
                        used_names.add(clean)
                        return f"Zone {zid}\n{clean}\n({len(info['photos'])} photos)"
    # fallback to existing object-based label logic
    ...
```

---

### TASK 17 — Add Topomap Visualization via Pyvis (Interactive HTML)

**What**: Replace the current static PNG output with an interactive HTML file using Pyvis. This lets you drag zones around to verify they match the physical store layout — directly helps debug map accuracy. No new model or data needed.

**Install**: `pip install pyvis`

**Usage** (add as alternative output in `generate_topomap.py`):
```python
from pyvis.network import Network

def render_interactive_map(G, zone_info, goal_zones, output="topomap.html"):
    net = Network(height="900px", width="100%", bgcolor="#0d1117", font_color="white")
    for node in G.nodes(data=True):
        nid, data = node
        label = data.get('label', f'Zone {nid}')
        color = "#f85149" if nid in goal_zones else "#21262d"
        net.add_node(nid, label=label, color=color, title=str(zone_info[nid]['all_objects']))
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, title=', '.join(data.get('shared', [])))
    net.show(output)
    print(f"Saved interactive map: {output}")
```

Alternatives to Pyvis for topomap visualization:
- **Cytoscape.js** — best for embedding in the shopping app frontend (takes JSON nodes/edges directly)
- **Gephi** (desktop app) — import JSON/GraphML, drag nodes manually to match real floor plan, free
- **D3.js** — most flexible for custom store map UI
- **Vis.js Network** — simpler than D3, interactive, import JSON directly
- **Plotly Graph Objects** — interactive HTML with hover tooltips, integrates with FastAPI
- **Neo4j** — graph database with visual browser, good if store map grows large

---

### Map Fix Priority Order

| Task | Fix | Impact |
|---|---|---|
| TASK 10 | Sort photos by walk order | Critical — everything else depends on this |
| TASK 11 | Filter generic objects from clustering | High |
| TASK 12 | Raise sim threshold to 0.40 | High |
| TASK 13 | Label normalization in topomap clustering | High |
| TASK 14 | Tighten non-adjacent edge threshold | Medium |
| TASK 15 | Snake/sequential layout | Medium |
| TASK 16 | OCR-based zone labels | Medium |
| TASK 17 | Pyvis interactive HTML output | Medium — helps validate fixes visually |

---

## How to Run

```bash
# Build store map (testing phase — run once with all store photos)
python generate_topomap.py --sim-threshold 0.40

# Start navigation server (runtime phase)
cd server
uvicorn server:app --reload
```

## Environment

- Python 3.12
- Windows 10 / WSL Ubuntu
- Ollama running locally with llama3.2-vision
- GroundingDINO weights: `weights/groundingdino_swint_ogc.pth`

---

## Android Frontend Integration

The Android app (Android Studio) connects to the FastAPI server via HTTP. The app is a thin client — it only captures photos and displays results. All AI processing (GroundingDINO, Ollama, OCR, map routing) runs on the server.

### Architecture

```
Android App (Android Studio)
        ↕  HTTPS/HTTP
FastAPI Server (UniGoal — GroundingDINO + Ollama + NetworkX)
```

### Android Tech Stack

| Need | Library |
|---|---|
| HTTP requests to FastAPI | Retrofit2 + OkHttp |
| Camera capture | CameraX (Jetpack) |
| Display annotated photo from server | Glide or Coil |
| JSON parsing | Gson or Moshi |
| Async/background tasks | Kotlin Coroutines |

### API Calls the Android App Makes

| Action | Call |
|---|---|
| Customer types product name | `POST /session` — body: `{"goal": "find milk"}` → returns `session_id` |
| Customer takes a photo | `POST /session/{id}/photo` — multipart image → returns action + guidance |
| Customer answers ASK question | `POST /session/{id}/answer` — body: `{"answer": "yes"}` |

### Navigation Loop Flow

```
Customer types "find milk"
        ↓
POST /session → get session_id
        ↓
Customer points camera at current aisle → tap Scan
        ↓
POST /session/{id}/photo (compressed to 640x480 before upload)
        ↓
Server returns:
  action: MOVE  → show guidance text + annotated photo → customer moves → repeat
  action: ASK   → show question + Yes/No buttons → POST /session/{id}/answer → continue
  action: ARRIVED → show success screen → end session
```

### Important Android Notes

- **Compress photo before upload**: resize to 640×480 before sending — reduces upload time and server processing time
  ```kotlin
  val compressed = Bitmap.createScaledBitmap(bitmap, 640, 480, true)
  ```
- **Show loading spinner**: server takes 2–5 seconds per photo — always show a loading state between upload and response
- **Store session_id**: save it after `POST /session` and attach to every subsequent request

---

## Deployment Path (How to Host the Server)

### Stage 1 — Development & Testing (Now)
Use **Ngrok** to expose your local FastAPI server to the internet — no cloud needed, works immediately.

```bash
# 1. Start FastAPI server locally
cd server && uvicorn server:app --reload

# 2. In a second terminal, expose it
ngrok http 8000

# 3. Ngrok gives you a public URL like:
#    https://abc123.ngrok.io
```

Set in Android app:
```kotlin
val BASE_URL = "https://abc123.ngrok.io/"
```

- Free tier available
- URL changes every time you restart Ngrok (use paid plan for fixed URL)
- Good for demos and testing from any internet connection

---

### Stage 2 — Store Pilot / Demo
Run the server on a **dedicated machine inside the store** connected to store WiFi.

```
Store WiFi Router
    ├── Store PC/Server  → FastAPI running (fixed local IP e.g. 192.168.1.10)
    └── Customer Phone   → connects to store WiFi → calls 192.168.1.10:8000
```

- Customer must connect to store WiFi
- No data leaves the store — fully private
- No cloud cost
- Find the server's local IP: run `ipconfig` (Windows) or `ifconfig` (Linux) on the server machine

---

### Stage 3 — Full Production (Cloud Server)
Deploy to a cloud GPU server so the app works on any internet connection without store WiFi.

**Recommended provider**: **RunPod** or **Vast.ai** (cheapest GPU servers for AI workloads)
**Alternative**: Google Cloud (GCP) or AWS EC2 with a GPU instance

**Minimum server specs**:
| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| GPU | None (slow, 5–10s/photo) | NVIDIA 8GB+ VRAM (fast, 0.5s/photo) |
| Storage | 5 GB | 10 GB |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

**Deployment steps on cloud server**:
```bash
# 1. SSH into server
ssh user@your-server-ip

# 2. Clone the project
git clone <your-repo>
cd Navigation-main

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama + pull model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2-vision

# 5. Download GroundingDINO weights into weights/
# (copy groundingdino_swint_ogc.pth to the weights/ folder)

# 6. Start server (with auto-restart)
uvicorn server.server:app --host 0.0.0.0 --port 8000

# 7. Optional: run behind Nginx with HTTPS (free SSL via Let's Encrypt)
```

Set in Android app:
```kotlin
val BASE_URL = "https://your-server-domain.com/"
```

---

### Deployment Stage Summary

| Stage | Method | Cost | Android connects to | Best For |
|---|---|---|---|---|
| Development | Ngrok tunnel from laptop | Free | `https://abc123.ngrok.io` | Building & testing |
| Store pilot | Store local PC + WiFi | Free | `http://192.168.1.10:8000` | Demo / single store |
| Production | Cloud GPU server | ~$20–100/month | `https://yourserver.com` | Live app, multiple stores |
