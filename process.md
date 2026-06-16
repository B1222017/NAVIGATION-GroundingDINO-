# UniGoal Indoor Navigation System - Full Setup & Replication Guide

> **Date:** 2026-05-19
> **Purpose:** Document everything implemented so another device can replicate the entire setup from scratch.

---

## 1. Project Overview

**UniGoal** is an indoor navigation system that uses **GroundingDINO** (an open-vocabulary object detector) to:

- **Detect objects** in photos taken by a phone camera (refrigerator, fire extinguisher, door, printer, etc.)
- **Automatically generate topological maps** from a folder of photos -- no hardcoded data, no manual labeling
- **Navigate in real-time** by uploading one photo at a time via a FastAPI server (uses Ollama VLM for decision-making)

The system works for **any indoor environment**. You take photos, run batch detection, and it clusters photos into zones, finds shared objects, builds a graph, and renders a navigation map with highlighted goals and shortest paths.

### Key Capabilities

1. **Batch Mode:** Process an entire folder of photos at once, detect all objects, output `detections.json`, and auto-generate a topological map PNG.
2. **Map Generation:** From existing detection results, produce a detailed topological map with zones, edges, navigation paths, and all detected objects with confidence scores.
3. **Real-Time Mode:** FastAPI server where a user uploads photos one at a time, and the system provides turn-by-turn navigation guidance using a VLM (Ollama + llama3.2-vision).

---

## 2. System Requirements

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10 / Windows 11 (tested on Windows 10 Pro 10.0.19045) |
| **Python** | **3.12.x** (Python 3.12). Do NOT use Python 3.14 -- it is not supported by PyTorch/GroundingDINO. |
| **CPU** | Any modern x86_64 CPU works. Detection takes ~2-5 seconds per photo on CPU. |
| **GPU** | Optional. CUDA GPU speeds up detection but is NOT required. The system auto-detects and falls back to CPU. |
| **RAM** | 8 GB minimum, 16 GB recommended (GroundingDINO model is ~662 MB) |
| **Disk** | ~3 GB for Python environment + model weights + GroundingDINO repo |
| **Git** | Required for cloning repos |

---

## 3. Installation Steps

### Step 1: Install Python 3.12

Download Python 3.12.x from https://www.python.org/downloads/

> **IMPORTANT:** Do NOT use Python 3.14. PyTorch 2.4.1 and several dependencies do not support it.

During installation, check "Add Python to PATH".

### Step 2: Create Project Directory Structure

```
C:\Users\<username>\Downloads\
    Navigation-main\
        Navigation-main\        <-- main project
    GroundingDINO\              <-- GroundingDINO repo (sibling of Navigation-main)
```

### Step 3: Clone or Copy the Navigation Project

```bash
cd C:\Users\<username>\Downloads
# If from GitHub:
git clone <repo-url> Navigation-main
# Or just copy/extract the Navigation-main folder
```

### Step 4: Clone GroundingDINO Repository

```bash
cd C:\Users\<username>\Downloads
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

This must be placed as a sibling to the `Navigation-main` folder. The config auto-detects its location at:
- `C:\Users\<username>\Downloads\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py`

You can also set the environment variable `GROUNDINGDINO_REPO` to point to a custom location.

### Step 5: Create Virtual Environment

```bash
cd C:\Users\<username>\Downloads\Navigation-main\Navigation-main
python -m venv .venv
```

### Step 6: Activate Virtual Environment

```bash
# Windows CMD:
.venv\Scripts\activate

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Git Bash / WSL:
source .venv/Scripts/activate
```

### Step 7: Install PyTorch (CPU version)

This is the specific version tested and working:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

> **IMPORTANT:** The `+cpu` suffix in the version means it was installed from the CPU wheel index. This exact command ensures you get the correct CPU-only build (~200 MB instead of ~2 GB for CUDA).

If you have an NVIDIA GPU with CUDA, use instead:
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 8: Install GroundingDINO

```bash
cd C:\Users\<username>\Downloads\GroundingDINO
pip install -e .
```

This installs GroundingDINO in editable/development mode so its config files are accessible.

### Step 9: Install transformers (SPECIFIC VERSION REQUIRED)

```bash
pip install transformers==4.44.2
```

> **CRITICAL:** You MUST use transformers 4.44.2. Newer versions (4.45+) break GroundingDINO's tokenizer usage. If you see errors about `BertTokenizer` or `clean_up_tokenization_spaces`, this is the fix.

### Step 10: Install Other Dependencies

```bash
pip install fastapi==0.110.0
pip install uvicorn[standard]==0.27.1
pip install python-multipart==0.0.9
pip install networkx==3.1
pip install requests==2.31.0
pip install pytest==8.0.2
pip install pytest-asyncio==0.23.5
pip install httpx==0.27.0
pip install matplotlib
pip install opencv-python
pip install Pillow
pip install scipy
pip install pydantic
pip install timm
pip install addict
pip install yapf
pip install pycocotools
```

Or install from the requirements file:
```bash
cd C:\Users\<username>\Downloads\Navigation-main\Navigation-main
pip install -r requirements-server.txt
```

Note: `requirements-server.txt` only contains the server dependencies (fastapi, uvicorn, etc.). You still need to install PyTorch, GroundingDINO, transformers, matplotlib, networkx, etc. separately as shown above.

### Step 11: (Optional) Install pillow-heif for HEIC Photo Support

If your photos are in HEIC format (common from iPhones):

```bash
pip install pillow-heif
```

Then convert HEIC to JPG before processing:
```python
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
img = Image.open("photo.HEIC")
img.save("photo.jpg", "JPEG")
```

### Complete One-Line Install (after creating venv)

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu && cd C:\Users\<username>\Downloads\GroundingDINO && pip install -e . && pip install transformers==4.44.2 fastapi==0.110.0 "uvicorn[standard]==0.27.1" python-multipart==0.0.9 networkx==3.1 requests==2.31.0 matplotlib opencv-python Pillow scipy pydantic timm addict yapf pycocotools pytest==8.0.2 pytest-asyncio==0.23.5 httpx==0.27.0
```

---

## 4. Model Weights

### GroundingDINO SwinT OGC (Required)

- **File:** `groundingdino_swint_ogc.pth`
- **Size:** ~662 MB
- **Download from:** https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```bash
# Create models directory and download
mkdir -p C:\Users\<username>\Downloads\Navigation-main\Navigation-main\models
# Download the file and place it at:
# C:\Users\<username>\Downloads\Navigation-main\Navigation-main\models\groundingdino_swint_ogc.pth
```

You can download using curl, wget, or your browser:
```bash
curl -L -o models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### SAM ViT-H (Optional - NOT loaded by default)

- **File:** `sam_vit_h_4b8939.pth`
- Not loaded by default to stay under 8 GB VRAM budget
- Only needed if you enable SAM segmentation masks

---

## 5. Project Structure

```
Navigation-main/
Navigation-main/
    |-- server/                          # Core server package
    |   |-- __init__.py                  # Package init
    |   |-- config.py                    # Configuration: paths, thresholds, env vars
    |   |-- perception.py                # GroundingDINO object detection wrapper
    |   |-- batch_mapper.py              # Batch photo processing + map generation
    |   |-- server.py                    # FastAPI app with REST endpoints
    |   |-- run_server.py                # Uvicorn entry point
    |   |-- models.py                    # Pydantic request/response models
    |   |-- session.py                   # Session state management
    |   |-- topomap.py                   # Topological map data structure + rendering
    |   |-- navigator.py                 # Navigation engine (path finding)
    |   |-- store_map.py                 # Pre-built CSIE office topological map
    |   |-- store_knowledge.py           # Static environment knowledge (rooms, landmarks)
    |   |-- goal_decomposer.py           # Goal text -> detection classes (uses Ollama)
    |   |-- vlm.py                       # VLM client for Ollama (image + prompt)
    |   |-- prompts.py                   # Prompt templates for VLM
    |   |-- annotator.py                 # Draw detection boxes on photos
    |   |-- tests/                       # Test suite
    |       |-- conftest.py
    |       |-- test_perception_smoke.py
    |       |-- test_topomap.py
    |       |-- test_api_*.py            # API endpoint tests
    |       |-- test_session.py
    |       |-- test_goal_decomposer.py
    |       |-- test_vlm.py
    |       |-- test_annotator.py
    |
    |-- generate_topomap.py              # Standalone topological map generator
    |-- models/                          # Model weights directory
    |   |-- groundingdino_swint_ogc.pth  # GroundingDINO weights (662 MB)
    |
    |-- requirements-server.txt          # Server dependencies
    |-- detections.json                  # Example detection output
    |-- detections_groundingdino.json    # Another detection output
    |-- detections_new_env.json          # Supermarket detection output
    |-- .venv/                           # Python virtual environment
    |-- output/                          # Runtime output (sessions, annotated photos)
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `server/config.py` | Central configuration. Auto-detects model paths, sets detection thresholds (box=0.30, text=0.25), server host/port, Ollama URL. |
| `server/perception.py` | Wraps GroundingDINO's `load_model()` and `predict()`. Converts detection boxes from center-format to corner-format. Returns top 5 detections sorted by score. |
| `server/batch_mapper.py` | Main batch processing script. Takes a folder of photos + goal text, runs GroundingDINO on every photo, builds object index, computes photo-to-photo connections, saves JSON, and optionally generates a topological map. |
| `generate_topomap.py` | Reads detection JSON, clusters photos into zones using Jaccard similarity, builds graph edges from shared objects, finds goal zones, computes navigation paths, and renders a detailed PNG map. |
| `server/server.py` | FastAPI app with endpoints: `POST /session` (start), `POST /session/{id}/photo` (upload photo), `POST /session/{id}/answer` (answer question), `GET /session/{id}/map` (get map). |

---

## 6. How to Run

### 6a. Batch Detection + Automatic Map Generation (One Command)

This is the main workflow: process a folder of photos, detect all objects, and generate a topological map.

```bash
cd C:\Users\<username>\Downloads\Navigation-main\Navigation-main

python -m server.batch_mapper --input <photo_folder> --goal "find refrigerator, fire extinguisher" --output detections.json --map --start <start_photo>.jpg
```

**Example with CSIE office photos:**
```bash
python -m server.batch_mapper --input C:\Users\<username>\Downloads\photos_csie --goal "find refrigerator, fire extinguisher" --output detections.json --map --start IMG_1434.jpg
```

**Example with supermarket photos:**
```bash
python -m server.batch_mapper --input C:\Users\<username>\Downloads\photos_supermarket --goal "find refrigerator, fire extinguisher" --output detections_new_env.json --map --start IMG_2001.jpg
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `--input` / `-i` | Folder containing JPG/PNG photos |
| `--goal` / `-g` | What to find (natural language) |
| `--output` / `-o` | Output JSON file path (default: `detections.json`) |
| `--map` / `-m` | Auto-generate topological map after detection |
| `--start` / `-s` | Starting photo filename for navigation path |
| `--extra-classes` | Additional object classes to detect |
| `--map-output` | Map output filename prefix (default: `topological_map`) |

**What happens:**
1. Parses goal text into detection classes (refrigerator, fire extinguisher, door, cabinet, chair, etc.)
2. Loads GroundingDINO model (~10 seconds on first load)
3. Runs detection on every photo (~2-5 seconds each on CPU)
4. Saves all detections to JSON
5. If `--map` is set: auto-clusters photos into zones, builds graph, renders map PNG

**Output files:**
- `detections.json` -- all detection results
- `topological_map.png` -- standard resolution map
- `topological_map_hires.png` -- high resolution map (300 DPI)

### 6b. Generate Topological Map from Existing Detections

If you already have a `detections.json` file and just want to regenerate the map:

```bash
python generate_topomap.py --input detections.json --goal "refrigerator,fire extinguisher" --start IMG_1434.jpg
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `--input` / `-i` | Detection JSON from batch_mapper (default: `detections_groundingdino.json`) |
| `--goal` / `-g` | Goal objects, comma-separated |
| `--start` / `-s` | Start photo filename |
| `--output` / `-o` | Output filename prefix (default: `topological_map`) |
| `--sim-threshold` | Jaccard similarity threshold for zone boundaries (default: 0.25) |

### 6c. Real-Time Navigation Server (One Photo at a Time)

This requires **Ollama** running with `llama3.2-vision` model for VLM-based decision making.

**Prerequisites:**
1. Install Ollama: https://ollama.com/
2. Pull the vision model: `ollama pull llama3.2-vision`
3. Start Ollama service

**Start the server:**
```bash
cd C:\Users\<username>\Downloads\Navigation-main\Navigation-main
python -m server.run_server
```

The server starts at `http://0.0.0.0:8000`.

**API Endpoints:**
```
POST /session                         -- Start navigation session (body: {"goal": "find the refrigerator"})
POST /session/{id}/photo              -- Upload a photo (multipart/form-data)
POST /session/{id}/answer             -- Answer a question from the system
GET  /session/{id}/map                -- Get topological map (JSON or PNG)
GET  /session/{id}/map?format=png     -- Get map as PNG image
GET  /session/{id}                    -- Get session state
GET  /session/{id}/photo/{node}.jpg   -- Get annotated photo
GET  /health                          -- Health check
```

**Environment variables for the server:**
```bash
set OLLAMA_URL=http://127.0.0.1:11434       # Ollama API URL
set OLLAMA_MODEL=llama3.2-vision             # VLM model name
set SERVER_HOST=0.0.0.0                      # Server bind address
set SERVER_PORT=8000                         # Server port
set GROUNDINGDINO_REPO=C:\path\to\GroundingDINO  # Custom GroundingDINO path
```

---

## 7. Key Configuration

All configuration is in `server/config.py`:

### Detection Thresholds

```python
GROUNDINGDINO_BOX_THRESHOLD = 0.30    # Minimum box confidence to keep a detection
GROUNDINGDINO_TEXT_THRESHOLD = 0.25   # Minimum text-image matching score
SAM_TOP_K_BOXES = 5                   # Maximum detections returned per photo
```

- Lower thresholds = more detections (more false positives)
- Higher thresholds = fewer detections (may miss objects)
- Current values (0.30/0.25) work well for indoor environments

### Model Paths (Auto-Detected)

The system auto-detects model weights in this order:
1. `<project_root>/models/groundingdino_swint_ogc.pth` (recommended)
2. `/home/user/UniGoal/data/models/groundingdino_swint_ogc.pth` (WSL fallback)

### GroundingDINO Config Path (Auto-Detected)

Searches these locations in order:
1. `GROUNDINGDINO_REPO` environment variable + `/groundingdino/config/GroundingDINO_SwinT_OGC.py`
2. `<project_parent>/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`
3. `C:\Users\user\Downloads\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py`
4. WSL path fallback

### Topological Map Clustering

In `generate_topomap.py`:
- `sim_threshold = 0.25` -- Jaccard similarity threshold for splitting zones
- `min_zone_size = 2` -- Minimum photos per zone (small zones get merged)
- Goal filtering: weak detections below `max(0.45, best_score * 0.4)` are removed

---

## 8. Known Issues & Fixes

### Python 3.14 Not Supported

PyTorch 2.4.1 does not have wheels for Python 3.14. You will get installation errors. **Use Python 3.12.**

### Specific PyTorch Version Required

```bash
# CORRECT - CPU version:
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# WRONG - don't use pip install torch (gets latest, may not work):
# pip install torch
```

### transformers Must Be 4.44.2

Newer versions of the `transformers` library changed the `BertTokenizer` behavior, which breaks GroundingDINO. You will see warnings about `clean_up_tokenization_spaces` or outright errors.

```bash
# Fix:
pip install transformers==4.44.2
```

### HEIC Photo Conversion

iPhone photos in HEIC format are not directly supported by PIL. Install `pillow-heif`:

```bash
pip install pillow-heif
```

Convert before processing:
```python
import pillow_heif
pillow_heif.register_heif_opener()
from PIL import Image
img = Image.open("photo.HEIC")
img.save("photo.jpg", "JPEG", quality=95)
```

### Windows Glob Duplicate Issue (Fixed)

On Windows, `*.jpg` and `*.JPG` glob patterns match the same files (case-insensitive filesystem). The `batch_mapper.py` handles this with deduplication:

```python
seen_names = set()
for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
    for f in photo_dir.glob(pattern):
        if f.name.lower() not in seen_names:
            seen_names.add(f.name.lower())
            jpg_files.append(f)
```

This is already implemented -- no action needed.

### CJK Font Warnings (Cosmetic)

When rendering maps with matplotlib, you may see warnings about missing CJK (Chinese/Japanese/Korean) fonts. These are cosmetic only and do not affect functionality. The maps render correctly with English labels.

### Ollama Not Required for Batch Mode

The batch detection + map generation workflow (`batch_mapper.py` and `generate_topomap.py`) does NOT require Ollama. It only uses GroundingDINO for detection. Ollama is only needed for the real-time navigation server (`run_server.py`).

---

## 9. Environments Tested

### CSIE Department Office (56 photos)

- **Photos:** 56 JPG images from a walking survey of the CS department office
- **Goal:** "find refrigerator, fire extinguisher"
- **Results:** 10 auto-detected zones, multiple object types including refrigerator, fire extinguisher, door, cabinet, sofa, printer, etc.
- **Detection file:** `detections_groundingdino.json`
- **Pre-built static map:** Available in `server/store_map.py` with 10 named nodes (Entrance, Lobby, Coffee Station, etc.)

### Supermarket (192 photos from 6 areas)

- **Photos:** 192 JPG images covering 6 areas of a supermarket
- **Goal:** "find refrigerator, fire extinguisher"
- **Results:** Auto-clustered into zones with detected products, shelving, signage, etc.
- **Detection file:** `detections_new_env.json`
- **Key point:** Demonstrates that the system works on completely different environments with zero code changes

### How to Test on Your Own Environment

1. Take 20-200+ photos walking through the space (every 2-3 meters, cover all areas)
2. Save photos as JPG in a folder
3. Run:
   ```bash
   python -m server.batch_mapper --input <your_photos> --goal "find <your_target>" --output my_detections.json --map --start <first_photo>.jpg
   ```
4. Check the generated `topological_map.png` for the auto-detected zones and navigation path

---

## 10. Quick Verification After Setup

Run this to verify everything is installed correctly:

```bash
cd C:\Users\<username>\Downloads\Navigation-main\Navigation-main

# 1. Check Python version (should be 3.12.x)
.venv\Scripts\python --version

# 2. Check PyTorch
.venv\Scripts\python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 3. Check GroundingDINO
.venv\Scripts\python -c "from groundingdino.util.inference import load_model; print('GroundingDINO OK')"

# 4. Check model weights exist
.venv\Scripts\python -c "from server.config import GROUNDINGDINO_WEIGHTS; print(f'Weights: {GROUNDINGDINO_WEIGHTS} (exists: {GROUNDINGDINO_WEIGHTS.exists()})')"

# 5. Run tests (optional)
.venv\Scripts\python -m pytest server/tests/ -v
```

---

## 11. Installed Package Versions (Reference)

These are the exact versions in the working `.venv` (Python 3.12):

| Package | Version |
|---------|---------|
| torch | 2.4.1+cpu |
| torchvision | 0.19.1+cpu |
| transformers | 4.44.2 |
| groundingdino | 0.1.0 (editable install) |
| fastapi | 0.110.0 |
| uvicorn | 0.27.1 |
| networkx | 3.6.1 |
| matplotlib | (latest) |
| opencv-python | 4.13.0.92 |
| Pillow | 12.2.0 |
| scipy | 1.17.1 |
| numpy | 2.4.4 |
| pydantic | (latest) |
| timm | (latest) |
| addict | 2.4.0 |
| yapf | 0.43.0 |
| pycocotools | 2.0.11 |
| requests | 2.31.0 |
| python-multipart | 0.0.9 |
| pytest | 8.0.2 |
| httpx | 0.27.0 |
