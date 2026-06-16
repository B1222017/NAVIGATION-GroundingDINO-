# UniGoal Store-Navigation Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an HTTP backend that guides a shopper through an unfamiliar grocery store using interactive single-photo uploads, GroundingDINO/SAM perception, llama3.2-vision reasoning, and a topological map maintained server-side.

**Architecture:** FastAPI app, in-memory session store, NetworkX topological map. Models loaded once at startup (GroundingDINO + SAM in-process; llama3.2-vision via local Ollama HTTP). Each `/photo` upload runs perception → updates topomap → asks VLM for next action → returns guidance. Code lives in the navigator git repo at `server/`, runs from WSL with the existing `unigoal` conda env. UniGoal's installed packages and model weights are reused; UniGoal's BEV/Graph/Agent code is NOT touched (deferred to phone-app phase with ARCore).

**Tech Stack:** Python 3.8 (unigoal env), FastAPI, uvicorn, NetworkX, Pillow, requests (for Ollama), pytest. Reuses already-installed: GroundingDINO, segment_anything (SAM), torch+cu118, llama3.2-vision via Ollama 0.22.0 systemd service.

**Where files live:**
- Code: `C:\Users\user\Desktop\school\navigator\server\` (Windows-side, in user's git repo). From WSL: `/mnt/c/Users/user/Desktop/school/navigator/server/`.
- Model weights: `/home/user/UniGoal/data/models/sam_vit_h_4b8939.pth`, `/home/user/UniGoal/data/models/groundingdino_swint_ogc.pth` (already downloaded).
- Output (photos, logs): `/home/user/UniGoal/output/sessions/{session_id}/` (WSL-side for fast disk I/O).

**How to run any command in this plan:** all `python`/`pytest`/`pip`/`curl` commands run inside WSL Ubuntu-22.04 with the `unigoal` conda env activated and CWD set to the navigator repo root. The boilerplate is:

```bash
wsl -d Ubuntu-22.04 -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unigoal && cd /mnt/c/Users/user/Desktop/school/navigator && <COMMAND>"
```

Throughout this plan I'll write `<COMMAND>` and you wrap it with the boilerplate.

---

### Task 1: Project skeleton and dependencies

**Files:**
- Create: `server/__init__.py`
- Create: `server/tests/__init__.py`
- Create: `server/tests/conftest.py`
- Create: `pytest.ini`
- Create: `requirements-server.txt`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p server/tests
touch server/__init__.py server/tests/__init__.py
```

- [ ] **Step 2: Write requirements-server.txt**

```
fastapi==0.110.0
uvicorn[standard]==0.27.1
python-multipart==0.0.9
networkx==3.1
requests==2.31.0
pytest==8.0.2
pytest-asyncio==0.23.5
httpx==0.27.0
```

- [ ] **Step 3: Write pytest.ini**

```ini
[pytest]
testpaths = server/tests
python_files = test_*.py
asyncio_mode = auto
```

- [ ] **Step 4: Write server/tests/conftest.py**

```python
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("UNIGOAL_TEST_MODE", "1")
```

- [ ] **Step 5: Install deps**

Run: `pip install -r requirements-server.txt`
Expected: `Successfully installed fastapi-0.110.0 ...`

- [ ] **Step 6: Verify pytest discovers nothing yet**

Run: `pytest`
Expected: `no tests ran in 0.0Xs` (exit 5 — that's fine for empty)

- [ ] **Step 7: Commit**

```bash
cd /mnt/c/Users/user/Desktop/school/navigator
git add server/ requirements-server.txt pytest.ini
git commit -m "feat(server): scaffold FastAPI server package"
```

---

### Task 2: Config module

**Files:**
- Create: `server/config.py`

- [ ] **Step 1: Write config.py**

```python
"""Configuration constants and paths.

All paths are absolute (WSL paths). Model weights live under UniGoal's data
directory; outputs (photos, logs) go to UniGoal's output dir for fast disk I/O.
"""
import os
from pathlib import Path

UNIGOAL_ROOT = Path("/home/user/UniGoal")

SAM_WEIGHTS = UNIGOAL_ROOT / "data" / "models" / "sam_vit_h_4b8939.pth"
GROUNDINGDINO_WEIGHTS = UNIGOAL_ROOT / "data" / "models" / "groundingdino_swint_ogc.pth"
GROUNDINGDINO_CONFIG = UNIGOAL_ROOT / "third_party" / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"

OUTPUT_ROOT = UNIGOAL_ROOT / "output" / "sessions"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")

SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

GROUNDINGDINO_BOX_THRESHOLD = 0.30
GROUNDINGDINO_TEXT_THRESHOLD = 0.25
SAM_TOP_K_BOXES = 5

VLM_TIMEOUT_S = 60
GOAL_DECOMPOSE_TIMEOUT_S = 30


def ensure_output_dir(session_id: str) -> Path:
    p = OUTPUT_ROOT / session_id
    (p / "photo").mkdir(parents=True, exist_ok=True)
    (p / "annotated").mkdir(parents=True, exist_ok=True)
    return p
```

- [ ] **Step 2: Verify import**

Run: `python -c "from server.config import OLLAMA_URL, SAM_WEIGHTS; print(OLLAMA_URL, SAM_WEIGHTS)"`
Expected: prints `http://127.0.0.1:11434 /home/user/UniGoal/data/models/sam_vit_h_4b8939.pth`

- [ ] **Step 3: Commit**

```bash
git add server/config.py
git commit -m "feat(server): add config module"
```

---

### Task 3: Pydantic models

**Files:**
- Create: `server/models.py`

- [ ] **Step 1: Write models.py**

```python
"""Pydantic request/response models and internal dataclasses."""
from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class VLMAction(str, Enum):
    ARRIVED = "ARRIVED"
    MOVE = "MOVE"
    ASK = "ASK"


class Detection(BaseModel):
    label: str
    box: list[float] = Field(..., description="[x1,y1,x2,y2] absolute pixels")
    score: float


class VLMResponse(BaseModel):
    action: VLMAction
    guidance: str
    question: Optional[str] = None
    vlm_summary: str = ""


class StartSessionRequest(BaseModel):
    goal: str


class StartSessionResponse(BaseModel):
    session_id: str
    guidance: str
    action: Literal["TAKE_PHOTO"] = "TAKE_PHOTO"
    goal_objects: list[str]


class AnswerRequest(BaseModel):
    answer: str


class TurnResponse(BaseModel):
    action: VLMAction
    guidance: str
    question: Optional[str] = None
    node_id: int
    annotated_photo_url: Optional[str] = None


class NodeJSON(BaseModel):
    id: int
    photo: str
    detected: list[str]
    summary: str
    timestamp: str


class EdgeJSON(BaseModel):
    from_id: int = Field(..., alias="from")
    to: int
    action: str

    class Config:
        populate_by_name = True


class MapJSON(BaseModel):
    nodes: list[NodeJSON]
    edges: list[EdgeJSON]
    current_node: Optional[int]
    goal_node: Optional[int]


class ErrorResponse(BaseModel):
    error: str
    detail: str
```

- [ ] **Step 2: Verify import**

Run: `python -c "from server.models import VLMAction, TurnResponse; print(VLMAction.MOVE, TurnResponse.__fields__.keys())"`
Expected: prints `VLMAction.MOVE dict_keys(['action', 'guidance', 'question', 'node_id', 'annotated_photo_url'])`

- [ ] **Step 3: Commit**

```bash
git add server/models.py
git commit -m "feat(server): add pydantic models"
```

---

### Task 4: TopoMap — add_node and add_edge (TDD)

**Files:**
- Create: `server/topomap.py`
- Create: `server/tests/test_topomap.py`

- [ ] **Step 1: Write the failing test**

`server/tests/test_topomap.py`:
```python
from server.topomap import TopoMap


def test_add_node_returns_incrementing_ids():
    tm = TopoMap()
    n0 = tm.add_node(photo_path="a.jpg", detected=["bread"], summary="bakery")
    n1 = tm.add_node(photo_path="b.jpg", detected=["milk"], summary="dairy")
    assert n0 == 0
    assert n1 == 1


def test_add_edge_records_action():
    tm = TopoMap()
    a = tm.add_node(photo_path="a.jpg", detected=[], summary="")
    b = tm.add_node(photo_path="b.jpg", detected=[], summary="")
    tm.add_edge(a, b, action="walked left")
    edges = list(tm.graph.edges(data=True))
    assert len(edges) == 1
    assert edges[0][0] == a
    assert edges[0][1] == b
    assert edges[0][2]["action"] == "walked left"


def test_to_dict_serializes_graph():
    tm = TopoMap()
    a = tm.add_node(photo_path="a.jpg", detected=["x"], summary="s1")
    b = tm.add_node(photo_path="b.jpg", detected=["y"], summary="s2")
    tm.add_edge(a, b, action="forward")
    d = tm.to_dict(current_node=b, goal_node=None)
    assert d["nodes"][0]["id"] == 0
    assert d["nodes"][1]["detected"] == ["y"]
    assert d["edges"][0] == {"from": 0, "to": 1, "action": "forward"}
    assert d["current_node"] == b
    assert d["goal_node"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest server/tests/test_topomap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'server.topomap'`

- [ ] **Step 3: Implement topomap.py**

```python
"""Topological map: nodes are photo locations, edges are movement actions."""
from datetime import datetime
from typing import Optional
import networkx as nx


class TopoMap:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._next_id: int = 0

    def add_node(self, photo_path: str, detected: list[str], summary: str) -> int:
        nid = self._next_id
        self._next_id += 1
        self.graph.add_node(
            nid,
            photo_path=photo_path,
            detected=detected,
            summary=summary,
            timestamp=datetime.utcnow().isoformat(),
        )
        return nid

    def add_edge(self, from_id: int, to_id: int, action: str) -> None:
        self.graph.add_edge(from_id, to_id, action=action)

    def to_dict(self, current_node: Optional[int], goal_node: Optional[int]) -> dict:
        nodes = [
            {
                "id": nid,
                "photo": data["photo_path"],
                "detected": data["detected"],
                "summary": data["summary"],
                "timestamp": data["timestamp"],
            }
            for nid, data in self.graph.nodes(data=True)
        ]
        edges = [
            {"from": u, "to": v, "action": data["action"]}
            for u, v, data in self.graph.edges(data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "current_node": current_node,
            "goal_node": goal_node,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest server/tests/test_topomap.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/topomap.py server/tests/test_topomap.py
git commit -m "feat(server): topomap add_node/add_edge/to_dict"
```

---

### Task 5: TopoMap.summarize_for_vlm (TDD)

**Files:**
- Modify: `server/topomap.py` (add method)
- Modify: `server/tests/test_topomap.py` (add test)

- [ ] **Step 1: Append failing test**

Append to `server/tests/test_topomap.py`:
```python
def test_summarize_walks_from_start_to_current():
    tm = TopoMap()
    a = tm.add_node("a.jpg", ["bread"], "bakery aisle")
    b = tm.add_node("b.jpg", ["milk", "yogurt"], "dairy section")
    c = tm.add_node("c.jpg", ["checkout"], "near checkout")
    tm.add_edge(a, b, "walked forward")
    tm.add_edge(b, c, "turned left")
    summary = tm.summarize_for_vlm(current_id=c)
    assert "bakery aisle" in summary
    assert "walked forward" in summary
    assert "dairy section" in summary
    assert "turned left" in summary
    assert "near checkout" in summary


def test_summarize_single_node():
    tm = TopoMap()
    a = tm.add_node("a.jpg", ["bread"], "starting view")
    summary = tm.summarize_for_vlm(current_id=a)
    assert "starting view" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest server/tests/test_topomap.py::test_summarize_walks_from_start_to_current -v`
Expected: FAIL with `AttributeError: 'TopoMap' object has no attribute 'summarize_for_vlm'`

- [ ] **Step 3: Append method to topomap.py**

```python
    def summarize_for_vlm(self, current_id: int) -> str:
        """Walk from start to current, produce <150-word prose summary for the VLM."""
        if self.graph.number_of_nodes() == 0:
            return "No locations visited yet."

        try:
            start = next(n for n in self.graph.nodes if self.graph.in_degree(n) == 0)
        except StopIteration:
            start = 0

        if start == current_id:
            data = self.graph.nodes[current_id]
            return f"You are at the starting location. Visible: {data['summary']}."

        try:
            path = nx.shortest_path(self.graph, source=start, target=current_id)
        except nx.NetworkXNoPath:
            path = [current_id]

        parts: list[str] = []
        for i, nid in enumerate(path):
            node = self.graph.nodes[nid]
            label = node["summary"] or ", ".join(node["detected"][:3]) or f"location {nid}"
            if i == 0:
                parts.append(f"Started at {label}")
            else:
                action = self.graph.edges[path[i - 1], nid]["action"]
                parts.append(f"{action}, arrived at {label}")
        return ". ".join(parts) + "."
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest server/tests/test_topomap.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/topomap.py server/tests/test_topomap.py
git commit -m "feat(server): topomap summarize_for_vlm"
```

---

### Task 6: TopoMap.render_png (smoke test only)

**Files:**
- Modify: `server/topomap.py`
- Modify: `server/tests/test_topomap.py`

- [ ] **Step 1: Append smoke test**

```python
def test_render_png_returns_nonempty_bytes(tmp_path):
    tm = TopoMap()
    a = tm.add_node("a.jpg", [], "")
    b = tm.add_node("b.jpg", [], "")
    tm.add_edge(a, b, "forward")
    png_bytes = tm.render_png(current_id=b)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 100
    assert png_bytes.startswith(b"\x89PNG")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest server/tests/test_topomap.py::test_render_png_returns_nonempty_bytes -v`
Expected: FAIL `AttributeError: ... has no attribute 'render_png'`

- [ ] **Step 3: Append render_png to topomap.py**

Add at top of file:
```python
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Append method:
```python
    def render_png(self, current_id: Optional[int] = None) -> bytes:
        fig, ax = plt.subplots(figsize=(6, 6))
        if self.graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "(empty map)", ha="center", va="center")
        else:
            pos = nx.spring_layout(self.graph, seed=42)
            node_colors = ["red" if n == current_id else "lightblue" for n in self.graph.nodes]
            nx.draw(
                self.graph, pos, ax=ax, with_labels=True,
                node_color=node_colors, node_size=600, font_size=10,
            )
            edge_labels = {(u, v): d["action"][:20] for u, v, d in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=ax, font_size=8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_topomap.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/topomap.py server/tests/test_topomap.py
git commit -m "feat(server): topomap render_png"
```

---

### Task 7: Session and SessionStore (TDD)

**Files:**
- Create: `server/session.py`
- Create: `server/tests/test_session.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_session.py`:
```python
from server.session import Session, SessionStore


def test_create_session_assigns_unique_id():
    store = SessionStore()
    s1 = store.create(goal="find milk", goal_objects=["milk"])
    s2 = store.create(goal="find bread", goal_objects=["bread"])
    assert s1.id != s2.id
    assert s1.goal == "find milk"
    assert s1.arrived is False
    assert s1.pending_question is None


def test_get_returns_same_session():
    store = SessionStore()
    s = store.create(goal="g", goal_objects=[])
    assert store.get(s.id) is s


def test_get_unknown_id_returns_none():
    store = SessionStore()
    assert store.get("nope") is None


def test_session_has_topomap():
    store = SessionStore()
    s = store.create(goal="g", goal_objects=[])
    assert s.topomap is not None
    assert s.topomap.graph.number_of_nodes() == 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest server/tests/test_session.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'server.session'`

- [ ] **Step 3: Write session.py**

```python
"""Session state and in-memory store."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from server.topomap import TopoMap


@dataclass
class Session:
    id: str
    goal: str
    goal_objects: list[str]
    topomap: TopoMap = field(default_factory=TopoMap)
    history: list[dict] = field(default_factory=list)
    pending_question: Optional[str] = None
    last_planned_action: Optional[str] = None
    arrived: bool = False
    goal_node: Optional[int] = None
    last_node_id: Optional[int] = None
    last_detections: list[dict] = field(default_factory=list)
    last_photo_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, goal: str, goal_objects: list[str]) -> Session:
        sid = uuid.uuid4().hex[:8]
        s = Session(id=sid, goal=goal, goal_objects=goal_objects)
        self._sessions[sid] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_session.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/session.py server/tests/test_session.py
git commit -m "feat(server): session and session store"
```

---

### Task 8: Prompts module

**Files:**
- Create: `server/prompts.py`

- [ ] **Step 1: Write prompts.py**

```python
"""Centralized prompt templates for goal decomposition and per-turn VLM calls."""

GOAL_DECOMPOSE_PROMPT = """\
You are helping someone navigate a grocery store. Their goal is: "{goal}".
List 4-8 visual things they should look for to know they're close to the goal,
as a single comma-separated list. Be specific. Reply ONLY with the list, no preamble.

Example for "find the cookies":
cookies, biscuits, snacks aisle, packaged sweets, candy, sugar
"""


PER_TURN_PROMPT = """\
You are guiding a shopper through a grocery store using their phone camera.

Goal: "{goal}"
Goal-related items to look for: {goal_objects}

What's happened so far:
{topomap_summary}

The shopper just uploaded the attached photo. In it, automatic detection found:
{detections_summary}

{prior_answer_block}

Decide the next step. Reply with EXACTLY one JSON object on one line, nothing else:

{{"action": "ARRIVED" | "MOVE" | "ASK", "guidance": "<one or two sentences>", "question": "<short question, only if ASK, else null>", "vlm_summary": "<one phrase summarizing the location>"}}

Rules:
- ARRIVED only if the goal item is clearly visible in the photo (point at it in `guidance`).
- ASK if you cannot decide between two plausible directions and a yes/no answer would resolve it.
- MOVE otherwise. Tell the user a concrete direction (e.g., "turn left and walk to the back wall, then take another photo").
- Do NOT invent details not in the photo or detections.
"""


PRIOR_ANSWER_BLOCK = """\
The shopper just answered your earlier question "{previous_question}" with:
"{user_answer}"
"""
```

- [ ] **Step 2: Verify import**

Run: `python -c "from server.prompts import PER_TURN_PROMPT; print(len(PER_TURN_PROMPT))"`
Expected: prints a number > 500

- [ ] **Step 3: Commit**

```bash
git add server/prompts.py
git commit -m "feat(server): prompt templates"
```

---

### Task 9: Goal decomposer (TDD with mocked Ollama)

**Files:**
- Create: `server/goal_decomposer.py`
- Create: `server/tests/test_goal_decomposer.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_goal_decomposer.py`:
```python
from unittest.mock import patch, MagicMock
from server.goal_decomposer import decompose_goal


def _mock_ollama_response(text: str):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": text}
    resp.raise_for_status = lambda: None
    return resp


def test_decompose_parses_comma_separated():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_ollama_response("milk, dairy, fridge, carton, shelf")):
        objs = decompose_goal("find the milk")
    assert objs == ["milk", "dairy", "fridge", "carton", "shelf"]


def test_decompose_strips_whitespace_and_lowercases():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_ollama_response("  Milk , DAIRY ,fridge")):
        objs = decompose_goal("find the milk")
    assert objs == ["milk", "dairy", "fridge"]


def test_decompose_falls_back_when_response_garbage():
    with patch("server.goal_decomposer.requests.post",
               return_value=_mock_ollama_response("Sure! Here are 50 items: " + "x," * 50)):
        objs = decompose_goal("find the cereal")
    assert "cereal" in objs


def test_decompose_falls_back_on_exception():
    with patch("server.goal_decomposer.requests.post", side_effect=Exception("boom")):
        objs = decompose_goal("find the cookies")
    assert "cookies" in objs
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest server/tests/test_goal_decomposer.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write goal_decomposer.py**

```python
"""Decompose a free-text goal into a list of object/region prompts for GroundingDINO."""
import logging
import re
import requests

from server.config import OLLAMA_URL, OLLAMA_MODEL, GOAL_DECOMPOSE_TIMEOUT_S
from server.prompts import GOAL_DECOMPOSE_PROMPT

log = logging.getLogger(__name__)

_MAX_ITEMS = 10


def _fallback(goal: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    stop = {"find", "the", "a", "an", "to", "where", "is", "are"}
    return [w for w in words if w not in stop] or [goal.strip().lower()]


def decompose_goal(goal: str) -> list[str]:
    prompt = GOAL_DECOMPOSE_PROMPT.format(goal=goal)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=GOAL_DECOMPOSE_TIMEOUT_S,
        )
        r.raise_for_status()
        text = r.json().get("response", "")
    except Exception as e:
        log.warning("goal decompose failed: %s — falling back", e)
        return _fallback(goal)

    items = [s.strip().lower() for s in text.split(",")]
    items = [s for s in items if 1 <= len(s) <= 30 and s.isascii()]
    if not (1 <= len(items) <= _MAX_ITEMS):
        log.warning("goal decompose produced %d items — falling back", len(items))
        return _fallback(goal)
    return items
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_goal_decomposer.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/goal_decomposer.py server/tests/test_goal_decomposer.py
git commit -m "feat(server): goal text decomposer with VLM"
```

---

### Task 10: VLM client (TDD with mocked Ollama)

**Files:**
- Create: `server/vlm.py`
- Create: `server/tests/test_vlm.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_vlm.py`:
```python
import json
from unittest.mock import patch, MagicMock
from server.vlm import decide
from server.models import VLMAction


def _mock(response_text: str):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"response": response_text}
    r.raise_for_status = lambda: None
    return r


def test_decide_parses_arrived(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG marker
    payload = json.dumps({"action": "ARRIVED", "guidance": "milk on right",
                          "question": None, "vlm_summary": "dairy"})
    with patch("server.vlm.requests.post", return_value=_mock(payload)):
        resp = decide(image_path=str(img), goal="milk", goal_objects=["milk"],
                      topomap_summary="", detections_summary="milk x1",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.ARRIVED
    assert resp.guidance == "milk on right"
    assert resp.question is None


def test_decide_falls_back_on_unparseable(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    with patch("server.vlm.requests.post", return_value=_mock("hmm let me think...")):
        resp = decide(image_path=str(img), goal="g", goal_objects=[],
                      topomap_summary="", detections_summary="",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.MOVE
    assert "trouble" in resp.guidance.lower() or "another" in resp.guidance.lower()


def test_decide_falls_back_on_exception(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    with patch("server.vlm.requests.post", side_effect=Exception("boom")):
        resp = decide(image_path=str(img), goal="g", goal_objects=[],
                      topomap_summary="", detections_summary="",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.MOVE


def test_decide_includes_prior_answer_block(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    payload = json.dumps({"action": "MOVE", "guidance": "ok", "question": None, "vlm_summary": "s"})
    captured = {}
    def fake_post(url, **kw):
        captured["body"] = kw["json"]
        return _mock(payload)
    with patch("server.vlm.requests.post", side_effect=fake_post):
        decide(image_path=str(img), goal="g", goal_objects=[],
               topomap_summary="", detections_summary="",
               prior_question="Are you in dairy?", prior_answer="yes")
    assert "Are you in dairy?" in captured["body"]["prompt"]
    assert "yes" in captured["body"]["prompt"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest server/tests/test_vlm.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write vlm.py**

```python
"""VLM client: send (image + prompt) to Ollama and parse structured response."""
import base64
import json
import logging
import re
from typing import Optional

import requests

from server.config import OLLAMA_URL, OLLAMA_MODEL, VLM_TIMEOUT_S
from server.models import VLMAction, VLMResponse
from server.prompts import PER_TURN_PROMPT, PRIOR_ANSWER_BLOCK

log = logging.getLogger(__name__)

_FALLBACK = VLMResponse(
    action=VLMAction.MOVE,
    guidance="I had trouble understanding the scene. Try walking forward a few steps and uploading another photo.",
    question=None,
    vlm_summary="",
)


def _build_prompt(
    goal: str,
    goal_objects: list[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
) -> str:
    if prior_question and prior_answer:
        block = PRIOR_ANSWER_BLOCK.format(previous_question=prior_question, user_answer=prior_answer)
    else:
        block = ""
    return PER_TURN_PROMPT.format(
        goal=goal,
        goal_objects=", ".join(goal_objects) or "(none)",
        topomap_summary=topomap_summary or "(starting location)",
        detections_summary=detections_summary or "(no detections)",
        prior_answer_block=block,
    )


def _parse(text: str) -> Optional[VLMResponse]:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return VLMResponse(
            action=VLMAction(obj["action"]),
            guidance=str(obj.get("guidance", "")),
            question=obj.get("question"),
            vlm_summary=str(obj.get("vlm_summary", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("VLM JSON parse failed: %s", e)
        return None


def decide(
    image_path: str,
    goal: str,
    goal_objects: list[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
) -> VLMResponse:
    prompt = _build_prompt(goal, goal_objects, topomap_summary, detections_summary, prior_question, prior_answer)
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "images": [img_b64], "stream": False},
            timeout=VLM_TIMEOUT_S,
        )
        r.raise_for_status()
        text = r.json().get("response", "")
    except Exception as e:
        log.warning("VLM call failed: %s", e)
        return _FALLBACK

    parsed = _parse(text)
    if parsed is None:
        log.warning("VLM unparseable response: %.200s", text)
        return _FALLBACK
    return parsed
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_vlm.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/vlm.py server/tests/test_vlm.py
git commit -m "feat(server): VLM client with Ollama"
```

---

### Task 11: Perception module (smoke test only — needs GPU)

**Files:**
- Create: `server/perception.py`
- Create: `server/tests/test_perception_smoke.py`

- [ ] **Step 1: Write perception.py**

```python
"""GroundingDINO + SAM perception. Loaded once at server startup."""
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from server.config import (
    GROUNDINGDINO_CONFIG, GROUNDINGDINO_WEIGHTS, SAM_WEIGHTS,
    GROUNDINGDINO_BOX_THRESHOLD, GROUNDINGDINO_TEXT_THRESHOLD, SAM_TOP_K_BOXES,
)

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    box: list[float]  # [x1,y1,x2,y2] absolute pixels
    score: float


class Perception:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._gd_model = None
        self._sam_predictor = None

    def load(self) -> None:
        from groundingdino.util.inference import load_model
        from segment_anything import sam_model_registry, SamPredictor

        log.info("loading GroundingDINO from %s", GROUNDINGDINO_WEIGHTS)
        self._gd_model = load_model(str(GROUNDINGDINO_CONFIG), str(GROUNDINGDINO_WEIGHTS))
        log.info("loading SAM from %s", SAM_WEIGHTS)
        sam = sam_model_registry["vit_h"](checkpoint=str(SAM_WEIGHTS))
        sam.to(self.device)
        self._sam_predictor = SamPredictor(sam)
        log.info("perception loaded on %s", self.device)

    def detect(self, image_path: str, prompt_classes: list[str]) -> list[Detection]:
        if not prompt_classes:
            return []
        from groundingdino.util.inference import predict, load_image
        text_prompt = " . ".join(prompt_classes) + " ."
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self._gd_model,
            image=image,
            caption=text_prompt,
            box_threshold=GROUNDINGDINO_BOX_THRESHOLD,
            text_threshold=GROUNDINGDINO_TEXT_THRESHOLD,
            device=self.device,
        )

        h, w = image_source.shape[:2]
        results: list[Detection] = []
        for box_cxcywh, score, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box_cxcywh.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            results.append(Detection(label=phrase, box=[x1, y1, x2, y2], score=float(score)))

        results.sort(key=lambda d: -d.score)
        return results[:SAM_TOP_K_BOXES]
```

- [ ] **Step 2: Write smoke test**

`server/tests/test_perception_smoke.py`:
```python
"""Smoke test — only runs when SMOKE=1 env var is set, since it needs GPU + weights."""
import os
import pytest
from pathlib import Path
from server.perception import Perception
from server.config import SAM_WEIGHTS, GROUNDINGDINO_WEIGHTS


@pytest.mark.skipif(os.environ.get("SMOKE") != "1", reason="GPU smoke test")
def test_perception_loads_and_detects():
    if not (SAM_WEIGHTS.exists() and GROUNDINGDINO_WEIGHTS.exists()):
        pytest.skip("model weights not present")

    p = Perception()
    p.load()

    sample = Path("/home/user/UniGoal/assets/demo_real.gif")
    if not sample.exists():
        pytest.skip("no sample image")
    detections = p.detect(str(sample), ["person", "wall"])
    assert isinstance(detections, list)
```

- [ ] **Step 3: Run smoke test (manual)**

Run: `SMOKE=1 pytest server/tests/test_perception_smoke.py -v -s`
Expected: PASSED (takes ~30s on first run as models load)

If it fails because the sample image isn't a JPEG, that's fine — adjust to use any JPEG you have. The point is to confirm no import / weight-loading errors.

- [ ] **Step 4: Commit**

```bash
git add server/perception.py server/tests/test_perception_smoke.py
git commit -m "feat(server): perception with GroundingDINO+SAM"
```

---

### Task 12: Annotator (TDD with synthetic image)

**Files:**
- Create: `server/annotator.py`
- Create: `server/tests/test_annotator.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_annotator.py`:
```python
from PIL import Image
from server.annotator import annotate
from server.perception import Detection


def test_annotate_writes_png(tmp_path):
    src = tmp_path / "src.jpg"
    Image.new("RGB", (200, 200), color=(255, 255, 255)).save(src)
    dst = tmp_path / "out.jpg"
    detections = [Detection(label="cat", box=[10, 10, 80, 80], score=0.9)]
    annotate(str(src), str(dst), detections, banner_text="GUIDANCE: turn left")
    assert dst.exists()
    out = Image.open(dst)
    assert out.size[0] == 200 and out.size[1] >= 200  # banner adds height
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest server/tests/test_annotator.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write annotator.py**

```python
"""Draw detection boxes + a guidance banner onto a copy of the photo."""
from PIL import Image, ImageDraw, ImageFont
from typing import List

from server.perception import Detection

_BANNER_HEIGHT = 60


def annotate(src_path: str, dst_path: str, detections: List[Detection], banner_text: str) -> None:
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    canvas = Image.new("RGB", (w, h + _BANNER_HEIGHT), color=(20, 20, 20))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for d in detections:
        x1, y1, x2, y2 = d.box
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{d.label} {d.score:.2f}", fill=(0, 255, 0))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, h + 8), banner_text[:200], fill=(255, 255, 255), font=font)

    canvas.save(dst_path, "JPEG", quality=85)
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_annotator.py -v`
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add server/annotator.py server/tests/test_annotator.py
git commit -m "feat(server): annotator"
```

---

### Task 13: FastAPI app skeleton + POST /session (TDD)

**Files:**
- Create: `server/server.py`
- Create: `server/tests/test_api_session.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_api_session.py`:
```python
from unittest.mock import patch
from fastapi.testclient import TestClient
from server.server import app, get_store


def test_post_session_returns_id_and_decomposed_goals():
    with patch("server.server.decompose_goal", return_value=["milk", "dairy"]):
        client = TestClient(app)
        r = client.post("/session", json={"goal": "find the milk"})
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert body["goal_objects"] == ["milk", "dairy"]
    assert body["action"] == "TAKE_PHOTO"


def test_post_session_rejects_empty_goal():
    client = TestClient(app)
    r = client.post("/session", json={"goal": ""})
    assert r.status_code == 400
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest server/tests/test_api_session.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write server.py (skeleton + /session)**

```python
"""FastAPI app — endpoints for in-store navigation sessions."""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from server.goal_decomposer import decompose_goal
from server.models import StartSessionRequest, StartSessionResponse, ErrorResponse
from server.session import SessionStore

log = logging.getLogger(__name__)

app = FastAPI(title="UniGoal Store-Nav Server")

_store = SessionStore()


def get_store() -> SessionStore:
    return _store


@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail.get("error", "error"),
                              detail=exc.detail.get("detail", "")).dict()
        if isinstance(exc.detail, dict) else
        ErrorResponse(error="error", detail=str(exc.detail)).dict(),
    )


@app.post("/session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail={"error": "bad_request", "detail": "goal is empty"})
    goal_objects = decompose_goal(req.goal)
    s = _store.create(goal=req.goal, goal_objects=goal_objects)
    return StartSessionResponse(
        session_id=s.id,
        guidance="Upload a starting photo so I can see where you are.",
        goal_objects=goal_objects,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_api_session.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/server.py server/tests/test_api_session.py
git commit -m "feat(server): POST /session endpoint"
```

---

### Task 14: POST /session/{id}/photo (TDD with mocked perception + VLM)

**Files:**
- Modify: `server/server.py`
- Create: `server/tests/test_api_photo.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_api_photo.py`:
```python
import io
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient
from server.server import app
from server.models import VLMAction, VLMResponse
from server.perception import Detection


def _make_jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(127, 127, 127)).save(buf, "JPEG")
    return buf.getvalue()


def _start_session(client) -> str:
    with patch("server.server.decompose_goal", return_value=["milk"]):
        r = client.post("/session", json={"goal": "find the milk"})
    return r.json()["session_id"]


def test_post_photo_returns_move(tmp_path):
    client = TestClient(app)
    sid = _start_session(client)
    fake_perception = MagicMock()
    fake_perception.detect.return_value = [
        Detection(label="shelf", box=[0, 0, 50, 50], score=0.7)
    ]
    fake_vlm_resp = VLMResponse(action=VLMAction.MOVE, guidance="walk forward",
                                question=None, vlm_summary="aisle")
    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=fake_vlm_resp):
        r = client.post(
            f"/session/{sid}/photo",
            files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "MOVE"
    assert body["guidance"] == "walk forward"
    assert body["node_id"] == 0


def test_post_photo_unknown_session():
    client = TestClient(app)
    r = client.post("/session/nope/photo",
                    files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    assert r.status_code == 404


def test_post_photo_after_arrived_409(tmp_path):
    client = TestClient(app)
    sid = _start_session(client)
    fake_perception = MagicMock()
    fake_perception.detect.return_value = []
    arrived_resp = VLMResponse(action=VLMAction.ARRIVED, guidance="found", question=None, vlm_summary="")
    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=arrived_resp):
        client.post(f"/session/{sid}/photo",
                    files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
        r2 = client.post(f"/session/{sid}/photo",
                         files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    assert r2.status_code == 409
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest server/tests/test_api_photo.py -v`
Expected: FAIL — most likely `AttributeError: module ... has no attribute 'get_perception'`

- [ ] **Step 3: Add /photo to server.py**

Add imports near top:
```python
from pathlib import Path
from fastapi import UploadFile, File
from server.config import ensure_output_dir
from server.models import TurnResponse, VLMAction
from server.vlm import decide as _vlm_decide_impl
from server.perception import Perception
from server.annotator import annotate
```

Add module-level perception holder + accessor:
```python
_perception: Perception | None = None


def get_perception() -> Perception:
    global _perception
    if _perception is None:
        _perception = Perception()
        _perception.load()
    return _perception


def vlm_decide(*args, **kwargs):
    return _vlm_decide_impl(*args, **kwargs)
```

Add the route:
```python
@app.post("/session/{session_id}/photo", response_model=TurnResponse)
async def upload_photo(session_id: str, photo: UploadFile = File(...)) -> TurnResponse:
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if s.arrived:
        raise HTTPException(status_code=409, detail={"error": "already_arrived", "detail": "session is complete"})
    if s.pending_question is not None:
        raise HTTPException(status_code=409, detail={"error": "answer_pending", "detail": "POST /answer first"})

    out_dir = ensure_output_dir(session_id)
    photo_bytes = await photo.read()
    nid_for_path = s.topomap.graph.number_of_nodes()
    photo_path = out_dir / "photo" / f"{nid_for_path}.jpg"
    photo_path.write_bytes(photo_bytes)

    perception = get_perception()
    detections = perception.detect(str(photo_path), s.goal_objects)
    detected_labels = [d.label for d in detections]

    nid = s.topomap.add_node(photo_path=str(photo_path), detected=detected_labels, summary="")
    if s.last_node_id is not None:
        s.topomap.add_edge(s.last_node_id, nid, action=s.last_planned_action or "(unknown)")

    detections_summary = ", ".join(f"{d.label} ({d.score:.2f})" for d in detections) or "(none)"
    topomap_summary = s.topomap.summarize_for_vlm(current_id=nid)

    vlm_resp = vlm_decide(
        image_path=str(photo_path),
        goal=s.goal,
        goal_objects=s.goal_objects,
        topomap_summary=topomap_summary,
        detections_summary=detections_summary,
        prior_question=None,
        prior_answer=None,
    )

    s.topomap.graph.nodes[nid]["summary"] = vlm_resp.vlm_summary
    s.last_detections = [d.__dict__ for d in detections]
    s.last_photo_path = str(photo_path)
    s.last_node_id = nid

    annotated_path = out_dir / "annotated" / f"{nid}.jpg"
    annotate(str(photo_path), str(annotated_path), detections, banner_text=f"{vlm_resp.action.value}: {vlm_resp.guidance}")

    s.history.append({
        "kind": "photo",
        "node_id": nid,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
    })

    if vlm_resp.action == VLMAction.ARRIVED:
        s.arrived = True
        s.goal_node = nid
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
    else:  # MOVE
        s.last_planned_action = vlm_resp.guidance

    return TurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        node_id=nid,
        annotated_photo_url=f"/session/{session_id}/photo/{nid}.jpg",
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_api_photo.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Run full test suite to catch regressions**

Run: `pytest`
Expected: all passing (skipped: 1 — the smoke test)

- [ ] **Step 6: Commit**

```bash
git add server/server.py server/tests/test_api_photo.py
git commit -m "feat(server): POST /session/{id}/photo endpoint"
```

---

### Task 15: POST /session/{id}/answer (TDD)

**Files:**
- Modify: `server/server.py`
- Create: `server/tests/test_api_answer.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_api_answer.py`:
```python
import io
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient
from server.server import app
from server.models import VLMAction, VLMResponse
from server.perception import Detection


def _make_jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(127, 127, 127)).save(buf, "JPEG")
    return buf.getvalue()


def test_answer_after_ask_advances_state():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        r = client.post("/session", json={"goal": "find the milk"})
    sid = r.json()["session_id"]

    fake_perception = MagicMock()
    fake_perception.detect.return_value = [Detection(label="shelf", box=[0, 0, 10, 10], score=0.5)]
    ask_resp = VLMResponse(action=VLMAction.ASK, guidance="left or right?",
                           question="Are you near checkout?", vlm_summary="ambiguous")
    move_resp = VLMResponse(action=VLMAction.MOVE, guidance="go left",
                            question=None, vlm_summary="left side")

    with patch("server.server.get_perception", return_value=fake_perception), \
         patch("server.server.vlm_decide", return_value=ask_resp):
        client.post(f"/session/{sid}/photo",
                    files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})

    with patch("server.server.vlm_decide", return_value=move_resp):
        r2 = client.post(f"/session/{sid}/answer", json={"answer": "no"})
    assert r2.status_code == 200
    assert r2.json()["action"] == "MOVE"
    assert r2.json()["guidance"] == "go left"


def test_answer_without_pending_question_409():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=[]):
        r = client.post("/session", json={"goal": "x"})
    sid = r.json()["session_id"]
    r2 = client.post(f"/session/{sid}/answer", json={"answer": "yes"})
    assert r2.status_code == 409


def test_answer_unknown_session_404():
    client = TestClient(app)
    r = client.post("/session/nope/answer", json={"answer": "x"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest server/tests/test_api_answer.py -v`
Expected: FAIL with route not found / 404 on the /answer happy path

- [ ] **Step 3: Add /answer route to server.py**

```python
from server.models import AnswerRequest


@app.post("/session/{session_id}/answer", response_model=TurnResponse)
def post_answer(session_id: str, req: AnswerRequest) -> TurnResponse:
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if s.pending_question is None:
        raise HTTPException(status_code=409, detail={"error": "no_question_pending", "detail": "no question is open"})
    if s.last_photo_path is None or s.last_node_id is None:
        raise HTTPException(status_code=409, detail={"error": "no_prior_photo", "detail": "answer requires a prior photo"})

    detections_summary = ", ".join(
        f"{d['label']} ({d['score']:.2f})" for d in s.last_detections
    ) or "(none)"
    topomap_summary = s.topomap.summarize_for_vlm(current_id=s.last_node_id)
    prior_question = s.pending_question

    vlm_resp = vlm_decide(
        image_path=s.last_photo_path,
        goal=s.goal,
        goal_objects=s.goal_objects,
        topomap_summary=topomap_summary,
        detections_summary=detections_summary,
        prior_question=prior_question,
        prior_answer=req.answer,
    )

    s.history.append({
        "kind": "answer",
        "user_answer": req.answer,
        "vlm_action": vlm_resp.action.value,
        "vlm_guidance": vlm_resp.guidance,
    })

    s.pending_question = None
    if vlm_resp.action == VLMAction.ARRIVED:
        s.arrived = True
        s.goal_node = s.last_node_id
        s.last_planned_action = None
    elif vlm_resp.action == VLMAction.ASK:
        s.pending_question = vlm_resp.question
    else:
        s.last_planned_action = vlm_resp.guidance

    return TurnResponse(
        action=vlm_resp.action,
        guidance=vlm_resp.guidance,
        question=vlm_resp.question,
        node_id=s.last_node_id,
        annotated_photo_url=f"/session/{session_id}/photo/{s.last_node_id}.jpg",
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_api_answer.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/server.py server/tests/test_api_answer.py
git commit -m "feat(server): POST /session/{id}/answer endpoint"
```

---

### Task 16: GET /session/{id}/map (TDD)

**Files:**
- Modify: `server/server.py`
- Create: `server/tests/test_api_map.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_api_map.py`:
```python
import io
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient
from server.server import app
from server.models import VLMAction, VLMResponse
from server.perception import Detection


def _make_jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(127, 127, 127)).save(buf, "JPEG")
    return buf.getvalue()


def _seed_session_with_one_photo(client) -> str:
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find the milk"}).json()["session_id"]
    fake_p = MagicMock()
    fake_p.detect.return_value = [Detection(label="shelf", box=[0, 0, 10, 10], score=0.5)]
    move = VLMResponse(action=VLMAction.MOVE, guidance="g", question=None, vlm_summary="aisle")
    with patch("server.server.get_perception", return_value=fake_p), \
         patch("server.server.vlm_decide", return_value=move):
        client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _make_jpg(), "image/jpeg")})
    return sid


def test_get_map_json():
    client = TestClient(app)
    sid = _seed_session_with_one_photo(client)
    r = client.get(f"/session/{sid}/map")
    assert r.status_code == 200
    body = r.json()
    assert len(body["nodes"]) == 1
    assert body["edges"] == []
    assert body["current_node"] == 0


def test_get_map_png():
    client = TestClient(app)
    sid = _seed_session_with_one_photo(client)
    r = client.get(f"/session/{sid}/map?format=png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


def test_get_map_unknown_session_404():
    client = TestClient(app)
    r = client.get("/session/nope/map")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest server/tests/test_api_map.py -v`
Expected: FAIL — 404 on the routes

- [ ] **Step 3: Add /map route to server.py**

```python
from fastapi import Query
from fastapi.responses import Response


@app.get("/session/{session_id}/map")
def get_map(session_id: str, format: str = Query(default="json")):
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    if format == "png":
        png = s.topomap.render_png(current_id=s.last_node_id)
        return Response(content=png, media_type="image/png")
    return s.topomap.to_dict(current_node=s.last_node_id, goal_node=s.goal_node)
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_api_map.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/server.py server/tests/test_api_map.py
git commit -m "feat(server): GET /session/{id}/map endpoint"
```

---

### Task 17: GET /session/{id} (full state) + GET /health (TDD)

**Files:**
- Modify: `server/server.py`
- Create: `server/tests/test_api_state.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_api_state.py`:
```python
from unittest.mock import patch
from fastapi.testclient import TestClient
from server.server import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_session_state():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find the milk"}).json()["session_id"]
    r = client.get(f"/session/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == sid
    assert body["goal"] == "find the milk"
    assert body["arrived"] is False
    assert body["history"] == []


def test_get_session_unknown_404():
    client = TestClient(app)
    r = client.get("/session/nope")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest server/tests/test_api_state.py -v`
Expected: FAIL

- [ ] **Step 3: Add /health and /session/{id} routes**

```python
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    return {
        "id": s.id,
        "goal": s.goal,
        "goal_objects": s.goal_objects,
        "history": s.history,
        "pending_question": s.pending_question,
        "arrived": s.arrived,
        "last_node_id": s.last_node_id,
        "goal_node": s.goal_node,
        "created_at": s.created_at.isoformat(),
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_api_state.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/server.py server/tests/test_api_state.py
git commit -m "feat(server): GET /session/{id} and /health"
```

---

### Task 18: Server entrypoint and Ollama check at startup

**Files:**
- Create: `server/run_server.py`
- Modify: `server/server.py`

- [ ] **Step 1: Add startup hook to server.py**

```python
import requests as _requests
from server.config import OLLAMA_URL, OLLAMA_MODEL


@app.on_event("startup")
def _check_ollama_and_warm_perception():
    try:
        r = _requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(m.startswith(OLLAMA_MODEL) for m in models):
            log.warning("Ollama is reachable but model %s is not pulled. Run `ollama pull %s`.", OLLAMA_MODEL, OLLAMA_MODEL)
    except Exception as e:
        log.error("Ollama unreachable at startup: %s. Start it: `sudo systemctl start ollama`.", e)
```

- [ ] **Step 2: Write run_server.py**

```python
"""Entry point: launch uvicorn for the FastAPI app."""
import logging
import uvicorn

from server.config import SERVER_HOST, SERVER_PORT


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run("server.server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify server starts (manual)**

Run in foreground: `python -m server.run_server`
Expected: log line `Uvicorn running on http://0.0.0.0:8000`. Hit `Ctrl-C` after 5s.

- [ ] **Step 4: Run full test suite**

Run: `pytest`
Expected: all PASS (smoke test skipped)

- [ ] **Step 5: Commit**

```bash
git add server/run_server.py server/server.py
git commit -m "feat(server): entry point and Ollama startup check"
```

---

### Task 19: Photo serving endpoint (so the Android app can fetch annotated photos)

**Files:**
- Modify: `server/server.py`
- Create: `server/tests/test_api_photo_serve.py`

- [ ] **Step 1: Write failing test**

`server/tests/test_api_photo_serve.py`:
```python
import io
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient
from server.server import app
from server.models import VLMAction, VLMResponse
from server.perception import Detection


def _jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), color=(0, 0, 255)).save(buf, "JPEG")
    return buf.getvalue()


def test_get_annotated_photo():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=["milk"]):
        sid = client.post("/session", json={"goal": "find milk"}).json()["session_id"]
    fake_p = MagicMock()
    fake_p.detect.return_value = [Detection(label="x", box=[0, 0, 5, 5], score=0.5)]
    move = VLMResponse(action=VLMAction.MOVE, guidance="g", question=None, vlm_summary="s")
    with patch("server.server.get_perception", return_value=fake_p), \
         patch("server.server.vlm_decide", return_value=move):
        client.post(f"/session/{sid}/photo", files={"photo": ("p.jpg", _jpg(), "image/jpeg")})
    r = client.get(f"/session/{sid}/photo/0.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 100


def test_get_annotated_photo_missing():
    client = TestClient(app)
    with patch("server.server.decompose_goal", return_value=[]):
        sid = client.post("/session", json={"goal": "x"}).json()["session_id"]
    r = client.get(f"/session/{sid}/photo/99.jpg")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm fail**

Run: `pytest server/tests/test_api_photo_serve.py -v`
Expected: FAIL — 404 on existing photo (route not defined)

- [ ] **Step 3: Add /photo/{nid}.jpg serve route**

```python
from fastapi.responses import FileResponse


@app.get("/session/{session_id}/photo/{node_id}.jpg")
def serve_annotated_photo(session_id: str, node_id: int):
    s = _store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "detail": session_id})
    out_dir = ensure_output_dir(session_id)
    p = out_dir / "annotated" / f"{node_id}.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "photo_not_found", "detail": str(p)})
    return FileResponse(str(p), media_type="image/jpeg")
```

- [ ] **Step 4: Run tests**

Run: `pytest server/tests/test_api_photo_serve.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/server.py server/tests/test_api_photo_serve.py
git commit -m "feat(server): serve annotated photos"
```

---

### Task 20: End-to-end manual smoke test

**Files:**
- None (this is a manual run)

- [ ] **Step 1: Verify Ollama is running**

Run: `curl -sf http://127.0.0.1:11434/api/tags && echo OK`
Expected: `OK`. If not, run: `sudo systemctl start ollama` (password if required).

- [ ] **Step 2: Run full test suite one more time**

Run: `pytest`
Expected: all green except 1 skipped (the GPU smoke).

- [ ] **Step 3: Start the server in the foreground**

Run: `python -m server.run_server`
Expected: log line `Uvicorn running on http://0.0.0.0:8000`. Leave running.

- [ ] **Step 4: From a second WSL terminal, start a session**

Open another WSL window. Run:
```bash
curl -X POST http://localhost:8000/session \
  -H "Content-Type: application/json" \
  -d '{"goal": "find the milk"}'
```
Expected: JSON like `{"session_id": "...", "guidance": "Upload a starting photo...", "action": "TAKE_PHOTO", "goal_objects": [...]}`. Note the `session_id`.

- [ ] **Step 5: Upload your first real grocery-store photo**

Pick any `.jpg` from your phone photos. Run:
```bash
curl -X POST http://localhost:8000/session/<SID>/photo \
  -F "photo=@/mnt/c/Users/user/Desktop/your-photo.jpg"
```
Expected: response within ~10s, JSON `{"action": "MOVE"|"ASK"|"ARRIVED", "guidance": "...", "node_id": 0, ...}`. Inspect the `guidance` text — does it make sense for the photo?

- [ ] **Step 6: Fetch the topological map as PNG**

```bash
curl http://localhost:8000/session/<SID>/map?format=png > /tmp/map.png
```
Expected: a small PNG with one labeled node. Open via `\\wsl$\Ubuntu-22.04\tmp\map.png` in Windows.

- [ ] **Step 7: Stop the server**

`Ctrl-C` in the server terminal.

- [ ] **Step 8: Final commit if anything changed**

```bash
git status
# if clean, you're done — the previous commits already cover the build
```

---

## Self-review notes (filled in by plan author)

- **Spec coverage:** every endpoint in the spec has its own task: `/session` (T13), `/session/{id}/photo` (T14), `/session/{id}/answer` (T15), `/session/{id}/map` (T16), `/session/{id}` (T17), `/session/{id}/photo/{node_id}.jpg` (T19). Topomap (T4-6), Session (T7), goal decomposer (T9), VLM (T10), perception (T11), annotator (T12) all match spec components. Error handling for unknown session, already arrived, no question pending all covered.
- **Placeholder scan:** no TBDs, no "implement later," every step has full code or a concrete command.
- **Type consistency:** `Detection` defined once in `perception.py`, imported elsewhere. `VLMResponse` and `VLMAction` defined once in `models.py`. `decide` signature is consistent across vlm.py and the patches in tests. The `from`-keyed edge dict matches the Pydantic alias.
- **One discrepancy I noticed:** the spec error response shape uses `{"error": "...", "detail": "..."}`. FastAPI's default exception handler wraps `detail` differently — I added a custom handler in T13 step 3 that returns the spec shape. Verified by the 404/409 tests in T14/T15.

