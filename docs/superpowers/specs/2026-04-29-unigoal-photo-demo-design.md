# UniGoal-Driven Interactive Store-Navigation Server — Design

**Status:** Approved 2026-04-29 (revised)
**Project:** Android in-store navigation app (grocery store as primary use case)
**This phase:** Backend HTTP server, validated via curl/Postman before Android frontend wiring.

## Goal

Build the production backend for an in-store navigation assistant. A user walking through an unfamiliar grocery store opens the app, types a goal ("find the milk"), uploads a photo of their current view, and receives guidance to move/turn (or sometimes a clarifying question). After moving, they upload a new photo and the loop continues until they arrive.

The same logical loop UniGoal uses for autonomous robots — perceive → consult goal → plan move → observe → repeat — applied to a human-in-the-loop guidance system. The server holds session state (a topological map of where the user has been, plus history) so the VLM can reason across turns instead of treating each photo independently.

## Why this design (vs. alternatives we discarded)

- **Batch CLI script (Approach A from earlier brainstorm):** ruled out — user wants interactive single-photo flow, not pre-captured sequence playback.
- **Synthesized depth + pose to drive UniGoal's BEV mapping (Approach B):** ruled out for now — phone photos lack reliable pose; visual SLAM drifts in repetitive grocery aisles. Deferred to the phone-app phase, when ARCore provides 6DoF pose for free.
- **CLI session instead of HTTP server:** ruled out — the Android app needs an HTTP backend anyway. Building it as a server now means no rewrite later; curl/Postman serve as the demo client.

## Scope

**In scope:**
- FastAPI HTTP server on the laptop (WSL Ubuntu)
- Per-session state: goal, topological map, conversation history, pending question
- Endpoints to start a session, upload a photo, answer a clarifying question, fetch the current map, fetch full session state
- Per-photo pipeline: GroundingDINO + SAM detection → topological-map update → llama3.2-vision (Ollama) reasoning → structured guidance response
- In-memory session storage (lost on restart — acceptable for demo)
- Save annotated photos and a JSON log to disk for debugging

**Out of scope (deferred to phone-app phase):**
- Metric BEV map (needs depth + pose; will use ARCore when Android frontend lands)
- Cross-frame scene graph with object continuity (needs pose)
- Persistent session storage (DB / disk)
- Authentication, user accounts, rate limiting
- Real-time camera streaming (current loop is photo-by-photo)
- Multi-user concurrency (assumed: one session at a time during demo)
- Android Kotlin frontend (separate project)

## API contract

All requests/responses are JSON unless noted. Photo uploads are multipart.

### `POST /session`
Start a new session.

**Request:**
```json
{ "goal": "find the milk" }
```

**Response (200):**
```json
{
  "session_id": "9f3c1a4e",
  "guidance": "Upload a starting photo so I can see where you are.",
  "action": "TAKE_PHOTO",
  "goal_objects": ["milk", "dairy", "carton", "fridge", "shelf"]
}
```

`goal_objects` is the goal-decomposer's output, returned for transparency / debugging.

### `POST /session/{id}/photo`
Upload current camera view.

**Request (multipart):**
- `photo`: image file (jpg/png), required
- `pose` (optional, future): JSON `{"x": float, "y": float, "heading_deg": float}` — ignored in this phase, stub for phone integration

**Response (200):**
```json
{
  "action": "MOVE" | "ASK" | "ARRIVED",
  "guidance": "Walk to the back wall and turn left when you reach the dairy section.",
  "question": null,           // populated only when action == "ASK"
  "node_id": 3,               // id of the topological-map node just created
  "annotated_photo_url": "/session/9f3c1a4e/photo/3.jpg"
}
```

When `action == "ARRIVED"`, the session is marked complete; further `/photo` calls return 409.

### `POST /session/{id}/answer`
Reply to a clarifying question (used after a response with `action == "ASK"`).

**Request:**
```json
{ "answer": "Yes, I can see a sign that says 'Aisle 4 - Dairy'" }
```

**Response:** same shape as `/photo` response. The answer is added to session history and the VLM is re-queried with the updated context. No new photo is uploaded; the most recent photo and its cached detections are reused (perception is not re-run).

### `GET /session/{id}/map`
Return the topological map.

- Default (`Accept: application/json` or `?format=json`): JSON nodes/edges
- `?format=png`: PNG rendering via NetworkX + matplotlib (300x300 graph layout, nodes labeled by id, edges labeled by action)

JSON shape:
```json
{
  "nodes": [
    {"id": 0, "photo": "photo/0.jpg", "detected": ["bread", "shelf"], "summary": "Aisle with bakery items", "timestamp": "..."},
    {"id": 1, "photo": "photo/1.jpg", "detected": ["produce", "tomatoes"], "summary": "...", "timestamp": "..."}
  ],
  "edges": [
    {"from": 0, "to": 1, "action": "Walked forward, turned left"}
  ],
  "current_node": 1,
  "goal_node": null
}
```

### `GET /session/{id}`
Full session state — goal, history, map, pending question. For debugging.

### Error shapes
```json
{ "error": "session_not_found", "detail": "no session with id abc123" }
```
- `400` — bad request body, missing photo, malformed JSON
- `404` — session id unknown
- `409` — operation invalid for current state (e.g., `/photo` after `ARRIVED`, `/answer` when no question pending)
- `500` — VLM/Ollama failure (with retry guidance in detail)
- `503` — Ollama unreachable at startup

## Internal architecture

### Files (all in `~/UniGoal/demo/`)
| File | Responsibility |
|---|---|
| `run_server.py` | Entrypoint. Loads models once. Starts uvicorn. |
| `server.py` | FastAPI app, route handlers. Thin — delegates to session.py. |
| `session.py` | `Session` class: goal, goal_objects, topomap, history, pending_question, arrived. `SessionStore` (in-memory dict). |
| `perception.py` | Loads GroundingDINO + SAM at startup. `detect(image, prompt_classes) → list[Detection]`. |
| `vlm.py` | Ollama HTTP client. `decide(image_path, goal, topomap_summary, history, prior_answer=None) → VLMResponse`. |
| `topomap.py` | NetworkX-based topological map. `add_node`, `add_edge`, `summarize_for_vlm`, `to_dict`, `render_png`. |
| `goal_decomposer.py` | One-time call at session start: goal text → list of GroundingDINO prompt classes. |
| `prompts.py` | Centralized prompt templates. |
| `models.py` | Pydantic models for request/response bodies, internal types (Detection, VLMResponse, Node, Edge). |
| `__init__.py` | Empty (package marker). |

### Data flow per `/photo` call

```
1. Validate session exists, not arrived, no pending question
2. Save uploaded photo to disk: output/sessions/{id}/photo/{node_id}.jpg
3. perception.detect(photo, session.goal_objects) → list[Detection]
4. topomap.add_node(photo_path, detected_objects, summary=None)
5. If prev_node exists: topomap.add_edge(prev_node, new_node, action=session.last_planned_action)
6. summary = topomap.summarize_for_vlm(current_node)
   # e.g., "You've visited 3 locations. At node 0 you saw bread+shelf,
   #  then walked left to node 1 (produce), then walked forward to node 2 (current).
   #  Goal not yet found."
7. vlm_response = vlm.decide(
       image_path=photo_path,
       goal=session.goal,
       topomap_summary=summary,
       history=session.history,
       prior_answer=None
   )
8. annotator: draw boxes, save annotated copy to output/sessions/{id}/annotated/{node_id}.jpg
9. update session.history with this turn
10. if vlm_response.action == "ASK": session.pending_question = vlm_response.question
    if vlm_response.action == "ARRIVED": session.arrived = True; session.goal_node = node_id
    else: session.last_planned_action = vlm_response.guidance  # for next edge label
11. return response JSON
```

### Goal decomposer (one-time per session)

Send a text-only request to llama3.2-vision via Ollama:

```
You are helping someone navigate a grocery store. Their goal is: "{goal}".
List 4-8 visual things they should look for to know they're close to the goal,
as a comma-separated list. Be specific. Reply ONLY with the list.

Example for "find the cookies":
cookies, biscuits, snacks aisle, packaged sweets, candy, sugar
```

Response is parsed by splitting on commas, lowercasing, stripping whitespace. If the call fails or returns garbage (>10 items, non-comma format), fall back to splitting the goal text on whitespace and using its noun phrases.

### Per-turn VLM prompt

```
You are guiding a shopper through a grocery store using their phone camera.

Goal: "{goal}"
Goal-related items to look for: {goal_objects}

What's happened so far:
{topomap_summary}

The shopper just uploaded the attached photo. In it, automatic detection found:
{detections_summary}

{prior_answer_block_if_any}

Decide the next step. Reply with EXACTLY one JSON object on one line, nothing else:

{
  "action": "ARRIVED" | "MOVE" | "ASK",
  "guidance": "<one or two sentences the shopper will read>",
  "question": "<a single yes/no or short-answer question, only if action=ASK, else null>",
  "vlm_summary": "<one phrase summarizing the location, e.g. 'aisle 4, dairy section'>"
}

Rules:
- ARRIVED only if the goal item is clearly visible in the photo (point at it in `guidance`).
- ASK if you cannot decide between two plausible directions and a yes/no answer
  from the user would resolve it.
- MOVE otherwise. Tell the user a concrete direction ("turn left and walk to the
  back wall, then take another photo").
- Do NOT invent details not in the photo or detections.
```

`{prior_answer_block_if_any}` is empty on a `/photo` call. On a `/answer` call we add:
```
The shopper just answered your earlier question "{previous_question}" with:
"{user_answer}"
```

The server parses the first JSON object on the first line of the response. On parse failure: log raw response, return action=`MOVE` with guidance=`"I had trouble understanding the scene. Try walking forward a few steps and uploading another photo."`

### Session state (in-memory)

```python
@dataclass
class Session:
    id: str
    goal: str
    goal_objects: list[str]
    topomap: TopoMap
    history: list[Turn]
    pending_question: str | None
    last_planned_action: str | None  # used to label the next edge
    arrived: bool
    created_at: datetime

class SessionStore:
    _sessions: dict[str, Session]
    def create(goal) -> Session: ...
    def get(id) -> Session: ...
```

`Turn` records each interaction:
```python
@dataclass
class Turn:
    timestamp: datetime
    kind: Literal["photo", "answer"]
    photo_path: str | None
    user_answer: str | None
    vlm_response: VLMResponse
    node_id: int
```

### Topological map

NetworkX `DiGraph`. Methods:
- `add_node(photo_path, detected_objects, vlm_summary) → node_id` (auto-incrementing int)
- `add_edge(from_id, to_id, action: str)`
- `summarize_for_vlm(current_id) → str` — produces the "what's happened so far" prose for the VLM. Walk the graph from start to current, list nodes with their summaries and edges with their actions. Keep under ~150 words.
- `render_png(highlight_current=True) → bytes` — matplotlib spring layout, nodes labeled with id, hovered tooltip = summary, edges labeled with action.
- `to_dict() → dict` — for JSON serialization.

## Error handling

| Failure | Behavior |
|---|---|
| Server starts; Ollama unreachable | Log fatal, exit non-zero. Don't start. |
| Server running; Ollama becomes unreachable mid-session | Per-request: 503 with `{"error": "vlm_unavailable", "detail": "Run `sudo systemctl start ollama` and retry"}` |
| GroundingDINO/SAM weights missing on startup | Log fatal with checked paths, exit non-zero |
| Photo upload corrupt/unreadable | 400 with detail |
| VLM returns malformed JSON | Log full response. Substitute action=MOVE, guidance="default fallback text", continue |
| GroundingDINO finds no detections (empty scene) | Don't fail — pass empty detections to VLM, let it reason from raw image |
| Same photo uploaded twice (same session) | Treat as a new node — user might be re-checking. No dedup. |
| Session id unknown | 404 |
| `/photo` after ARRIVED | 409 |
| `/answer` with no pending question | 409 |

## Output / logs

For each session, on disk under `output/sessions/{session_id}/`:
- `photo/{node_id}.jpg` — original uploaded photos
- `annotated/{node_id}.jpg` — same photos with detection boxes + overlay text
- `log.json` — append-only log of all turns (one JSON object per turn)
- `map.png` — written on each `/map?format=png` call (overwritten)
- `state.json` — written on session arrival (final state snapshot)

Server stdout: structured INFO logs (one per request), DEBUG logs (VLM raw responses) toggled by `--log-level debug`.

## Performance / hardware expectations

Hardware: RTX 3080 Laptop, 8 GB VRAM. CPU 16+ threads.

Per-photo latency budget (warm models):
- GroundingDINO + SAM: ~1.5 s
- VLM (llama3.2-vision via Ollama, 11B@Q4): ~3-6 s for a typical 1080p photo + ~200 token response
- Total: **~5-8 s per turn**

VRAM concerns:
- GroundingDINO + SAM together: ~3 GB
- Ollama VLM: ~7 GB
- Combined load > 8 GB → Ollama auto-evicts when not in use; perception runs first, releases, then VLM. We won't hold both at once. Latency penalty for VLM cold-load: +2-4 s on first request, negligible after.

If first-request latency is too high, we can pre-warm at server startup (issue a dummy VLM call after model load).

## Testing

Empirical / smoke testing only. No CI for the demo phase.

1. **Server startup** — run, verify `GET /health` returns 200, models loaded, Ollama reachable.
2. **End-to-end happy path** — start session, upload 3-5 photos via curl with a known goal that's reachable, verify ARRIVED comes back at the right photo. Manually inspect annotated images and `map.png`.
3. **Question-asking path** — craft a session where the photo is ambiguous, verify VLM uses `ASK`, verify `/answer` updates state and VLM response changes.
4. **Failure paths** — bad photo, bad session id, photo after arrival, malformed JSON.
5. **Latency check** — record per-photo time over 10 photos. If > 10 s p50, investigate.

## Launch instructions (for the user)

```bash
# 1. Open WSL
wsl

# 2. Activate env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
cd ~/UniGoal

# 3. Verify Ollama is running
curl -sf http://127.0.0.1:11434/api/tags >/dev/null && echo OK || echo "Ollama not running"

# 4. Start the server
python demo/run_server.py     # listens on 0.0.0.0:8000

# 5. From another terminal (or Postman, or eventually the Android app):
# Start a session
curl -X POST http://localhost:8000/session \
  -H "Content-Type: application/json" \
  -d '{"goal": "find the milk"}'
# → {"session_id": "abc123", "guidance": "Upload a starting photo", "action": "TAKE_PHOTO", ...}

# Upload a photo
curl -X POST http://localhost:8000/session/abc123/photo \
  -F "photo=@/mnt/c/Users/user/Desktop/photo1.jpg"
# → {"action": "MOVE", "guidance": "Walk forward to the back wall, then turn left.", ...}

# Answer a clarifying question
curl -X POST http://localhost:8000/session/abc123/answer \
  -H "Content-Type: application/json" \
  -d '{"answer": "Yes, I see a Dairy sign."}'

# Get the map as a PNG
curl http://localhost:8000/session/abc123/map?format=png > map.png

# Get full state for debugging
curl http://localhost:8000/session/abc123 | jq
```

The Android frontend will eventually hit the same endpoints with multipart photo uploads, displaying `guidance` text, rendering `question` as a text input, and showing `map.png` on demand.

## Decision points after this build

| Outcome | Next step |
|---|---|
| Pipeline works on real grocery photos | Begin Android integration: replace curl client with Kotlin OkHttp/Retrofit calls. Add ARCore pose to `/photo` (currently stub). |
| Detection misses retail items frequently | Augment goal decomposer prompt to include common retail packaging cues, or fine-tune GroundingDINO. |
| VLM guidance is unreliable | Replace VLM-driven decision with rule-based heuristic on detection bounding-box positions. Fall back gracefully. |
| Latency > 10s p50 | Move to GPU-resident model (skip Ollama eviction), consider a smaller VLM, or batch perception. |
| Sessions need persistence (production) | Add SQLite backing store keyed by session_id. |
