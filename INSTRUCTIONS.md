# How to run the server

Everything is already installed. These are just the runtime steps.

## 1. Open WSL

```powershell
wsl -d Ubuntu-22.04
```

## 2. Make sure Ollama is up

```bash
curl -sf http://127.0.0.1:11434/api/tags && echo OK
```

If that fails:

```bash
sudo systemctl start ollama
```

## 3. Activate the conda env and `cd` into the project

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
cd /mnt/c/Users/user/Desktop/school/navigator
```

## 4. (Optional) run the tests

```bash
pytest
```

Should print `35 passed, 1 skipped`. The skipped one is the GPU smoke test — to include it: `SMOKE=1 pytest`.

## 5. Start the server

```bash
python -m server.run_server
```

Wait for `Uvicorn running on http://0.0.0.0:8000`. Leave this terminal alone.

## 6. From a second WSL terminal, drive the API

Start a session:

```bash
curl -X POST http://localhost:8000/session \
  -H "Content-Type: application/json" \
  -d '{"goal": "find the milk"}'
```

Note the `session_id` it returns, then upload a photo (first call is slow, ~30–60 s while perception + VLM warm up):

```bash
curl --max-time 300 -X POST http://localhost:8000/session/<SID>/photo \
  -F "photo=@/mnt/c/Users/user/Desktop/your-photo.jpg"
```

Get the topological map as PNG:

```bash
curl http://localhost:8000/session/<SID>/map?format=png > /tmp/map.png
```

Open it from Windows at `\\wsl$\Ubuntu-22.04\tmp\map.png`.

## 7. Stop the server

`Ctrl-C` in the server terminal.

---

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/session` | Start a session. Body: `{"goal": "..."}` |
| `POST` | `/session/{id}/photo` | Upload a photo. Multipart `photo=@file.jpg` |
| `POST` | `/session/{id}/answer` | Reply to an `ASK`. Body: `{"answer": "..."}` |
| `GET` | `/session/{id}/map` | Topomap as JSON. `?format=png` for image |
| `GET` | `/session/{id}/photo/{nid}.jpg` | Annotated photo for node `nid` |
| `GET` | `/session/{id}` | Full session state |
| `GET` | `/health` | Liveness check |

## If something breaks

- **Server crashes mid-photo with no output** → likely VRAM OOM. Quit other GPU users (Chrome, games) and retry. The `unigoal` env intentionally skips SAM load to fit in 8 GB.
- **`curl: (52) Empty reply from server`** → server died. Check the server terminal for a Python traceback or `Killed`.
- **First `/photo` returns the fallback "I had trouble..." guidance** → llama3.2-vision didn't return parseable JSON. Try again; it usually settles after the first warm call.
- **`POST /session` hangs** → Ollama not reachable. Re-run step 2.

## Where files live

- Photos and annotated photos: `/home/user/UniGoal/output/sessions/{session_id}/`
- Model weights (don't move): `/home/user/UniGoal/data/models/`
- Code: this repo (`server/`)
