# UniGoal Install Progress

## Decisions
- Path: **A — WSL2 + Ubuntu**
- Dataset: skip HM3D for now (too large, ~130GB). Add later if benchmark eval needed.
- LLM: Ollama (local llama3.2-vision)
- LLM API alternative: available via `configs/config_habitat.yaml` if Ollama too slow

## Steps

- [x] 1. Clone https://github.com/bagh2178/UniGoal.git
- [x] 2. Install WSL2 + Ubuntu 22.04 (`wsl --install -d Ubuntu-22.04 --no-launch`) — 2026-04-29
- [x] 3. ~~REBOOT WINDOWS~~ — not needed, WSL2 was already enabled
- [x] 4. First-launch Ubuntu, create UNIX user/password — user `user` (uid 1000, sudo) exists
- [x] 5. Inside WSL: install build tools — already present (build-essential, cmake, git, wget, curl, unzip, pkg-config)
- [ ] 6. Inside WSL: install Miniconda
- [ ] 7. Inside WSL: `conda create -n unigoal python==3.8`
- [~] 8. CUDA toolkit — defer. habitat-sim conda install pulls cuda runtime; install `cuda-nvcc` dev tools from nvidia channel only when compiling pytorch3d/detectron2.
- [x] 9. Cloned fresh into ~/UniGoal in WSL
- [x] 10. habitat-sim 0.2.3 (conda) + habitat-lab 0.2.3 (pip -e third_party/habitat-lab) installed. pytorch 2.1.2+cu118 installed (cuda available, RTX 3080 detected). nvcc 11.8 installed in env.
- [x] 11. LightGlue, detectron2, pytorch3d 0.7.9 (compiled for sm_86)
- [x] 12. Grounded-SAM @ 5cb813f, segment_anything (-e), GroundingDINO (-e, CUDA ext compiled)
- [x] 13. SAM (2.4G) + GroundingDINO (662M) weights in ~/UniGoal/data/models/
- [x] 14. requirements.txt + faiss-gpu (conda pytorch channel)
- [x] 15. Ollama 0.22.0 (systemd service on 127.0.0.1:11434), llama3.2-vision:latest (7.8G) pulled
- [x] 16. Smoke test passed — `python main.py --help` prints argparse usage cleanly

## Install complete 2026-04-29

## Next steps (not yet done)
- Datasets — skipped intentionally (HM3D ~130G). Add later if benchmark eval is needed (see README Step 2).
- Run real inference: `cd ~/UniGoal && conda activate unigoal && python main.py --goal_type text --goal "..."`
- VRAM caveat: llama3.2-vision is 11B; on 8GB RTX 3080 Laptop it'll use most of VRAM. If OOM at inference, switch to API mode in `configs/config_habitat.yaml`.

## Project: in-store grocery navigation app

**Goal:** Android app where the user uploads phone photos to a laptop-side server; server uses UniGoal's perception + llama3.2-vision to give navigation guidance ("turn left", "you're at the milk"). Build the backend first as an HTTP server, validate via curl, then wire Android.

### Documents
- **Design spec:** `docs/superpowers/specs/2026-04-29-unigoal-photo-demo-design.md`
- **Implementation plan:** `docs/superpowers/plans/2026-04-29-unigoal-store-nav-server.md`

### Resume tomorrow
1. Open Claude Code in this same directory: `C:\Users\user\Desktop\school\navigator`
2. Tell Claude: *"Resume the in-store-nav backend build — execute the plan."*
3. Claude will pick subagent-driven vs inline execution, then start at Task 1 of the plan.
4. Build target: `server/` directory inside this repo. ~20 tasks, ~2-3 hours.

### Dependencies that are already ready
- WSL Ubuntu 22.04 + conda env `unigoal` (Python 3.8)
- UniGoal at `/home/user/UniGoal/` with model weights downloaded
- Ollama 0.22.0 systemd service running on `127.0.0.1:11434`
- llama3.2-vision pulled

### What still needs `pip install` (first thing the plan does)
fastapi, uvicorn, python-multipart, networkx, requests, pytest, pytest-asyncio, httpx — all in `requirements-server.txt` (will be created by Task 1).

## Hardware
- GPU: RTX 3080 Laptop, 8GB VRAM, driver 546.80 (supports CUDA up to 12.3)
- Will use CUDA 11.8 to match habitat-sim 0.2.3 binaries
