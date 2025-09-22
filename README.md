cat > README.md <<'README'
# RAG Indexer — FastAPI + Ollama Reranker + React frontend

Lightweight RAG (retrieval-augmented generation) demo: FastAPI backend that retrieves from a Postgres-backed vector store, re-ranks candidates via Ollama, synthesizes an answer, and a React frontend. Docker Compose is included to run the full stack (recommended). You can also run backend and frontend locally for development.

---

## Contents

- `backend_server.py` — FastAPI RAG backend
- `frontend/` — React app
- `docker-compose.yml` + `nginx.conf` — full-stack compose with nginx reverse proxy
- `Dockerfile` (backend/frontend) — per-service images
- `.env` — environment variables (not committed)
- `ollama token.txt` — **must not be committed** (if present, remove & rotate)

---

## Quick status checks

Verify services (after Docker Compose or running locally):

- Backend health: `http://localhost:8081/healthz`
- Ollama default API: `http://localhost:11434/api/health` (or check with a simple `curl` to `/api/chat`)
- Frontend: `http://localhost:3000` (dev) or `http://localhost` (when served via nginx in the compose)

---

## 1 — Security first (important)

1. **Do not commit secrets** (`.env`, `ollama token.txt`, access tokens). Add them to `.gitignore`.
2. If a secret was accidentally committed:
   - **Immediately rotate/revoke** it (very important).
   - Use `git-filter-repo` or BFG in a mirror to purge history, then force-push cleaned refs (admin coordination required).
   - See “Cleaning history” section near the bottom of this README for commands.

---

## 2 — Example `.env`

Create `.env` in repo root or export env vars in your environment:

```env
# .env (example)
OLLAMA_BASE_URL=http://localhost:11434
GEN_MODEL=smollm:135m
EMBED_MODEL=all-minilm
EMBED_DIM=384

PGHOST=
PGPORT=
PGDATABASE=
PGUSER=
PGPASSWORD=
PG_TABLE=

# Optional
```
3 — Run everything with Docker Compose (recommended)

From repo root:

# Build & start everything (foreground)
docker compose up --build

# Or detached
docker compose up --build -d

# Follow logs for a service
docker compose logs -f backend
docker compose logs -f ollama
docker compose logs -f nginx

Common Docker issues & fixes

Docker daemon not running → Start Docker Desktop / Docker Engine.

Ollama image pull denied / manifest unknown → ensure ollama/ollama:latest is accessible from your environment or run a local Ollama server and point OLLAMA_BASE_URL to it. Alternatively, use the host Ollama binary / service.

Frontend Docker build fails on npm ci with lockfile mismatch:

Locally: cd frontend && npm install then commit the resulting package-lock.json (if you control the lockfile).

Or remove frontend/package-lock.json before building if you intentionally want a fresh install (note: builds become non-reproducible).

Or run docker compose build frontend --no-cache after fixing the lockfile.

4 — Run backend locally (no Docker)

Create & activate a venv (recommended):

python3 -m venv .venv
source .venv/bin/activate         # WSL / macOS / Linux
# Windows (PowerShell): .venv\Scripts\Activate.ps1


Install dependencies:

pip install -r requirements.txt


Start backend:

uvicorn backend_server:app --host 0.0.0.0 --port 8081 --reload


Health & test:

curl -s http://localhost:8081/healthz | jq .

5 — Run frontend locally (dev)

From frontend/:

# If using environment variable (Linux/WSL/macOS)
REACT_APP_API_BASE=http://localhost:8081 npm start

# PowerShell:
$env:REACT_APP_API_BASE='http://localhost:8081'; npm start


Alternative: add "proxy": "http://localhost:8081" to frontend/package.json so create-react-app proxies API calls.

If index.html missing or npm start complains, ensure frontend/public/index.html exists and you are in the correct folder.

6 — Test the RAG endpoint

Use python3 -m json.tool for pretty JSON:

curl -sS -X POST "http://localhost:8081/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query":{"text":"Give me an overview of the uploaded documents"},"top_k":3}' \
| python3 -m json.tool


Expected: JSON object containing answer and sources (or an error with detail).

If you see "detail": "Retrieval/Rerank error: Server disconnected without sending a response.", check backend logs and Ollama availability (models may not be pulled).

7 — Ollama notes

Ollama server API default port: 11434. OLLAMA_BASE_URL should point to that.

If you get "model \"smollm:135m\" not found", you must pull or install that model in Ollama or change GEN_MODEL to an available model.

Test Ollama directly:

curl -sS -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm:135m","messages":[{"role":"user","content":"hello"}],"stream":false}'