# CampusGPT

AI-powered smart university information platform using Large Language Models and Retrieval-Augmented Generation (RAG).

CampusGPT is a local campus information assistant. It combines a Next.js chat interface with a FastAPI service, a retrieval-augmented generation (RAG) pipeline, a local ChromaDB collection, and Ollama for answer generation.

## Architecture

1. The browser submits a question to the FastAPI `POST /ask` endpoint.
2. The backend classifies and routes the question. Timetable and other structured routes are handled by the project code; document questions use the existing retrieval pipeline.
3. Document retrieval uses BGE-M3 embeddings and the BGE reranker against the `aiml_knowledge` ChromaDB collection.
4. When generation is needed, the backend calls the configured local Ollama model. The current default is `qwen2.5:3b`.
5. The API returns an answer and structured source metadata. Health and interactive API documentation are available at `/health` and `/docs`.

The backend entry point is `backend.main:app`. The frontend is in `frontend/campus-gpt/` and calls the backend through `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## Prerequisites

- Python 3.10 was used in the previously recorded focused regression runs. Other Python versions have not been verified; this audit shell has no accessible Python interpreter or package inventory.
- Node.js and pnpm. The frontend package declares pnpm `11.24.0` as its package manager.
- Ollama, with the configured generation model available locally.
- The compatible local ChromaDB collection and approved source material. These are intentionally excluded from this repository.
- Network access may be needed on first use to download the BGE embedding and reranker model weights if they are not already cached.

### Backend dependency manifest limitation

This checkout does not contain a Python dependency manifest such as `requirements.txt` or `pyproject.toml`. As a result, a clean backend installation cannot currently be reproduced from the repository alone. Do not treat a developer machine's `pip freeze` as a project lockfile. Add and review a pinned dependency manifest before claiming reproducible backend installation.

## Local setup

### 1. Prepare local data and services

The university corpus, generated evaluation materials, and local vector database are not distributed here. Obtain authorization for the source material, prepare the compatible `aiml_knowledge` collection using the project's reviewed local workflow, and place it in the expected local database location. Do not run database preparation or ingestion against valuable local data without first reviewing the script's behavior and keeping a backup.

Install and start Ollama separately, and make the model named by `CAMPUSGPT_OLLAMA_MODEL` available. The default configured model is `qwen2.5:3b`. The pipeline also loads `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` through its model libraries.

### 2. Install backend dependencies

Create and activate a virtual environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The backend package installation step is not reproducible yet because this repository has no Python dependency manifest. Use only a reviewed dependency list for this checkout; do not guess package versions from another environment.

### 3. Configure and start the backend

The backend reads optional settings from process environment variables. In particular, `OLLAMA_HOST` selects the Ollama endpoint and `CAMPUSGPT_OLLAMA_MODEL` selects the generation model; defaults are defined in `scripts/generate_answer.py`. Additional context, token, reranker, and CPU thread settings are also defined in the backend code. No backend `.env.example` is currently supplied.

After the reviewed Python dependencies, Ollama model, and local ChromaDB collection are available, run from the project root:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`; open `http://127.0.0.1:8000/docs` for the API interface.

### 4. Configure and start the frontend

From `frontend/campus-gpt/`, copy `.env.example` to `.env.local` if you need to change the API URL. The example points to the local backend and contains no credentials.

```powershell
cd frontend/campus-gpt
pnpm install
pnpm dev
```

The package's `dev` script runs Next.js. Open the local URL printed by Next.js (normally `http://localhost:3000`).

## Configuration

| Setting | Purpose | Default in code |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Frontend API origin | `http://localhost:8000` |
| `OLLAMA_HOST` | Ollama service address | `http://localhost:11434` |
| `CAMPUSGPT_OLLAMA_MODEL` | Local answer-generation model | `qwen2.5:3b` |

Other optional `CAMPUSGPT_*` settings and reranker settings are read in `scripts/generate_answer.py` and `scripts/reranker.py`; review those files for the current defaults before overriding them. Never commit actual credentials or machine-specific environment files.

## Source data and limitations

- Local university source data, evaluation question sets/results, audit reports, and ChromaDB files are excluded by `.gitignore` and are not included in the release candidate.
- The shared `aiml_knowledge` ChromaDB collection serves document routes with department filters where applicable; the dedicated timetable and faculty-information sources in this release are AIML-specific, and equivalent timetable datasets for other departments are not included.
- Answer quality and source availability depend on the compatible local corpus and ChromaDB collection supplied by the operator.
- The repository currently has no Python dependency lock/manifest and no root `LICENSE` file. Resolve both before describing the project as reproducible or granting reuse rights.
- The frontend's API URL is public client configuration, not a place for secrets. Keep credentials on the backend or in local environment files.

## Repository safety

`.gitignore` excludes university data, local databases, evaluation and audit outputs, environment files, credentials, temporary logs, backups, model weights, and frontend build/dependency directories. Existing test and evaluation helper scripts are also excluded until their fixtures have been reviewed for embedded institutional data. Before staging a release, inspect `git status --short --ignored` and explicitly stage only the reviewed source files. Do not use `git add .`.
