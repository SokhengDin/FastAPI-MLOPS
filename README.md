# Irrigation MLOps

ML system for irrigation need prediction (High / Medium / Low) built on FastAPI, Next.js, MLflow, and DVC.

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI + uvicorn |
| Frontend | Next.js 16 / React 19 / Tailwind v4 |
| Experiment tracking | MLflow 3.x (basic-auth) |
| Data versioning | DVC + MinIO S3 |
| Packaging | uv + pyproject.toml |
| Containers | Docker + Docker Compose |

## Project Structure

```
.
├── api/                  # FastAPI application
│   ├── core/             # Config, settings, in-memory job store
│   ├── handlers/         # Route handlers (predict, train, metrics)
│   ├── middleware/        # Rate limiting, API key auth
│   └── schemas/          # Pydantic request/response models
├── pipeline/             # ML pipeline stages
│   ├── preprocess_pipeline.py   # Encode, split, sample weights
│   ├── train_pipeline.py        # Train, log to MLflow, feature importance
│   ├── evaluation_pipeline.py   # Classification report, metrics.json
│   └── predict_pipeline.py      # Inference with MLflow tracing
├── web/                  # Next.js dashboard
│   └── app/
│       ├── components/   # UI primitives and uPlot charts
│       └── page.js       # Main dashboard (Overview / Model / History tabs)
├── data/                 # Raw CSV data tracked by DVC, not git
├── models/               # Trained artifacts — gitignored, mounted as volume
├── notebooks/            # Original competition notebook
├── utils/                # Shared logger
├── main.py               # FastAPI entry point
├── params.yaml           # Single source of truth for all hyperparams
├── dvc.yaml              # DVC pipeline stage definitions
├── Dockerfile            # API image
├── Dockerfile.mlflow     # MLflow image with auth + boto3
└── docker-compose.yml    # Orchestration
```

## Quick Start

```bash
# 1. Copy and fill env
cp .env.example .env

# 2. Pull data from MinIO via DVC
dvc pull

# 3. Run locally
uv run uvicorn main:app --reload        # API  → :8000
cd web && npm install && npm run dev    # Web  → :3000
docker compose up mlflow                # MLflow → :5001
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/predict` | — | Predict irrigation need (rate limited 20/min) |
| POST | `/api/v1/train` | `x-api-key` | Start background training job |
| GET | `/api/v1/train/{job_id}/status` | — | Poll training progress + logs |
| GET | `/api/v1/train/history` | — | All training runs this session |
| GET | `/api/v1/metrics` | — | Model evaluation metrics |
| GET | `/api/v1/metrics/model-info` | — | Current model + MLflow run info |
| GET | `/api/v1/metrics/feature-importance` | — | Feature importance scores |

## Deployment

```bash
docker compose up -d --build
```

Services exposed publicly (joined to `alpha_network`):
- `web` → port 3000
- `api` → port 8000

Services internal only (no host port binding):
- `mlflow` → accessible only between containers

> **CI/CD not configured yet.** Deployment is currently manual via `docker compose` on the server. Automated deploy on push to `main` is planned.

## Environment Variables

See `.env.example` for the full list. Key ones:

```
MLFLOW_URI                  MLflow tracking server URL
MLFLOW_TRACKING_USERNAME    MLflow basic auth username
MLFLOW_TRACKING_PASSWORD    MLflow basic auth password
MINIO_ENDPOINT              MinIO S3 endpoint
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
API_KEY                     x-api-key required for /train endpoint
NEXT_PUBLIC_API_URL         API URL used by the frontend
```
