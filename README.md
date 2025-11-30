# Type 2 Diabetes Prediction - Deployment

This repository contains a FastAPI app (`main.py`) that serves a diabetes prediction model and SHAP explanations.

Quick local run (virtualenv):

```bash
# create virtualenv (if you don't already have one)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Docker (build and run):

```bash
# build
docker build -t diabetes-app:v1 .

# run
docker run -p 8000:8000 --env MISTRAL_API_KEY="your_key_here" diabetes-app:v1
```

Deploy options:
- Render / Railway: push to Git and connect the repo; use the provided `Dockerfile` or set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Heroku: use the included `Procfile` and deploy with container or Python buildpack.

Notes & recommendations:
- Keep secret keys out of the repository; use environment variables (e.g., `MISTRAL_API_KEY`) or a secrets manager.
- Consider storing large model artifacts externally (S3/Blob) and loading them at startup instead of committing them to Git.
- If `shap` installation fails in Docker on alpine or minimal images, install additional system libs (e.g., `libgomp1` or `libomp-dev`) as needed.
