# Crop Intelligence Platform

End-to-end project for 13-crop recommendation, yield prediction, model comparison,
stacking, and ranked crop selection using the real merged datasets stored in
the `data/` folder.

## Final Dataset Files

Keep these files inside `data/`:

```text
data/
  real_crop_recommendation_13.csv
  real_crop_production_13.csv
  real_market_lookup_13.csv
  real_crop_reference_profiles_13.csv
  real_project_master_13.csv
```

The code resolves these files from the project `data/` folder first and falls
back to `C:\Users\santhosh\Downloads` only if needed.

## Structure

```text
backend/
  app/
    api/
    services/
    static/
  ml/
data/
models/
reports/
scripts/
requirements.txt
Dockerfile
```

## Verified Metrics

Latest verified training run on the real 13-crop pipeline:

- Stacking accuracy: `1.0000`
- LightGBM accuracy: `0.9962`
- CatBoost accuracy: `0.9885`
- Yield R²: `0.8581`
- Yield RMSE: `515.36`

Saved report:

```text
reports/training_metrics.json
```

## Run From VS Code Terminal

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional extras for true TabNet and SHAP:

```bash
pip install -r requirements-optional.txt
```

Check imports:

```bash
python scripts\verify_imports.py
```

Inspect the real recommendation dataset:

```bash
python scripts\inspect_dataset.py
```

Train all models:

```bash
python scripts\train.py
```

Smoke test prediction:

```bash
python scripts\smoke_predict.py
```

Smoke test API:

```bash
python scripts\smoke_api.py
```

Start the app:

```bash
python scripts\run_api.py
```

Open:

```text
http://localhost:8000
```

## Deploy On Render

Best deployment target for this project: `Render`

Reason:
- FastAPI backend + static frontend are served together
- Python ML stack fits Render better than Netlify or Vercel
- This repo includes a `render.yaml` blueprint for direct deployment

This repo is configured so Render will:
1. install dependencies
2. install Linux dependency `libgomp1` through Docker
3. train the `realistic_v2` models during Docker build
4. start the FastAPI app

Files used for deployment:
- `render.yaml`
- `runtime.txt`
- `requirements.txt`
- `Dockerfile`

After connecting the GitHub repo in Render, choose the Docker/Blueprint flow.
The Docker image will build and run the project directly.

```text
Dockerfile: ./Dockerfile
```

## API

- `GET /api/health`
- `GET /api/metadata`
- `POST /api/train`
- `POST /api/predict`

## Main Edited Files

- `backend/ml/config.py`
- `backend/ml/training.py`
- `backend/ml/inference.py`
- `backend/app/schemas.py`
- `backend/app/services/predictor.py`
- `backend/app/api/routes.py`
- `backend/app/static/app.js`
- `scripts/train.py`
- `scripts/inspect_dataset.py`

## Notes

- The recommendation model uses `real_crop_recommendation_13.csv`.
- The yield model uses `real_project_master_13.csv`.
- Ranking uses the market lookup and crop reference profile CSVs.
- If `pytorch-tabnet` is unavailable locally, the project uses a compatible fallback for that model slot so the pipeline still runs.
- SHAP is optional and may be skipped if it is not installed for the active Python version.
- `requirements.txt` contains the stable core dependencies. `requirements-optional.txt` adds the heavier optional packages.
"# hemalatha-mam-ai-project-crop" 
