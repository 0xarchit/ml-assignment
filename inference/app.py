import asyncio
import json
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "template"
DATASET_DIR = PROJECT_DIR / "dataset"

MODEL_CONFIG = {
    "consumer": {
        "title": "Shopping Trends",
        "dir": OUTPUTS_DIR / "Consumer_Shopping_Trends",
        "data": DATASET_DIR / "Consumer_Shopping_Trends_2026 (6).csv",
        "url": "https://www.kaggle.com/datasets/minahilfatima12328/consumer-shopping-trends-analysis"
    },
    "f1": {
        "title": "F1 Strategy",
        "dir": OUTPUTS_DIR / "F1_Strategy",
        "data": DATASET_DIR / "f1_strategy_dataset_v4.csv",
        "url": "https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction"
    },
    "sleep": {
        "title": "Sleep Health",
        "dir": OUTPUTS_DIR / "Sleep_Health_And_Daily_Performance",
        "data": DATASET_DIR / "sleep_health_dataset.csv",
        "url": "https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset"
    },
}

class PredictRequest(BaseModel):
    features: dict[str, Any]

app = FastAPI(title="Inference Engine")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(TEMPLATE_DIR)), name="static")

ARTIFACTS = {}

def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(x) for x in obj]
    if hasattr(obj, "item"): return obj.item()
    if pd.isna(obj): return None
    return obj

def get_json(p: Path):
    if not p.exists(): return {}
    with p.open("r") as f: return json.load(f)

def get_csv_dict(p: Path):
    if not p.exists(): return []
    return pd.read_csv(p).to_dict(orient="records")

@app.on_event("startup")
async def startup():
    for key, cfg in MODEL_CONFIG.items():
        try:
            m_dir = cfg["dir"] / "model"
            met_dir = cfg["dir"] / "metrics"
            m_path = m_dir / "model.joblib"
            if not m_path.exists(): continue
            
            df = pd.read_csv(cfg["data"])
            model = joblib.load(m_path)
            feat_cols = get_json(m_dir / "feature_columns.json")
            tg_info = get_json(m_dir / "target_info.json")
            tg = tg_info.get("target_col")
            
            # Better background data prep for SHAP
            bg_df = df.head(50).copy()
            ignore = [tg.lower(), "person_id", "id", "customer_id", "driver_id"]
            bg_df = bg_df.drop(columns=[c for c in bg_df.columns if c.lower() in ignore], errors="ignore")
            
            # Ensure numeric columns are numeric and others are dummy encoded
            for c in bg_df.columns:
                if pd.api.types.is_numeric_dtype(df[c].dtype):
                    bg_df[c] = pd.to_numeric(bg_df[c], errors="coerce").fillna(df[c].median())
                else:
                    bg_df[c] = bg_df[c].astype(str)

            bg_enc = pd.get_dummies(bg_df, dtype=float).reindex(columns=feat_cols, fill_value=0.0).astype(float)
            
            # Use TreeExplainer if possible, otherwise generic Explainer
            try:
                explainer = shap.TreeExplainer(model, bg_enc)
            except:
                explainer = shap.Explainer(model, bg_enc)
            
            ARTIFACTS[key] = {
                "title": cfg["title"],
                "url": cfg["url"],
                "model": model,
                "explainer": explainer,
                "features": feat_cols,
                "labels": get_json(m_dir / "label_classes.json"),
                "target": tg_info,
                "metrics": get_csv_dict(met_dir / "final_metircs.csv"),
                "df": df,
                "path": str(cfg["data"])
            }
        except Exception as e:
            print(f"Error loading {key}: {e}")

@app.get("/")
async def index(request: Request):
    models = [{"id": k, "name": v["title"]} for k, v in ARTIFACTS.items()]
    return templates.TemplateResponse(request=request, name="index.html", context={"models": models})

@app.get("/api/info/{id}")
async def info(id: str):
    if id not in ARTIFACTS: raise HTTPException(404)
    art = ARTIFACTS[id]
    df = art["df"]
    tg = art["target"].get("target_col")
    dist = df[tg].value_counts(normalize=True).to_dict() if tg in df.columns else {}
    return clean({
        "title": art["title"],
        "url": art["url"],
        "dataset": {
            "rows": len(df),
            "cols": len(df.columns),
            "target": tg,
            "dist": dist
        },
        "model": {
            "type": art["target"].get("model_name"),
            "features": len(art["features"]),
            "labels": art["labels"]
        },
        "metrics": art["metrics"]
    })

@app.get("/api/fields/{id}")
async def fields(id: str):
    if id not in ARTIFACTS: raise HTTPException(404)
    art = ARTIFACTS[id]
    df = art["df"]
    tg = art["target"].get("target_col")
    out = []
    for c in df.columns:
        if c == tg or c.lower() in ["person_id", "id", "customer_id", "driver_id"]: continue
        t = "number" if pd.api.types.is_numeric_dtype(df[c].dtype) else "select"
        opts = [x for x in df[c].unique().tolist() if pd.notna(x)] if t == "select" else []
        out.append({"name": c, "type": t, "options": clean(opts), "default": ""})
    return out

@app.get("/api/sample/{id}")
async def sample(id: str):
    if id not in ARTIFACTS: raise HTTPException(404)
    art = ARTIFACTS[id]
    row = art["df"].sample(1).iloc[0].to_dict()
    return clean(row)

@app.post("/api/run/{id}")
async def run(id: str, req: PredictRequest):
    if id not in ARTIFACTS: raise HTTPException(404)
    art = ARTIFACTS[id]
    
    try:
        raw = req.features
        cols = art["df"].columns
        tg = art["target"].get("target_col")
        ignore = [tg.lower(), "person_id", "id", "customer_id", "driver_id"]
        row = {c: raw.get(c, art["df"][c].iloc[0]) for c in cols if c.lower() not in ignore}
        df = pd.DataFrame([row])
        
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(art["df"][c].dtype):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(art["df"][c].median())
        
        enc = pd.get_dummies(df).reindex(columns=art["features"], fill_value=0.0).astype(float)
        
        pred = await asyncio.to_thread(art["model"].predict, enc)
        prob = await asyncio.to_thread(art["model"].predict_proba, enc)
        
        shap_values = await asyncio.to_thread(art["explainer"].shap_values, enc)
        
        idx = int(pred[0])
        lbl = art["labels"][idx] if idx < len(art["labels"]) else str(idx)
        probs = {art["labels"][i] if i < len(art["labels"]) else str(i): float(p) for i, p in enumerate(prob[0])}
        
        # SHAP values for tree models (like XGBoost) often return a list for each class
        if isinstance(shap_values, list):
            sv = shap_values[idx][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, idx]
        else:
            sv = shap_values[0]
        
        contributions = {}
        for i, feat in enumerate(art["features"]):
            orig = feat.split("_")[0]
            contributions[orig] = contributions.get(orig, 0) + float(sv[i])
            
        top_contrib = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
        return {
            "result": lbl, 
            "scores": probs,
            "shap": top_contrib
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(400, str(e))

@app.get("/favicon.ico")
async def fav(): return FileResponse(str(TEMPLATE_DIR / "favicon.ico"))
