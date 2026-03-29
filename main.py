"""Auto-generated FastAPI wrapper for layer_mvp_0022."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.layer_mvp_0022 import get_vaccine_search_volume_data, get_covid_clinical_trials_count, calculate_granger_causality, preprocess_search_data, preprocess_trials_data

app = FastAPI(
    title="Layer Mvp 0022",
    description="Auto-generated MVP API",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"service": "layer_mvp_0022", "status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

