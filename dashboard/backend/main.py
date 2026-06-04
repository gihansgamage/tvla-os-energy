from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import subprocess
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

@app.get("/api/analyses")
def list_analyses():
    if not RESULTS_DIR.exists():
        return []
    
    analyses = []
    for item in RESULTS_DIR.iterdir():
        if item.is_dir() and item.name.startswith("analysis_"):
            try:
                summary_file = item / "summary.json"
                if summary_file.exists():
                    analyses.append({
                        "id": item.name,
                        "created_at": item.stat().st_ctime,
                    })
            except Exception as e:
                print(f"Error reading {item}: {e}")
                
    analyses.sort(key=lambda x: x["created_at"], reverse=True)
    return analyses

@app.get("/api/analyses/{analysis_id}/summary")
def get_summary(analysis_id: str):
    summary_path = RESULTS_DIR / analysis_id / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    with open(summary_path, "r") as f:
        return json.load(f)

@app.get("/api/analyses/{analysis_id}/plot/{plot_name}")
def get_plot(analysis_id: str, plot_name: str):
    # Security: Ensure plot_name doesn't contain path traversal
    if ".." in plot_name or "/" in plot_name:
        raise HTTPException(status_code=400, detail="Invalid plot name")
        
    plot_path = RESULTS_DIR / analysis_id / "plots" / plot_name
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(plot_path)

@app.post("/api/analyze")
def run_analysis():
    script_path = SCRIPTS_DIR / "analyze_traces.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail="Analyze script not found")
    
    try:
        # Run analyze_traces.py
        process = subprocess.run(
            ["python3", str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
             raise HTTPException(status_code=500, detail=f"Script failed: {process.stderr}")
        return {"status": "success", "message": "Analysis completed successfully.", "output": process.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
