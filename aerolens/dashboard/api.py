import os
import json
import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from aerolens.inference.pipeline_orchestrator import PipelineOrchestrator

app = FastAPI(title="AeroLens Sentinel Technician Dashboard API")

# Initialize orchestrator
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
orchestrator = PipelineOrchestrator()

# Mount annotated output directory to serve images to dashboard
annotated_out_dir = os.path.join(base_dir, "data")
app.mount("/data", StaticFiles(directory=annotated_out_dir), name="data")

# Mount dashboard static frontend folder
static_dir = os.path.join(base_dir, "dashboard", "static")
os.makedirs(static_dir, exist_ok=True)

@app.post("/upload_frame")
async def upload_frame(
    file: UploadFile = File(...),
    aircraft_id: str = Form("AERO-DEV-01"),
    zone: str = Form("wing_spar")
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")
            
        result = orchestrator.process_frame(frame, aircraft_id=aircraft_id, zone=zone)
        # Modify the image path to be accessible via URL
        if result["annotated_image_path"]:
            # e.g. path is .../data/annotated_output/name.jpg -> convert to web path
            fn = os.path.basename(result["annotated_image_path"])
            result["annotated_image_url"] = f"/data/annotated_output/{fn}"
        else:
            result["annotated_image_url"] = ""
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inspection_history")
def get_inspection_history(limit: int = 100, offset: int = 0):
    try:
        records = orchestrator.logger.get_history(limit=limit, offset=offset)
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/criticality_map")
def get_criticality_map():
    try:
        return orchestrator.scorer.zone_criticality
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hardware_status")
def get_hardware_status():
    try:
        # Load diagnostics & benchmarks
        diag_path = os.path.join(base_dir, "telemetry", "hardware_check.log")
        diag_text = ""
        if os.path.exists(diag_path):
            with open(diag_path, "r") as f:
                diag_text = f.read()
                
        bench_path = os.path.join(base_dir, "telemetry", "inference_benchmark.json")
        bench_data = {}
        if os.path.exists(bench_path):
            with open(bench_path, "r") as f:
                bench_data = json.load(f)
                
        return {
            "execution_provider": orchestrator.detector.provider,
            "system_diagnostics": diag_text,
            "benchmark_stats": bench_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_report", response_class=HTMLResponse)
def generate_report(aircraft_id: str = Form("AERO-DEV-01")):
    try:
        records = orchestrator.logger.get_history(limit=100)
        # Filter records for target aircraft
        aircraft_records = [r for r in records if r["aircraft_id"] == aircraft_id]
        
        # Urgency summary
        counts = {"IMMEDIATE_GROUND": 0, "SCHEDULED_REPAIR": 0, "MONITOR": 0}
        rows_html = ""
        for idx, rec in enumerate(aircraft_records):
            if rec["defect_class"] == "no_defects":
                badge_color = "#6b7280"
                urg_band = "CLEAR"
                conf_val = 0.0
                sev_val = 0.0
            else:
                counts[rec["urgency_band"]] += 1
                badge_color = "red" if rec["urgency_band"] == "IMMEDIATE_GROUND" else "orange" if rec["urgency_band"] == "SCHEDULED_REPAIR" else "green"
                urg_band = rec["urgency_band"]
                conf_val = rec["confidence"]
                sev_val = rec["severity_score"]
                
            img_html = ""
            if rec.get("image_path"):
                img_html = f'<img src="{rec["image_path"]}" style="max-height: 90px; max-width: 130px; object-fit: contain; border-radius: 4px; border: 1px solid #cbd5e0;" />'
            else:
                img_html = "N/A"
                
            rows_html += f"""
            <tr>
                <td>{idx+1}</td>
                <td>{rec["timestamp"]}</td>
                <td><strong>{rec["zone"].replace('_', ' ').title()}</strong></td>
                <td>{rec["defect_class"].replace('_', ' ').upper()}</td>
                <td>{conf_val:.2f}</td>
                <td>{sev_val:.3f}</td>
                <td><span style="color: white; background-color: {badge_color}; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">{urg_band}</span></td>
                <td>{img_html}</td>
            </tr>
            """
        anomalies_html = f"<p>No defect records found for Aircraft ID {aircraft_id}.</p>"
        if rows_html:
            anomalies_html = f"""
            <table class="records-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Timestamp</th>
                        <th>Zone Location</th>
                        <th>Defect Class</th>
                        <th>Confidence</th>
                        <th>Severity Score</th>
                        <th>Urgency Band</th>
                        <th>Annotated Image</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """
            
        summary_html = f"""
        <html>
        <head>
            <title>AeroLens Sentinel - Inspection Summary Report</title>
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
                .meta-table {{ width: 100%; margin-bottom: 30px; border-collapse: collapse; }}
                .meta-table td {{ padding: 8px; border: 0; }}
                .summary-card {{ display: inline-block; padding: 15px 25px; margin-right: 20px; border-radius: 8px; color: white; font-weight: bold; }}
                .card-red {{ background-color: #e53e3e; }}
                .card-orange {{ background-color: #dd6b20; }}
                .card-green {{ background-color: #38a169; }}
                table.records-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                table.records-table th, table.records-table td {{ border: 1px solid #cbd5e0; padding: 12px; text-align: left; vertical-align: middle; }}
                table.records-table th {{ background-color: #f7fafc; }}
            </style>
        </head>
        <body>
            <h1>AeroLens Sentinel — Defect Inspection Summary Report</h1>
            <table class="meta-table">
                <tr>
                    <td><strong>Aircraft ID:</strong> {aircraft_id}</td>
                    <td><strong>Report Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</td>
                </tr>
                <tr>
                    <td><strong>Technician ID:</strong> Tech-MRO-Admin</td>
                    <td><strong>Connectivity State:</strong> Hangar Local (Offline Cache Mode)</td>
                </tr>
            </table>
 
            <h2>Urgency Priority Band Summary</h2>
            <div class="summary-card card-red">Immediate Ground: {counts['IMMEDIATE_GROUND']}</div>
            <div class="summary-card card-orange">Scheduled Repair: {counts['SCHEDULED_REPAIR']}</div>
            <div class="summary-card card-green">Monitor Only: {counts['MONITOR']}</div>
 
            <h2>Detected Surface Anomalies</h2>
            {anomalies_html}
            
            <p style="margin-top: 50px; font-size: 11px; color: #718096; border-top: 1px solid #e2e8f0; padding-top: 10px;">
                CONFIDENTIAL — Aerospace MRO Proprietary Record. Generated fully on-device via AeroLens Sentinel Edge Triangulation System.
            </p>
            
            <script type="text/javascript">
                window.onload = function() {{
                    setTimeout(function() {{
                        window.print();
                    }}, 500);
                }}
            </script>
        </body>
        </html>
        """
        return summary_html
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static frontend index.html on root '/'
@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("Technician Dashboard UI index.html not found.", status_code=404)

# Mount static files directory at last to avoid masking routes
app.mount("/", StaticFiles(directory=static_dir), name="static")
