import uvicorn
import os
import sys

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    os.environ["PYTHONPATH"] = parent_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    
    print("Starting AeroLens Sentinel Technician Dashboard...")
    print("Point your browser to http://127.0.0.1:8000")
    uvicorn.run("dashboard.api:app", host="127.0.0.1", port=8000, reload=True)
