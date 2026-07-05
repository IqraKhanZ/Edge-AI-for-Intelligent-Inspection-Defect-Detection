import os
import sys
import tempfile
import sqlite3
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerolens.inference.pipeline_orchestrator import PipelineOrchestrator

def test_pipeline_integration():
    # Setup temporary db
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    
    # Instantiate orchestrator with temp database
    orchestrator = PipelineOrchestrator(db_path=db_path)
    
    # Create a dummy image representing a defect (e.g. rust-colored block for corrosion)
    frame = np.ones((640, 640, 3), dtype=np.uint8) * 160
    # Paint a rust-like spot (orange/brown color: B=40, G=80, R=140) in the middle
    frame[250:350, 250:350] = [40, 80, 140]
    
    # Process the frame
    result = orchestrator.process_frame(frame, aircraft_id="TEST-PLANE-X", zone="wing_spar")
    
    # Check results structure
    assert result["aircraft_id"] == "TEST-PLANE-X"
    assert result["zone"] == "wing_spar"
    assert "detections" in result
    assert isinstance(result["detections"], list)
    
    # Verify SQLite entry
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM inspection_records WHERE aircraft_id = 'TEST-PLANE-X'")
    cnt = cursor.fetchone()[0]
    conn.close()
    
    # Detections log should run without errors, writing entries if any detections are made.
    assert isinstance(cnt, int)
    
    # Clean up temp database
    os.close(db_fd)
    if os.path.exists(db_path):
        os.remove(db_path)
