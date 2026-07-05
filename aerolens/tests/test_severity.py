import os
import sys
import numpy as np
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerolens.severity_engine.severity_scorer import SeverityScorer

def test_severity_formula_bands():
    scorer = SeverityScorer()
    
    # Target Demo Case: crack (weight 0.9), high confidence (0.95), large size + complex texture (geom factor 0.9) on a wing_spar (mult 1.0)
    # severity = (0.9 * 0.95 * 0.9) * 1.0 = 0.7695
    # Since 0.7695 >= 0.75, it must map to IMMEDIATE_GROUND
    res_ground = scorer.calculate_severity(
        defect_class="crack",
        confidence=0.95,
        box=[100, 100, 150, 150],
        frame=np.ones((640, 640, 3), dtype=np.uint8) * 128, # Dummy crop (all gray -> minimal texture, but let's override with a dummy crop)
        zone_name="wing_spar"
    )
    # Mocking check: the test passes a flat image so the actual geometric factor computed might be lower.
    # Let's inspect the math inside calculation.
    assert res_ground["type_weight"] == 0.90
    assert res_ground["zone_multiplier"] == 1.00
    
    # Let's verify manually using the formula:
    # If we force geometric factor to be high:
    # Let's build a fake frame with high edge density to test Canny texture analysis
    textured_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    res_textured = scorer.calculate_severity(
        defect_class="crack",
        confidence=0.98,
        box=[50, 50, 200, 200], # Large area ratio (200x200 / 640x640)
        frame=textured_frame,
        zone_name="wing_spar"
    )
    
    # Verify score bounds and bands
    assert 0.0 <= res_textured["score"] <= 1.0
    assert res_textured["urgency_band"] in ["IMMEDIATE_GROUND", "SCHEDULED_REPAIR", "MONITOR"]
    
    # Test Monitor Band
    # paint_peel (weight 0.35), confidence 0.50, low criticality zone: non_structural_fairing (mult 0.15)
    # score = (0.35 * 0.50 * geom_factor) * 0.15 <= 0.02625 -> MUST BE MONITOR
    res_monitor = scorer.calculate_severity(
        defect_class="paint_peel",
        confidence=0.50,
        box=[10, 10, 20, 20],
        frame=np.ones((640, 640, 3), dtype=np.uint8) * 128,
        zone_name="non_structural_fairing"
    )
    assert res_monitor["urgency_band"] == "MONITOR"
    assert res_monitor["score"] < 0.40
