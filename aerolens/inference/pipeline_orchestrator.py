import os
import cv2
import uuid
import time
from aerolens.inference.detector import EdgeDetector
from aerolens.severity_engine.severity_scorer import SeverityScorer
from aerolens.telemetry.logger import TelemetryLogger

class PipelineOrchestrator:
    def __init__(self, config_path=None, zone_map_path=None, db_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.detector = EdgeDetector(config_path)
        self.scorer = SeverityScorer(config_path, zone_map_path)
        self.logger = TelemetryLogger(db_path)
        self.output_dir = os.path.join(base_dir, "data", "annotated_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_frame(self, frame, aircraft_id="AERO-DEV-01", zone="wing_spar"):
        """
        Runs full pipeline: detection -> severity calculation -> logging -> frame annotation
        """
        raw_detections = self.detector.detect(frame)
        results = []
        annotated_frame = frame.copy()
        
        # Urgency color mapping: BGR format
        color_map = {
            "IMMEDIATE_GROUND": (0, 0, 255),      # Red
            "SCHEDULED_REPAIR": (0, 140, 255),    # Orange/Amber
            "MONITOR": (0, 200, 0)                # Green
        }
        
        for det in raw_detections:
            cls = det["class"]
            conf = det["confidence"]
            box = det["box"]
            
            # Severity evaluation
            sev = self.scorer.calculate_severity(cls, conf, box, frame, zone)
            
            # Record result item
            res_item = {
                "class": cls,
                "confidence": conf,
                "box": box,
                "severity_score": sev["score"],
                "urgency_band": sev["urgency_band"],
                "details": sev
            }
            results.append(res_item)
            
            # Draw color-coded bounding box
            color = color_map.get(sev["urgency_band"], (255, 255, 255))
            x, y, w, h = box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
            
            # Label overlay
            lbl = f"{cls.upper()}: {conf:.2f} (Sev: {sev['score']:.2f})"
            urg_lbl = f"[{sev['urgency_band']}]"
            cv2.putText(annotated_frame, lbl, (x, max(15, y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(annotated_frame, urg_lbl, (x, max(30, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        annotated_path = ""
        # Save image and log all scans
        if results:
            img_filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
            annotated_path = os.path.join(self.output_dir, img_filename)
            cv2.imwrite(annotated_path, annotated_frame)
            
            # Log all detections into SQLite database
            for res in results:
                relative_img_path = f"/data/annotated_output/{img_filename}"
                self.logger.log_detection(
                    aircraft_id=aircraft_id,
                    zone=zone,
                    defect_class=res["class"],
                    confidence=res["confidence"],
                    severity_score=res["severity_score"],
                    urgency_band=res["urgency_band"],
                    image_path=relative_img_path
                )
        else:
            # If no defects, save clean image and log as no_defects
            img_filename = f"clear_{uuid.uuid4().hex[:8]}.jpg"
            annotated_path = os.path.join(self.output_dir, img_filename)
            cv2.imwrite(annotated_path, frame)
            
            relative_img_path = f"/data/annotated_output/{img_filename}"
            self.logger.log_detection(
                aircraft_id=aircraft_id,
                zone=zone,
                defect_class="no_defects",
                confidence=0.0,
                severity_score=0.0,
                urgency_band="MONITOR",
                image_path=relative_img_path
            )
                
        return {
            "aircraft_id": aircraft_id,
            "zone": zone,
            "detections": results,
            "annotated_image_path": annotated_path.replace("\\", "/") if annotated_path else ""
        }
