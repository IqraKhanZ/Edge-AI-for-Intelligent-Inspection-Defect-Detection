import os
import json
import cv2
import numpy as np
import yaml

class SeverityScorer:
    def __init__(self, config_path=None, zone_map_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if config_path is None:
            config_path = os.path.join(base_dir, "config", "config.yaml")
        if zone_map_path is None:
            zone_map_path = os.path.join(base_dir, "criticality_map", "zone_criticality.json")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        with open(zone_map_path, "r") as f:
            self.zone_criticality = json.load(f)
            
        self.class_weights = self.config["severity_weights"]
        self.thresholds = self.config["urgency_thresholds"]

    def compute_geometric_factor(self, crop, box, frame_shape):
        """
        Computes geometric_severity_factor (0 to 1 scale):
        Fuses box-to-frame area ratio and Canny edge density within the defect crop.
        """
        if crop is None or crop.size == 0:
            return 0.1
            
        # 1. Area Ratio
        box_area = box[2] * box[3]
        frame_area = frame_shape[0] * frame_shape[1]
        area_ratio = box_area / frame_area
        # Scale area ratio so that a defect covering 5% of the frame gets a high area factor of 1.0
        area_factor = min(area_ratio / 0.05, 1.0)
        
        # 2. Edge/Texture Irregularity
        if len(crop.shape) == 3:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = crop
            
        # Calculate Canny edge density inside the crop
        edges = cv2.Canny(gray_crop, 50, 150)
        num_edge_pixels = cv2.countNonZero(edges)
        total_pixels = gray_crop.size
        
        edge_density = num_edge_pixels / total_pixels if total_pixels > 0 else 0
        # Normalize edge density: an edge density of 25% or more maps to 1.0
        texture_factor = min(edge_density / 0.25, 1.0)
        
        # Fuse: equal parts size and texture irregularity
        geometric_factor = 0.5 * area_factor + 0.5 * texture_factor
        return float(round(max(0.05, min(geometric_factor, 1.0)), 3))

    def calculate_severity(self, defect_class, confidence, box, frame, zone_name):
        """
        severity_score = (defect_type_weight * confidence_score * geometric_severity_factor) * zone_criticality_multiplier
        """
        # 1. Defect class weight
        type_weight = self.class_weights.get(defect_class, 0.50)
        
        # 2. Crop defect for texture analysis
        x, y, w, h = box
        img_h, img_w = frame.shape[:2]
        # Safeguard box coordinates
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        crop = frame[y1:y2, x1:x2]
        
        # 3. Geometric factor
        geom_factor = self.compute_geometric_factor(crop, box, (img_w, img_h))
        
        # 4. Zone criticality lookup
        zone_mult = self.zone_criticality.get(zone_name, 0.50)
        
        # 5. Formula calculation
        score = (type_weight * confidence * geom_factor) * zone_mult
        score = float(round(score, 3))
        
        # 6. Urgency band determination
        if score >= self.thresholds["immediate_ground"]:
            urgency = "IMMEDIATE_GROUND"
        elif score >= self.thresholds["scheduled_repair"]:
            urgency = "SCHEDULED_REPAIR"
        else:
            urgency = "MONITOR"
            
        return {
            "score": score,
            "type_weight": type_weight,
            "geometric_factor": geom_factor,
            "zone_multiplier": zone_mult,
            "urgency_band": urgency
        }
