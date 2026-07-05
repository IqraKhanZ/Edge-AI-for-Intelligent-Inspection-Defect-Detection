import os
import cv2
import numpy as np
import yaml
import onnxruntime as ort

class EdgeDetector:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.conf_threshold = self.config["model"]["confidence_threshold"]
        self.target_res = tuple(self.config["model"]["input_resolution"])
        self.classes = ["crack", "corrosion", "dent", "paint_peel"]
        self.provider = "YOLOv8 ONNX Engine"
        
        # Load trained ONNX model
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "optimized", "tensorrt", "model.onnx")
        
        if not os.path.exists(self.model_path):
            print(f"[Detector] ONNX model not found at {self.model_path}.")
            self.session = None
        else:
            print(f"[Detector] Loading ONNX model session: {self.model_path}")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        """
        Runs both the YOLOv8 model and OpenCV heuristics, pooling and merging all detections
        via Non-Maximum Suppression (NMS) to maximize recall and eliminate false negatives.
        """
        all_detections = []
        
        # 1. Gather Deep Learning (YOLOv8 ONNX) predictions
        if self.session is not None:
            try:
                yolo_thresh = max(self.conf_threshold, 0.50)
                # Preprocess frame to (1, 3, 640, 640) normalized float32
                h_orig, w_orig = frame.shape[:2]
                input_img = cv2.resize(frame, (640, 640))
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
                input_tensor = np.expand_dims(input_img, axis=0)
                
                # Run ONNX inference
                outputs = self.session.run(None, {self.input_name: input_tensor})
                output = outputs[0][0] # (8, 8400)
                
                # Parse detections: shape (8, 8400) -> xc, yc, w, h, class_0, class_1, class_2, class_3
                boxes = output[:4, :].T # (8400, 4)
                scores = output[4:, :].T # (8400, 4)
                
                class_ids = np.argmax(scores, axis=1)
                confidences = np.max(scores, axis=1)
                
                mask = confidences >= yolo_thresh
                boxes = boxes[mask]
                class_ids = class_ids[mask]
                confidences = confidences[mask]
                
                for i in range(len(boxes)):
                    xc, yc, w, h = boxes[i]
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    
                    rx = int(x1 * w_orig / 640)
                    ry = int(y1 * h_orig / 640)
                    rw = int(w * w_orig / 640)
                    rh = int(h * h_orig / 640)
                    
                    all_detections.append({
                        "box": [rx, ry, rw, rh],
                        "class": self.classes[class_ids[i]],
                        "confidence": float(round(confidences[i], 2))
                    })
            except Exception as e:
                print(f"[Detector] YOLO ONNX inference failed: {e}")
            
        # 2. Gather OpenCV Heuristic predictions
        try:
            heuristics_detections = self._detect_heuristics(frame)
            all_detections.extend(heuristics_detections)
        except Exception as e:
            print(f"[Detector] OpenCV heuristics failed: {e}")
            
        # 3. Merge predictions using Non-Maximum Suppression (NMS)
        return self._nms(all_detections)

    def _detect_heuristics(self, frame):
        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, self.target_res)
        detections = []
        
        smoothed = cv2.bilateralFilter(resized, 9, 75, 75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        
        # 1. Crack Detection: Focus on dark, thin, non-circular line contours
        edges_crack = cv2.Canny(gray, 50, 180)
        contours_crack, _ = cv2.findContours(edges_crack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_crack:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 15 and h > 15:
                perimeter = cv2.arcLength(cnt, False)
                area = cv2.contourArea(cnt)
                if perimeter > 100:
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
                    if circularity < 0.2:
                        conf = min(0.65 + (perimeter / 2000.0), 0.98)
                        if conf >= self.conf_threshold:
                            rx = int(x * w_orig / self.target_res[0])
                            ry = int(y * h_orig / self.target_res[1])
                            rw = int(w * w_orig / self.target_res[0])
                            rh = int(h * h_orig / self.target_res[1])
                            detections.append({
                                "box": [rx, ry, rw, rh],
                                "class": "crack",
                                "confidence": float(round(conf, 2))
                            })
                            
        # 2. Corrosion Detection: Search for rust-colored (brown/orange) clusters
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        lower_brown = np.array([8, 60, 45])
        upper_brown = np.array([22, 230, 180])
        mask_rust = cv2.inRange(hsv, lower_brown, upper_brown)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_rust = cv2.morphologyEx(mask_rust, cv2.MORPH_OPEN, kernel)
        
        contours_rust, _ = cv2.findContours(mask_rust, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_rust:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area > 400 and w > 20 and h > 20:
                conf = min(0.55 + (area / 12000.0), 0.95)
                if conf >= self.conf_threshold:
                    rx = int(x * w_orig / self.target_res[0])
                    ry = int(y * h_orig / self.target_res[1])
                    rw = int(w * w_orig / self.target_res[0])
                    rh = int(h * h_orig / self.target_res[1])
                    detections.append({
                        "box": [rx, ry, rw, rh],
                        "class": "corrosion",
                        "confidence": float(round(conf, 2))
                    })
                    
        # 3. Paint Peel Detection: Adaptive contrast thresholding to find boundary transitions
        gray_peel = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        thresh_peel = cv2.adaptiveThreshold(
            gray_peel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
        )
        kernel_peel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh_peel = cv2.morphologyEx(thresh_peel, cv2.MORPH_CLOSE, kernel_peel)
        thresh_peel = cv2.morphologyEx(thresh_peel, cv2.MORPH_OPEN, kernel_peel)
        
        contours_peel, _ = cv2.findContours(thresh_peel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_peel:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area > 150 and w > 10 and h > 10:
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
                if circularity < 0.85:
                    conf = min(0.60 + (1.0 - circularity) * 0.35, 0.98)
                    if conf >= self.conf_threshold:
                        rx = int(x * w_orig / self.target_res[0])
                        ry = int(y * h_orig / self.target_res[1])
                        rw = int(w * w_orig / self.target_res[0])
                        rh = int(h * h_orig / self.target_res[1])
                        detections.append({
                            "box": [rx, ry, rw, rh],
                            "class": "paint_peel",
                            "confidence": float(round(conf, 2))
                        })
                        
        # 4. Dent Detection: Sunken circular/oval structural gradients
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=100,
                                   param1=50, param2=38, minRadius=25, maxRadius=80)
        if circles is not None:
            for circ in circles[0]:
                cx, cy, r = circ
                x = int(cx - r)
                y = int(cy - r)
                w = int(2 * r)
                h = int(2 * r)
                crop = gray[max(0, int(cy-r)):min(self.target_res[1], int(cy+r)),
                            max(0, int(cx-r)):min(self.target_res[0], int(cx+r))]
                if crop.size > 0:
                    mean_val = np.mean(crop)
                    std_val = np.std(crop)
                    if 120 < mean_val < 210 and 15 < std_val < 45:
                        conf = float(round(0.48 + (std_val / 100.0), 2))
                        if conf >= self.conf_threshold:
                            rx = int(x * w_orig / self.target_res[0])
                            ry = int(y * h_orig / self.target_res[1])
                            rw = int(w * w_orig / self.target_res[0])
                            rh = int(h * h_orig / self.target_res[1])
                            detections.append({
                                "box": [rx, ry, rw, rh],
                                "class": "dent",
                                "confidence": conf
                            })
                            
        # Apply basic Non-Maximum Suppression to remove duplicate overlapping detections
        return self._nms(detections)

    def _nms(self, detections):
        if not detections:
            return []
            
        # Group by class first
        grouped = {}
        for d in detections:
            grouped.setdefault(d["class"], []).append(d)
            
        keep = []
        iou_thresh = self.config["model"]["iou_threshold"]
        
        for cls, items in grouped.items():
            # Sort by confidence descending
            items.sort(key=lambda x: x["confidence"], reverse=True)
            while items:
                best = items.pop(0)
                keep.append(best)
                remaining = []
                for item in items:
                    if self._compute_iou(best["box"], item["box"]) < iou_thresh:
                        remaining.append(item)
                items = remaining
                
        return keep

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        unionArea = boxAArea + boxBArea - interArea
        if unionArea == 0:
            return 0
        return interArea / float(unionArea)
