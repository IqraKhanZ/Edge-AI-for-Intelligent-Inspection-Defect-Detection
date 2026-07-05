import os
import json

def simulate_training():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    telemetry_dir = os.path.join(base_dir, "telemetry")
    models_dir = os.path.join(base_dir, "models")
    
    os.makedirs(telemetry_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "optimized", "tensorrt"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "optimized", "openvino"), exist_ok=True)
    
    # 4 classes metrics (crack, corrosion, dent, paint_peel)
    metrics = {
        "mAP_0.5": 0.812,
        "mAP_0.5_0.95": 0.548,
        "epochs": 100,
        "batch_size": 16,
        "class_metrics": [
            {"class": "crack", "precision": 0.85, "recall": 0.79, "map_0.5": 0.82},
            {"class": "corrosion", "precision": 0.78, "recall": 0.81, "map_0.5": 0.80},
            {"class": "dent", "precision": 0.82, "recall": 0.76, "map_0.5": 0.79},
            {"class": "paint_peel", "precision": 0.76, "recall": 0.83, "map_0.5": 0.83}
        ]
    }
    
    # Write metrics to telemetry
    metrics_path = os.path.join(telemetry_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Write placeholder files to pretend we have ONNX and OpenVINO models
    # Since we are using an emulation layer, we'll write lightweight files
    # containing model signatures or empty bytes.
    onnx_path = os.path.join(models_dir, "optimized", "tensorrt", "model.onnx")
    with open(onnx_path, "w") as f:
        f.write("MOCK_ONNX_MODEL_TENSORRT_FP16")
        
    ov_xml_path = os.path.join(models_dir, "optimized", "openvino", "model.xml")
    with open(ov_xml_path, "w") as f:
        f.write('<net name="aerolens_yolov8n" version="11"></net>')
        
    ov_bin_path = os.path.join(models_dir, "optimized", "openvino", "model.bin")
    with open(ov_bin_path, "wb") as f:
        f.write(b"\x00" * 1024) # 1 KB of empty bytes
        
    print(f"Simulated training metrics saved to: {metrics_path}")
    print("Mock model paths created successfully.")

if __name__ == "__main__":
    simulate_training()
