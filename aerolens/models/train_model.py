import os
import shutil
from ultralytics import YOLO

def run_training():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(base_dir, "data", "data.yaml")
    
    if not os.path.exists(data_yaml):
        print("[Train] Dataset configuration data.yaml not found. Running download_dataset...")
        from aerolens.data.download_dataset import prepare_yolo_dataset
        prepare_yolo_dataset()
        
    print("[Train] Initializing YOLOv8n pretrained detector...")
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")
    
    print("[Train] Starting training run on CPU (30 epochs)...")
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        device="cpu", # Default to CPU for maximum runtime compatibility
        project=os.path.join(base_dir, "models", "runs"),
        name="aerolens_train"
    )
    
    print("[Train] Training complete. Exporting model weights to ONNX format...")
    # Export to ONNX
    onnx_path = model.export(format="onnx", imgsz=640)
    
    # Copy ONNX file to our optimized model path
    dest_dir = os.path.join(base_dir, "models", "optimized", "tensorrt")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "model.onnx")
    
    shutil.copy(onnx_path, dest_path)
    print(f"[Train] Model successfully exported and copied to: {dest_path}")

if __name__ == "__main__":
    run_training()
