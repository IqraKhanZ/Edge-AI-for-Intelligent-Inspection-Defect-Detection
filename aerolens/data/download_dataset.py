import os
import yaml

def prepare_yolo_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    # Path to train, val, test manifests
    train_manifest = os.path.join(data_dir, "train.txt")
    val_manifest = os.path.join(data_dir, "val.txt")
    test_manifest = os.path.join(data_dir, "test.txt")
    
    # Ensure they exist (if not, we run split_dataset)
    if not os.path.exists(train_manifest):
        print("[Dataset] Manifests not found. Re-running split_dataset...")
        from aerolens.data.split_dataset import split_data
        split_data()
        
    # Create data.yaml required by YOLOv8
    yolo_config = {
        "path": data_dir.replace("\\", "/"),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {
            0: "crack",
            1: "corrosion",
            2: "dent",
            3: "paint_peel"
        }
    }
    
    config_path = os.path.join(data_dir, "data.yaml")
    with open(config_path, "w") as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
        
    print(f"[Dataset] YOLOv8 configuration saved to: {config_path}")
    print("[Dataset] Dataset preparation completed. Zero-connectivity local cache is ready.")

if __name__ == "__main__":
    prepare_yolo_dataset()
