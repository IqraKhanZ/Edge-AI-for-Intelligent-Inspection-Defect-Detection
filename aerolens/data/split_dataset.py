import os
import glob
import random

def split_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "annotated")
    
    images = glob.glob(os.path.join(data_dir, "*.jpg"))
    # Normalize paths to use relative forward-slashes
    img_rel_paths = [os.path.relpath(img, base_dir).replace("\\", "/") for img in images]
    
    # Sort for reproducibility
    img_rel_paths.sort()
    
    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(img_rel_paths)
    
    total = len(img_rel_paths)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    train_set = img_rel_paths[:train_end]
    val_set = img_rel_paths[train_end:val_end]
    test_set = img_rel_paths[val_end:]
    
    # Write manifests
    with open(os.path.join(base_dir, "data", "train.txt"), "w") as f:
        f.write("\n".join(train_set) + "\n")
        
    with open(os.path.join(base_dir, "data", "val.txt"), "w") as f:
        f.write("\n".join(val_set) + "\n")
        
    with open(os.path.join(base_dir, "data", "test.txt"), "w") as f:
        f.write("\n".join(test_set) + "\n")
        
    print(f"Data split completed:")
    print(f"  - Train: {len(train_set)} images")
    print(f"  - Val:   {len(val_set)} images")
    print(f"  - Test:  {len(test_set)} images")

if __name__ == "__main__":
    split_data()
