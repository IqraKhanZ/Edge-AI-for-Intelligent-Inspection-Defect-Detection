import os
import glob

CLASSES = ["crack", "corrosion", "dent", "paint_peel"]

def validate_dataset():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "annotated")
    images = glob.glob(os.path.join(data_dir, "*.jpg"))
    annotations = glob.glob(os.path.join(data_dir, "*.txt"))
    
    img_basenames = {os.path.splitext(os.path.basename(f))[0] for f in images}
    ann_basenames = {os.path.splitext(os.path.basename(f))[0] for f in annotations}
    
    orphaned_imgs = img_basenames - ann_basenames
    orphaned_anns = ann_basenames - img_basenames
    
    class_counts = {c: 0 for c in CLASSES}
    invalid_labels = []
    
    # Process each annotation file
    for ann in annotations:
        base = os.path.splitext(os.path.basename(ann))[0]
        if base in orphaned_anns:
            continue
            
        with open(ann, "r") as f:
            lines = f.readlines()
            
        if not lines:
            invalid_labels.append((ann, "Empty file"))
            continue
            
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                invalid_labels.append((ann, f"Incorrect format (expected 5 fields, got {len(parts)})"))
                continue
                
            try:
                class_idx = int(parts[0])
                if class_idx < 0 or class_idx >= len(CLASSES):
                    invalid_labels.append((ann, f"Invalid class index {class_idx}"))
                else:
                    class_counts[CLASSES[class_idx]] += 1
            except ValueError:
                invalid_labels.append((ann, f"Non-integer class index '{parts[0]}'"))
                
    # Output class distribution summary table
    print("\n=== Annotation Validation Report ===")
    print(f"Total Valid Images: {len(images) - len(orphaned_imgs)}")
    print(f"Total Valid Annotations: {len(annotations) - len(orphaned_anns)}")
    print(f"Orphaned Images (no annotation): {len(orphaned_imgs)}")
    for o_img in orphaned_imgs:
        print(f"  - {o_img}.jpg")
    print(f"Orphaned Annotations (no image): {len(orphaned_anns)}")
    for o_ann in orphaned_anns:
        print(f"  - {o_ann}.txt")
        
    print("\n--- Class Distribution ---")
    print(f"{'Class Name':<15} | {'Count':<6}")
    print("-" * 25)
    for c, count in class_counts.items():
        print(f"{c:<15} | {count:<6}")
    print("-" * 25)
    
    if invalid_labels:
        print("\n[WARNING] Invalid annotations found:")
        for path, reason in invalid_labels:
            print(f"  - {os.path.basename(path)}: {reason}")
    else:
        print("\nAll labels map cleanly to supported schema classes.")

if __name__ == "__main__":
    validate_dataset()
