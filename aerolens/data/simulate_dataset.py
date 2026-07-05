import os
import random
import numpy as np
import cv2

CLASSES = ["crack", "corrosion", "dent", "paint_peel"]

def generate_noise_texture(img, intensity=10):
    noise = np.random.normal(0, intensity, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def draw_rivet(img, x, y):
    cv2.circle(img, (x, y), 8, (180, 180, 180), -1)
    cv2.circle(img, (x, y), 8, (100, 100, 100), 1)
    cv2.circle(img, (x - 2, y - 2), 2, (240, 240, 240), -1)

def draw_defect(img, defect_type, x, y, w, h):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    if defect_type == "crack":
        # Draw thin jagged lines
        points = []
        cx, cy = x + w // 2, y + h // 2
        curr_x, curr_y = cx - w // 2, cy - h // 2
        points.append((curr_x, curr_y))
        steps = 5
        dx = w // steps
        for i in range(steps):
            curr_x += dx
            curr_y += random.randint(-15, 15)
            points.append((curr_x, curr_y))
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], (40, 40, 40), 2)
            cv2.line(mask, points[i], points[i+1], 255, 2)
            
    elif defect_type == "corrosion":
        # Draw brownish/orange irregular texture patch
        pts = []
        cx, cy = x + w // 2, y + h // 2
        num_pts = 8
        for i in range(num_pts):
            angle = i * (2 * np.pi / num_pts)
            rx = w // 2 + random.randint(-w // 6, w // 6)
            ry = h // 2 + random.randint(-h // 6, h // 6)
            px = int(cx + rx * np.cos(angle))
            py = int(cy + ry * np.sin(angle))
            pts.append([px, py])
        pts = np.array(pts, np.int32)
        
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (40, 80, 140)) # BGR for brown-ish/orange (rust)
        cv2.fillPoly(mask, [pts], 255)
        # Add some texture to corrosion
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
    elif defect_type == "dent":
        # Draw shaded ellipses for deformation look
        cx, cy = x + w // 2, y + h // 2
        cv2.ellipse(img, (cx, cy), (w // 2, h // 2), 0, 0, 360, (110, 110, 110), -1)
        # Highlight shadow
        cv2.ellipse(img, (cx - 2, cy - 2), (w // 2 - 3, h // 2 - 3), 0, 0, 360, (210, 210, 210), 2)
        cv2.ellipse(img, (cx + 2, cy + 2), (w // 2 - 3, h // 2 - 3), 0, 0, 360, (80, 80, 80), 2)
        cv2.ellipse(mask, (cx, cy), (w // 2, h // 2), 0, 0, 360, 255, -1)
        
    elif defect_type == "paint_peel":
        # Draw irregular white/light grey peeled metal region
        pts = []
        cx, cy = x + w // 2, y + h // 2
        num_pts = 10
        for i in range(num_pts):
            angle = i * (2 * np.pi / num_pts)
            rx = w // 2 + random.randint(-w // 5, w // 5)
            ry = h // 2 + random.randint(-h // 5, h // 5)
            px = int(cx + rx * np.cos(angle))
            py = int(cy + ry * np.sin(angle))
            pts.append([px, py])
        pts = np.array(pts, np.int32)
        
        cv2.fillPoly(img, [pts], (220, 220, 220))
        cv2.polylines(img, [pts], True, (150, 150, 150), 1)
        cv2.fillPoly(mask, [pts], 255)

    return mask

def simulate_image(filename, label_path, defect_type, img_size=(640, 640)):
    # Generate grey background
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 160
    img = generate_noise_texture(img, 5)
    
    # Draw some rivets for realistic aircraft surface look
    for ry in range(40, img_size[1], 150):
        for rx in range(40, img_size[0], 120):
            draw_rivet(img, rx, ry)
            
    # Random size and location for defect
    w = random.randint(40, 150)
    h = random.randint(40, 150)
    x = random.randint(100, img_size[0] - 100 - w)
    y = random.randint(100, img_size[1] - 100 - h)
    
    draw_defect(img, defect_type, x, y, w, h)
    
    # Write image
    cv2.imwrite(filename, img)
    
    # Write annotation (YOLO format: class_idx x_center y_center width height)
    class_idx = CLASSES.index(defect_type)
    x_center = (x + w / 2) / img_size[0]
    y_center = (y + h / 2) / img_size[1]
    norm_w = w / img_size[0]
    norm_h = h / img_size[1]
    
    with open(label_path, "w") as f:
        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    annotated_dir = os.path.join(base_dir, "data", "annotated")
    
    os.makedirs(annotated_dir, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(raw_dir, c), exist_ok=True)
        
    print("Generating simulated defect images...")
    
    # Generate 20 images per class for training/testing simulation
    for c in CLASSES:
        for i in range(20):
            img_name = f"{c}_{i:03d}.jpg"
            raw_path = os.path.join(raw_dir, c, img_name)
            ann_image_path = os.path.join(annotated_dir, img_name)
            ann_label_path = os.path.join(annotated_dir, f"{c}_{i:03d}.txt")
            
            # Generate the simulated defect file
            simulate_image(ann_image_path, ann_label_path, c)
            
            # Save a copy in raw/ for raw reference
            cv2.imwrite(raw_path, cv2.imread(ann_image_path))
            
    # Also generate a few synthetic samples (synthetic prefix)
    for c in ["dent", "paint_peel"]:
        for i in range(5):
            img_name = f"synthetic_{c}_{i:03d}.jpg"
            ann_image_path = os.path.join(annotated_dir, img_name)
            ann_label_path = os.path.join(annotated_dir, f"synthetic_{c}_{i:03d}.txt")
            simulate_image(ann_image_path, ann_label_path, c)
            
    print("Dataset simulation complete.")

if __name__ == "__main__":
    main()
