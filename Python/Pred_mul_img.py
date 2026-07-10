import glob
import os
from ultralytics import YOLO

# ---- Config ----
MODEL_PATH = "best.pt"
TEST_DIR = "Path to test images"
OUTPUT_DIR = "Path to save predicted images"  
NUM_IMAGES = 428         
BATCH_SIZE = 8           
CONF_THRESH = 0.70       # confidence threshold for predictions

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load custom trained model ----
model = YOLO(MODEL_PATH)

# ---- Collect image paths ----
exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
image_paths = []
for ext in exts:
    image_paths.extend(glob.glob(os.path.join(TEST_DIR, ext)))

image_paths = sorted(image_paths)

if len(image_paths) < NUM_IMAGES:
    print(f"Warning: only found {len(image_paths)} images in {TEST_DIR}, "
          f"using all of them (requested {NUM_IMAGES}).")
else:
    image_paths = image_paths[:NUM_IMAGES]

print(f"Running inference on {len(image_paths)} images...\n")

# ---- Run batched inference ----
all_results = []
for i in range(0, len(image_paths), BATCH_SIZE):
    batch = image_paths[i:i + BATCH_SIZE]
    results = model(batch, conf=CONF_THRESH, verbose=False)
    all_results.extend(results)

# ---- Process and print predictions ----
for img_path, result in zip(image_paths, all_results):
    boxes = result.boxes
    filename = os.path.basename(img_path)

    print(f"Image: {filename}")

    if boxes is None or len(boxes) == 0:
        print("  -> No detections")
    else:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            print(f"  -> Class: {cls_name:20s} Conf: {conf:.3f}  "
                  f"BBox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

    # Save annotated image
    save_path = os.path.join(OUTPUT_DIR, filename)
    result.save(filename=save_path)

    print()

print(f"Done. Annotated images saved to: {OUTPUT_DIR}")
