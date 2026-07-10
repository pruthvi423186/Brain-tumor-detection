import os
import glob
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO, SAM

# ---- Config ----
DETECTION_MODEL_PATH = r"D:\Downloads\runs\detect\runs\train\yolo26n_custom\weights\best.pt"
SEG_MODEL_PATH = "mobile_sam.pt"   # swap to "sam2_t.pt" if you want more precise masks (slower, more VRAM)

TEST_DIR = r"D:\Datasets\Data\test\images"
OUTPUT_DIR = r"D:\Datasets\Data\test\seg_predictions"
NUM_IMAGES = 428
CONF_THRESH = 0.70

MASK_COLOR = (0, 255, 255)   # cyan overlay for the segmented tumor region
MASK_ALPHA = 0.4             # overlay transparency

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load models ----
det_model = YOLO(DETECTION_MODEL_PATH)   # your custom trained tumor detector
seg_model = SAM(SEG_MODEL_PATH)          # pretrained, class-agnostic segmenter

# ---- Collect test images ----
exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
image_paths = []
for ext in exts:
    image_paths.extend(glob.glob(os.path.join(TEST_DIR, ext)))
image_paths = sorted(image_paths)

if len(image_paths) < NUM_IMAGES:
    print(f"Warning: only found {len(image_paths)} images, using all of them "
          f"(requested {NUM_IMAGES}).")
else:
    image_paths = image_paths[:NUM_IMAGES]

print(f"Running detection + segmentation on {len(image_paths)} images...\n")

# ---- Analytics accumulators ----
class_counts = defaultdict(int)          # total detections per class (for pie/bar)
running_counts = defaultdict(list)       # cumulative count per class over image index (for area chart)
cumulative = defaultdict(int)
all_class_names = set()

# ---- Process each image ----
for idx, img_path in enumerate(image_paths, start=1):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {filename}, skipping.")
        continue

    overlay = img.copy()

    # Step 1: detect + classify tumor with your custom model
    det_result = det_model(img_path, conf=CONF_THRESH, verbose=False)[0]
    boxes = det_result.boxes

    print(f"Image: {filename}")

    if boxes is None or len(boxes) == 0:
        print("  -> No tumor detected")
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)
        for cls_name in all_class_names:
            running_counts[cls_name].append(cumulative[cls_name])
        print()
        continue

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = det_model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        # Step 2: segment the exact tumor shape inside that box using SAM
        seg_result = seg_model(img_path, bboxes=[xyxy], verbose=False)[0]

        if seg_result.masks is None or len(seg_result.masks.data) == 0:
            print(f"  -> {cls_name} ({conf:.2f}) detected, but segmentation failed")
            continue

        mask = seg_result.masks.data[0].cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        pixel_area = int(mask_resized.sum())
        pct_of_image = 100 * pixel_area / (img.shape[0] * img.shape[1])

        ys, xs = np.where(mask_resized > 0)
        centroid = (int(xs.mean()), int(ys.mean())) if len(xs) > 0 else None

        print(f"  -> Class: {cls_name:20s} Conf: {conf:.3f}  "
              f"BBox: [{xyxy[0]:.1f},{xyxy[1]:.1f},{xyxy[2]:.1f},{xyxy[3]:.1f}]  "
              f"MaskArea: {pixel_area}px ({pct_of_image:.2f}% of image)  "
              f"Centroid: {centroid}")

        # Track for analytics
        class_counts[cls_name] += 1
        cumulative[cls_name] += 1
        all_class_names.add(cls_name)

        # Draw mask overlay
        colored_mask = np.zeros_like(img)
        colored_mask[mask_resized > 0] = MASK_COLOR
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, MASK_ALPHA, 0)

        # Draw box + label
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), MASK_COLOR, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(overlay, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MASK_COLOR, 1, cv2.LINE_AA)

    # record cumulative counts per class after this image (for area chart)
    for cls_name in all_class_names:
        running_counts[cls_name].append(cumulative[cls_name])

    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, overlay)
    print()

print(f"Done processing. Segmented + annotated images saved to: {OUTPUT_DIR}\n")

# =========================================================
#                     ANALYTICS CHARTS
# =========================================================
if class_counts:
    labels = list(class_counts.keys())
    counts = list(class_counts.values())

    # ---- Pie chart: proportion of each tumor class ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Tumor Class Distribution (Pie)")
    pie_path = os.path.join(OUTPUT_DIR, "analytics_pie.png")
    fig.savefig(pie_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Bar chart: total detections per class ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, counts, color="#042AFF")
    ax.set_xlabel("Tumor Class")
    ax.set_ylabel("Total Detections")
    ax.set_title("Tumor Detections per Class (Bar)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    bar_path = os.path.join(OUTPUT_DIR, "analytics_bar.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Area chart: cumulative detections per class over image index ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, len(image_paths) + 1)
    for cls_name, series in running_counts.items():
        # pad series to match total number of processed images
        padded = series + [series[-1]] * (len(x) - len(series)) if series else [0] * len(x)
        ax.plot(x, padded, label=cls_name)
        ax.fill_between(x, padded, alpha=0.3)
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Cumulative Detections")
    ax.set_title("Cumulative Tumor Detections Across Test Set (Area)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    area_path = os.path.join(OUTPUT_DIR, "analytics_area.png")
    fig.savefig(area_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Analytics charts saved:")
    print(f"  Pie:  {pie_path}")
    print(f"  Bar:  {bar_path}")
    print(f"  Area: {area_path}")
else:
    print("No detections across the test set — skipping analytics charts.")