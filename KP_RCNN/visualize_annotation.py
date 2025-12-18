import os
import json
import cv2
import numpy as np
from pathlib import Path

# ============ CONFIG ============
IMAGES_DIR = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/train"
ANN_PATH = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/annotations/tomato_keypoints_train1.json"
OUT_DIR = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/annotated_train"
os.makedirs(OUT_DIR, exist_ok=True)

# Keypoint names
KEYPOINT_NAMES = [
    "cutting top",
    "cutting bottom",
    "cutting point",
    "peduncle further",
    "peduncle end"
]

# Keypoint colors (distinct colors for each keypoint)
KEYPOINT_COLORS = [
    (255, 0, 0),  # Blue for cutting top
    (0, 255, 0),  # Green for cutting bottom
    (0, 0, 255),  # Red for cutting point
    (255, 255, 0),  # Cyan for peduncle further
    (255, 0, 255)  # Magenta for peduncle end
]

# ============ LOAD COCO JSON ============
with open(ANN_PATH, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
annotations = coco["annotations"]

# ============ GROUP ANNOTATIONS BY IMAGE ============
ann_by_image = {}
for ann in annotations:
    img_id = ann["image_id"]
    if img_id not in ann_by_image:
        ann_by_image[img_id] = []
    ann_by_image[img_id].append(ann)


# ============ DRAW FUNCTIONS ============
def draw_bbox(img, bbox, label, color):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_segmentation(img, seg, color):
    if not seg:
        return
    for poly in seg:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.polylines(img, [pts], True, color, 2)


def draw_keypoints_with_names(img, keypoints):
    """
    Draw keypoints with their names
    COCO keypoints are [x1, y1, v1, x2, y2, v2, ...]
    """
    # Create an overlay for transparent effects
    overlay = img.copy()

    for i in range(0, len(keypoints), 3):
        kp_idx = i // 3
        if kp_idx >= len(KEYPOINT_NAMES):
            break

        x, y, v = keypoints[i:i + 3]

        if v > 0:  # visible or occluded
            x, y = int(x), int(y)
            color = KEYPOINT_COLORS[kp_idx]
            kp_name = KEYPOINT_NAMES[kp_idx]

            # Draw the keypoint circle on overlay
            cv2.circle(overlay, (x, y), 5, color, -1)
            cv2.circle(overlay, (x, y), 6, (255, 255, 255), 1)  # White border



    # Blend overlay with original image (0.6 = 60% overlay, 0.4 = 40% original)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Draw text on top (fully opaque for readability)
    for i in range(0, len(keypoints), 3):
        kp_idx = i // 3
        if kp_idx >= len(KEYPOINT_NAMES):
            break

        x, y, v = keypoints[i:i + 3]

        if v > 0:
            x, y = int(x), int(y)
            color = KEYPOINT_COLORS[kp_idx]
            kp_name = KEYPOINT_NAMES[kp_idx]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                kp_name, font, font_scale, thickness
            )

            text_x = x - text_width // 2
            text_y = y - 10

            if text_y < text_height + 5:
                text_y = y + 15
            if text_x < 0:
                text_x = 0
            if text_x + text_width > img.shape[1]:
                text_x = img.shape[1] - text_width

            # Draw text with full opacity
            cv2.putText(
                img,
                kp_name,
                (text_x, text_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )


def draw_keypoint_connections(img, keypoints):
    """
    Optionally draw lines connecting keypoints to show skeleton structure
    Adjust the connections based on your specific keypoint topology
    """
    # Define connections (pairs of keypoint indices)
    # Example: connecting cutting points and peduncle points
    connections = [
        (0, 2),  # cutting top to cutting point
        (1, 2),  # cutting bottom to cutting point
        (2, 3),  # cutting point to peduncle further
        (3, 4),  # peduncle further to peduncle end
    ]

    kp_coords = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i + 3]
        if v > 0:
            kp_coords.append((int(x), int(y), True))
        else:
            kp_coords.append((0, 0, False))

    # Create overlay for transparent lines
    overlay = img.copy()

    # Draw connections on overlay
    for start_idx, end_idx in connections:
        if (start_idx < len(kp_coords) and end_idx < len(kp_coords) and
                kp_coords[start_idx][2] and kp_coords[end_idx][2]):
            start_pt = (kp_coords[start_idx][0], kp_coords[start_idx][1])
            end_pt = (kp_coords[end_idx][0], kp_coords[end_idx][1])
            cv2.line(overlay, start_pt, end_pt, (255, 255, 255), 2)

    # Blend with transparency (0.5 = 50% transparent)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)


# ============ PROCESS EACH IMAGE ============
for img_id, img_info in images.items():
    img_path = os.path.join(IMAGES_DIR, img_info["file_name"])
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Failed to load: {img_path}")
        continue

    anns = ann_by_image.get(img_id, [])
    for ann in anns:
        cat_name = categories.get(ann["category_id"], "unknown")

        # Use a consistent color per annotation instance
        np.random.seed(ann["id"])  # Consistent color per annotation
        bbox_color = tuple(int(x) for x in np.random.randint(50, 255, size=3))

        # Draw bounding box
        if "bbox" in ann:
            draw_bbox(img, ann["bbox"], cat_name, bbox_color)

        # Draw segmentation
        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            draw_segmentation(img, ann["segmentation"], bbox_color)

        # Draw keypoints with names
        if "keypoints" in ann:
            draw_keypoint_connections(img, ann["keypoints"])  # Optional: draw skeleton
            draw_keypoints_with_names(img, ann["keypoints"])

    out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f"✅ Saved annotated image: {out_path}")

print(f"\n✨ All annotated images saved to: {OUT_DIR}")