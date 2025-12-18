import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# --- Paths ---
json_path = 'KP_RCNN/output/mc_output/mc_pass_00.json'
image_dir = '/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/test'
output_dir = 'output/test'
os.makedirs(output_dir, exist_ok=True)

# --- Visualization Settings ---
circle_radius = 4
circle_color = (0, 255, 0)      # Green for keypoints
text_color = (0, 255, 255)      # Yellow for confidence
box_color = (0, 0, 255)         # Red for bounding box
line_color = (255, 0, 0)        # Blue for skeleton
line_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
visibility_threshold = 0

# Define skeleton connections (adjust for your keypoint layout)
skeleton = [(0, 1), (1, 2), (2, 3), (3, 4)]

# --- Load predictions ---
with open(json_path, 'r') as f:
    predictions = json.load(f)

# --- Visualization Loop ---
for pred in tqdm(predictions, desc="Visualizing keypoints"):
    img_name = pred['image_name']
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Failed to load {img_name}")
        continue

    keypoints = pred['keypoints']
    score = pred.get('score', 0.0)
    points = []
    x_coords, y_coords = [], []

    # Draw keypoints and collect visible coordinates
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        cv2.circle(img, (int(x), int(y)), circle_radius, circle_color, -1)
        cv2.putText(img, f"{v:.2f}", (int(x)+5, int(y)-5), font, font_scale, text_color, 1)
        points.append((int(x), int(y)))
        x_coords.append(x)
        y_coords.append(y)


    # Draw skeleton connections
    for start, end in skeleton:
        if start < len(points) and end < len(points):
            # optionally check confidence threshold:
            cv2.line(img, points[start], points[end], line_color, line_thickness)

    # Draw bounding box around visible keypoints
    if x_coords and y_coords:
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(img, f"score: {score:.2f}", (xmin, ymin - 10), font, font_scale + 0.1, box_color, 2)

    # Save visualized image
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, img)

print(f"\n✅ Visualization complete. Images saved to: {output_dir}")
