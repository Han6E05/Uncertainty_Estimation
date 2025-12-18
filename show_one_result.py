import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def visualize_json_on_image(image_path, json_path):
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Image not found: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load JSON
    with open(json_path, 'r') as f:
        detections = json.load(f)

    for det in detections:
        if Path(det["image_name"]).name != Path(image_path).name:
            continue  # Skip if image name doesn't match

        # Draw bounding box
        x, y, w, h = det["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Inside your loop over detections:
        keypoints = np.array(det["keypoints"]).reshape(-1, 3)  # [N, 3]
        for x, y, v in keypoints:
            if v >= 0:  # Only draw if labeled or visible
                cv2.circle(image, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)
    # Show image
    plt.imshow(image)
    plt.axis('off')
    plt.title("Detections")
    plt.show()

# Example usage
visualize_json_on_image("/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/real_test/00001.jpg", "KP_RCNN/output/real_output/mc_pass_00.json")
