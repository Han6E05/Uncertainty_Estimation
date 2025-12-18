import json
import cv2
import os

# paths
ann_file = "KP_RCNN/dataset/annotations/tomato_keypoints_val1.json"
results_file = "KP_RCNN/output/inference/formatted_keypoints.json"
images_dir = "KP_RCNN/dataset/images/val"
save_dir = "output/val"
os.makedirs(save_dir, exist_ok=True)

# load annotations (id -> filename mapping)
with open(ann_file, "r") as f:
    ann = json.load(f)
id_to_name = {img["id"]: img["file_name"] for img in ann["images"]}

# load predictions
with open(results_file, "r") as f:
    preds = json.load(f)

# loop over predictions
for pred in preds:
    image_id = pred["image_id"]
    if image_id not in id_to_name:
        continue

    img_name = id_to_name[image_id]
    img_path = os.path.join(images_dir, img_name)
    if not os.path.exists(img_path):
        print("Missing image:", img_path)
        continue

    img = cv2.imread(img_path)
    keypoints = pred["keypoints"]
    score = pred.get("score", 0.0)  # overall confidence of the person

    # each keypoint = (x, y, confidence)
    kpts = [(keypoints[i], keypoints[i+1], keypoints[i+2])
            for i in range(0, len(keypoints), 3)]

    # draw keypoints with confidence
    for idx, (x, y, conf) in enumerate(kpts):
        if conf > 0.3:  # only draw confident points
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(img, f"{conf:.2f}", (int(x)+2, int(y)-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    # draw bounding box around keypoints
    x_coords = [x for (x, y, conf) in kpts if conf > 0.3]
    y_coords = [y for (x, y, conf) in kpts if conf > 0.3]
    if x_coords and y_coords:
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        # put overall score
        cv2.putText(img, f"score: {score:.2f}", (xmin, ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # save result
    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, img)

print("Visualization with keypoints, bounding box, and confidence saved to:", save_dir)
