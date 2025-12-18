"""
KeypointsRandomHorizontalFlip demonstration for TomatoPoseEstimationDataset.

Adjust DATA_DIR/IMAGES_DIR/JSON_FILE and dataset constructor args to your environment.
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Replace this import path with the actual place you defined TomatoPoseEstimationDataset
# from my_project.datasets import TomatoPoseEstimationDataset
# Here we assume the class is already in the Python path.


from super_gradients.training.samples import PoseEstimationSample
# Import the transform (the one you registered earlier)
from super_gradients.training.transforms import KeypointsRandomHorizontalFlip
from train import TomatoPoseEstimationDataset


# ------------------------
# Helper: auto compute flip_index from keypoint names
# ------------------------
def compute_flip_index_from_names(names: list) -> list:
    """
    Try to find left<->right pairs in the keypoint name list. Returns an index mapping
    flip_index such that kps_flipped = kps[:, flip_index].
    If no pair is found for a name, it maps to itself.
    """
    n = len(names)
    idx_map = list(range(n))

    # helper that tries common left/right substrings
    def mirror(name):
        variants = []
        # common patterns
        variants.append(name.replace('left', 'right'))
        variants.append(name.replace('Left', 'Right'))
        variants.append(name.replace('LEFT', 'RIGHT'))
        variants.append(name.replace('l_', 'r_'))
        variants.append(name.replace('r_', 'l_'))
        variants.append(name.replace('_l', '_r'))
        variants.append(name.replace('_r', '_l'))
        # single-letter prefixes/suffixes
        variants.append(name.replace(' L', ' R'))
        variants.append(name.replace(' R', ' L'))
        return variants

    name_to_index = {n: i for i, n in enumerate(names)}
    used = set()
    for i, name in enumerate(names):
        if i in used:
            continue
        # try to find mirrored name
        found = False
        for cand in mirror(name):
            j = name_to_index.get(cand)
            if j is not None:
                idx_map[i] = j
                idx_map[j] = i
                used.add(i)
                used.add(j)
                found = True
                break
        if not found:
            idx_map[i] = i  # map to itself (no pair found)
            used.add(i)

    return idx_map


# ------------------------
# Plot helper
# ------------------------
def plot_sample(image_bgr, keypoints=None, bboxes=None, title=""):
    image = image_bgr[..., ::-1]  # BGR -> RGB
    plt.figure(figsize=(8, 8))
    plt.imshow(image.astype(np.uint8))
    ax = plt.gca()

    if keypoints is not None:
        # Accept [N, K, 3] or [K,3]
        if keypoints.ndim == 3:
            kps = keypoints[0]
        else:
            kps = keypoints
        xs = kps[:, 0]
        ys = kps[:, 1]
        vs = kps[:, 2] if kps.shape[1] > 2 else np.ones(len(xs))
        ax.scatter(xs[vs > 0], ys[vs > 0], s=40)
        for i, (x, y, v) in enumerate(kps):
            if v > 0:
                ax.text(x, y, str(i), color='yellow', fontsize=8)

    if bboxes is not None:
        # bboxes expected as [N,4] XYWH
        for x, y, w, h in np.asarray(bboxes):
            rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
            ax.add_patch(rect)

    plt.title(title)
    plt.axis("off")


# ------------------------
# Config — EDIT THESE
# ------------------------
DATA_DIR = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset"         # folder containing images_dir and json file
IMAGES_DIR = "images/test"                  # relative images folder inside DATA_DIR
JSON_FILE = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/annotations/tomato_keypoints_test1.json"  # the JSON produced by split_tomato_pose_dataset
SAMPLE_INDEX = 0                       # which sample to visualize
MANUAL_FLIP_INDEX = None               # set to list if you already know mapping, else None

# ------------------------
# Create dataset instance (edit args if your constructor differs)
# ------------------------
dataset = TomatoPoseEstimationDataset(
    data_dir=DATA_DIR,
    images_dir=IMAGES_DIR,
    json_file=JSON_FILE,
    transforms=[],            # no transforms for loading raw sample
    edge_links=[],            # if required; dataset constructor accepts this
    edge_colors=None,
    keypoint_colors=None,
)

# ------------------------
# Get keypoint names (from the JSON) and compute flip_index
# ------------------------
# This relies on how your TomatoPoseEstimationDataset reads the json.
# In your class, you extracted "joints = json_annotations['categories'][0]['keypoints']".
# If dataset doesn't expose names, load JSON directly:
with open(os.path.join(DATA_DIR, JSON_FILE), "r") as f:
    json_ann = json.load(f)
keypoint_names = json_ann["categories"][0]["keypoints"]
print("Keypoint names:", keypoint_names)

if MANUAL_FLIP_INDEX is None:
    flip_index = compute_flip_index_from_names(keypoint_names)
else:
    flip_index = MANUAL_FLIP_INDEX

print("Computed flip_index:", flip_index)

# ------------------------
# Load sample and inspect
# ------------------------
sample = dataset.load_sample(SAMPLE_INDEX)
img = sample.image  # BGR (cv2)
kps = sample.joints  # shape: [N, K, 3]
bboxes = sample.bboxes_xywh  # shape: [N, 4] or None

print("Image shape:", img.shape)
print("Keypoints shape:", kps.shape)
print("BBoxes shape:", None if bboxes is None else bboxes.shape)

# Detect whether keypoints look normalized (max <= 1.0) or pixel coords
kps_max = kps[..., :2].max()
norm_flag = kps_max <= 1.01
print("Keypoints appear normalized (0..1)?", norm_flag)

# ------------------------
# Create transform and force flip (prob=1.0) for debug
# ------------------------
transform = KeypointsRandomHorizontalFlip(flip_index=[1,0,2,3,4], prob=1.0)
flipped_sample = transform.apply_to_sample(PoseEstimationSample(
    image=img.copy(),
    mask=sample.mask,
    joints=kps.copy(),
    areas=sample.areas,
    bboxes_xywh=bboxes.copy() if bboxes is not None else None,
    is_crowd=sample.is_crowd,
    additional_samples=None
))

# ------------------------
# Verify coordinate arithmetic (pixel vs normalized)
# ------------------------
cols = img.shape[1]
orig_kps = kps.copy()
flipped_kps = flipped_sample.joints.copy()

# For each mapped index i -> j = flip_index[i], check x_before + x_after == cols - 1 (pixel)
all_ok = True
if not norm_flag:
    for i, j in enumerate(flip_index):
        # handle multi-person: just check first person [0]
        x_before = orig_kps[0, i, 0]
        x_after = flipped_kps[0, j, 0]
        if not np.isclose(x_before + x_after, cols - 1, atol=1e-3):
            print(f"Mapping check failed for i={i} -> j={j}: x_before+x_after = {x_before + x_after} (expected {cols-1})")
            all_ok = False
    if all_ok:
        print("Pixel-coordinate flip mapping checks passed.")
else:
    # normalized coordinates should sum to ~1.0
    for i, j in enumerate(flip_index):
        x_before = orig_kps[0, i, 0]
        x_after = flipped_kps[0, j, 0]
        if not np.isclose(x_before + x_after, 1.0, atol=1e-3):
            print(f"Norm mapping failed i={i}->{j}: {x_before + x_after} (expected 1.0)")
            all_ok = False
    if all_ok:
        print("Normalized-coordinate flip mapping checks passed.")

# ------------------------
# Plot original & flipped samples
# ------------------------
plot_sample(img, kps, bboxes, title="Original")
plt.show()

plot_sample(flipped_sample.image, flipped_sample.joints, flipped_sample.bboxes_xywh, title="Flipped")
plt.show()

# ------------------------
# Quick tip for heatmap pipelines:
# If your training converts kps -> heatmaps, ensure you either:
#  - regenerate heatmaps from flipped keypoints after the transform, OR
#  - flip the heatmap arrays with np.fliplr and reorder channels with flip_index.
# ------------------------
