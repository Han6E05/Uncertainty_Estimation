import matplotlib.pyplot as plt
import numpy as np

# =============================
# Example Data
# =============================

# Keypoints: [x, y, confidence]
keypoints1 = [
      179.4890594482422, 22.53984260559082, 0, 178.65939331054688, 39.16816329956055, 0, 172.98985290527344, 27.898242950439453, 0, 157.22579956054688, 30.940433502197266, 0, 140.701171875, 56.52246856689453, 0,
    ]

keypoints2 = [
    217.00936889648438, 27.25156021118164, 0, 216.00625610351562, 47.35586166381836, 0, 209.15158081054688, 33.73007583618164, 0,
    190.09217834472656, 37.408203125, 0, 170.11328125, 68.337890625,0
    ]

# BBoxes converted to [x_min, y_min, x_max, y_max]
bbox1 = [
      175.13,
      0.0,
      94.92+175.13,
      110.63
    ]  # KP-RCNN
bbox2 = [
      178.91,
      3.57,
      91.18+178.91,
      112.73
    ]    # YOLO

# =============================
# Helper Functions
# =============================

def extract_points(keypoints):
    """Extract x, y, confidence from keypoint list"""
    return [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]

def calculate_iou_xyxy(box1, box2):
    """IoU for bboxes in [x_min, y_min, x_max, y_max] format"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

# =============================
# Data Processing
# =============================

points1 = extract_points(keypoints1)
points2 = extract_points(keypoints2)

x1, y1, conf1 = zip(*points1)
x2, y2, conf2 = zip(*points2)

# Per-keypoint differences
distances = [np.hypot(x2[i]-x1[i], y2[i]-y1[i]) for i in range(len(x1))]
x_diffs = [abs(x2[i]-x1[i]) for i in range(len(x1))]
y_diffs = [abs(y2[i]-y1[i]) for i in range(len(y1))]
uncertainty = [(x_diffs[i]**2 + y_diffs[i]**2)/2 for i in range(len(x1))]

# Global stats
mean_x_diff, mean_y_diff = np.mean(x_diffs), np.mean(y_diffs)
global_uncertainty = np.mean(uncertainty)

# BBox metrics
bbox_iou = calculate_iou_xyxy(bbox1, bbox2)
center1 = [(bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2]
center2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]
center_dist = np.hypot(center2[0]-center1[0], center2[1]-center1[1])

# =============================
# Visualization
# =============================

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# --- Plot Set 1 ---
ax1 = fig.add_subplot(gs[0,0])
sc1 = ax1.scatter(x1, y1, c=conf1, cmap="viridis", s=120, edgecolor="k")
ax1.add_patch(plt.Rectangle((bbox1[0], bbox1[1]), bbox1[2]-bbox1[0], bbox1[3]-bbox1[1],
                            edgecolor="red", facecolor="none", lw=2, label="BBox1"))
ax1.plot(*center1, "r*", ms=15)
ax1.set_title("Keypoints Set 1", fontsize=14, fontweight="bold")
ax1.invert_yaxis()
ax1.grid(alpha=0.3)
plt.colorbar(sc1, ax=ax1, label="Confidence")

# --- Plot Set 2 ---
ax2 = fig.add_subplot(gs[0,1])
sc2 = ax2.scatter(x2, y2, c=conf2, cmap="viridis", s=120, edgecolor="k")
ax2.add_patch(plt.Rectangle((bbox2[0], bbox2[1]), bbox2[2]-bbox2[0], bbox2[3]-bbox2[1],
                            edgecolor="blue", facecolor="none", lw=2, label="BBox2"))
ax2.plot(*center2, "b*", ms=15)
ax2.set_title("Keypoints Set 2", fontsize=14, fontweight="bold")
ax2.invert_yaxis()
ax2.grid(alpha=0.3)
plt.colorbar(sc2, ax=ax2, label="Confidence")

# --- Overlay ---
ax3 = fig.add_subplot(gs[0,2])
ax3.scatter(x1, y1, c="red", s=120, marker="o", label="Set 1")
ax3.scatter(x2, y2, c="blue", s=120, marker="^", label="Set 2")
ax3.add_patch(plt.Rectangle((bbox1[0], bbox1[1]), bbox1[2]-bbox1[0], bbox1[3]-bbox1[1],
                            edgecolor="red", facecolor="none", lw=2, ls="--"))
ax3.add_patch(plt.Rectangle((bbox2[0], bbox2[1]), bbox2[2]-bbox2[0], bbox2[3]-bbox2[1],
                            edgecolor="blue", facecolor="none", lw=2, ls="--"))
for i in range(len(x1)):
    ax3.plot([x1[i], x2[i]], [y1[i], y2[i]], "k--", alpha=0.5)
    ax3.text((x1[i]+x2[i])/2, (y1[i]+y2[i])/2, str(i+1),
             fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
ax3.set_title(f"Overlay Comparison (IoU={bbox_iou:.3f})", fontsize=14, fontweight="bold")
ax3.invert_yaxis()
ax3.grid(alpha=0.3)
ax3.legend()

# --- Uncertainty Heatmap ---
ax4 = fig.add_subplot(gs[1,:])
idx = np.arange(1, len(uncertainty)+1)
bars = ax4.bar(idx, uncertainty, color=plt.cm.Reds(np.array(uncertainty)/max(uncertainty)),
               edgecolor="black")
ax4.set_title("Variance-Based Uncertainty per Keypoint", fontsize=14, fontweight="bold")
ax4.set_xlabel("Keypoint Index")
ax4.set_ylabel("Uncertainty (variance)")
ax4.grid(axis="y", alpha=0.3)
for i, bar in enumerate(bars):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
             f"{uncertainty[i]:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# =============================
# Report
# =============================
print("="*70)
print("           🔍 KEYPOINT & BBOX COMPARISON REPORT")
print("="*70)
print("\n📦 BOUNDING BOXES:")
print(f"  Box 1: {bbox1}")
print(f"  Box 2: {bbox2}")
print(f"  IoU: {bbox_iou:.3f}")
print(f"  Center Distance: {center_dist:.2f} px")

print("\n📍 KEYPOINT DIFFERENCES:")
print(f"{'Point':<7} {'Dist':<8} {'X-Diff':<8} {'Y-Diff':<8} {'Variance':<10}")
print("-"*60)
for i in range(len(x1)):
    print(f"{i+1:<7} {distances[i]:<8.2f} {x_diffs[i]:<8.2f} {y_diffs[i]:<8.2f} {uncertainty[i]:<10.2f}")

print("\n📊 GLOBAL STATS:")
print(f"  Mean X diff: {mean_x_diff:.2f}")
print(f"  Mean Y diff: {mean_y_diff:.2f}")
print(f"  Avg uncertainty: {global_uncertainty:.2f}")
print(f"  Max keypoint uncertainty: {max(uncertainty):.2f}")
print("="*70)
