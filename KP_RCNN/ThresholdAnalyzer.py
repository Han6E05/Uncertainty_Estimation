import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp
from pathlib import Path


class ThresholdAnalyzer:
    def __init__(self, annotations_path, uncertainty_dir):
        self.annotations_path = annotations_path
        self.uncertainty_dir = Path(uncertainty_dir)
        self.results = []
        self.oks_sigmas = np.array([0.15, 0.15, 0.15, 0.15])

    def load_annotations(self):
        """Load COCO format annotations"""
        with open(self.annotations_path, 'r') as f:
            self.annotations_data = json.load(f)

        self.image_to_keypoints = {}
        self.image_to_bbox = {}
        self.image_id_to_file = {}

        for image in self.annotations_data['images']:
            self.image_id_to_file[image['id']] = image['file_name']

        for ann in self.annotations_data['annotations']:
            image_id = ann['image_id']
            keypoints = ann['keypoints']
            kps = np.array(keypoints).reshape(-1, 3)
            self.image_to_keypoints[image_id] = kps
            self.image_to_bbox[image_id] = ann['bbox']

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def calculate_oks(self, pred_kps, gt_kps, bbox_area):
        """Calculate Object Keypoint Similarity (OKS)"""
        num_kps = 4
        oks_sum = 0.0
        valid_kps = 0
        s = np.sqrt(bbox_area)

        for i in range(num_kps):
            pred_x, pred_y = pred_kps[i * 2], pred_kps[i * 2 + 1]
            gt_x, gt_y, visibility = gt_kps[i]

            if visibility == 0:
                continue

            d = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            sigma = self.oks_sigmas[i] if i < len(self.oks_sigmas) else 0.15
            kp_oks = np.exp(-d ** 2 / (2 * (s ** 2) * (sigma ** 2)))
            oks_sum += kp_oks
            valid_kps += 1

        return oks_sum / valid_kps if valid_kps > 0 else 0.0

    def analyze_all_files(self):
        """Analyze all uncertainty JSON files"""
        self.load_annotations()
        uncertainty_files = list(self.uncertainty_dir.glob("*.json"))
        print(f"Found {len(uncertainty_files)} uncertainty files")

        for u_file in uncertainty_files:
            try:
                file_stem = u_file.stem
                if 'uncertainty' in file_stem:
                    image_name = file_stem.replace('_uncertainty', '') + '.png'
                else:
                    image_name = file_stem + '.png'

                image_id = None
                for img_id, img_file in self.image_id_to_file.items():
                    if img_file == image_name:
                        image_id = img_id
                        break

                if image_id is None:
                    continue

                with open(u_file, 'r') as f:
                    uncertainty_data = json.load(f)

                pred_keypoints = uncertainty_data['mean_keypoints']
                gt_keypoints = self.image_to_keypoints[image_id]
                pred_bbox = uncertainty_data['mean_bbox']
                gt_bbox = self.image_to_bbox[image_id]
                bbox_area = pred_bbox[2] * pred_bbox[3]

                oks = self.calculate_oks(pred_keypoints, gt_keypoints, bbox_area)

                result = {
                    'image_id': image_id,
                    'file_name': image_name,
                    'global_uncertainty': uncertainty_data['global_uncertainty'],
                    'oks': oks,
                }
                self.results.append(result)

            except Exception as e:
                continue

        print(f"Successfully processed {len(self.results)} files")

    def find_statistical_thresholds(self):
        """Find thresholds using statistical significance tests"""
        if len(self.results) == 0:
            return None

        oks_values = np.array([r['oks'] for r in self.results])
        uncertainty_values = np.array([r['global_uncertainty'] for r in self.results])

        print("=" * 50)
        print("FINDING STATISTICAL THRESHOLDS")
        print("=" * 50)
        print(f"OKS range: [{oks_values.min():.3f}, {oks_values.max():.3f}]")
        print(f"Uncertainty range: [{uncertainty_values.min():.3f}, {uncertainty_values.max():.3f}]")

        def find_optimal_threshold(values, name):
            """Find threshold that gives most statistically significant separation"""
            best_pvalue = 1.0
            best_threshold = np.median(values)
            best_method = "median"

            # Test multiple potential thresholds
            test_thresholds = np.percentile(values, range(20, 81, 5))

            for threshold in test_thresholds:
                group1 = values[values <= threshold]
                group2 = values[values > threshold]

                if len(group1) < 10 or len(group2) < 10:
                    continue

                # Try multiple statistical tests
                try:
                    # Mann-Whitney U test
                    mw_stat, mw_p = mannwhitneyu(group1, group2, alternative='two-sided')

                    # T-test
                    t_stat, t_p = ttest_ind(group1, group2, equal_var=False)

                    # KS-test
                    ks_stat, ks_p = ks_2samp(group1, group2)

                    # Use the most significant p-value
                    current_p = min(mw_p, t_p, ks_p)

                    if current_p < best_pvalue:
                        best_pvalue = current_p
                        best_threshold = threshold
                        best_method = f"p={current_p:.6f}"

                except:
                    continue

            print(f"{name} threshold: {best_threshold:.3f} ({best_method})")
            return best_threshold

        # Find thresholds
        oks_threshold = find_optimal_threshold(oks_values, "OKS")
        uncertainty_threshold = find_optimal_threshold(uncertainty_values, "Uncertainty")

        return {
            'oks_threshold': oks_threshold,
            'uncertainty_threshold': uncertainty_threshold,
            'oks_values': oks_values,
            'uncertainty_values': uncertainty_values
        }

    def plot_distributions_with_thresholds(self, thresholds):
        """Plot distributions of OKS and uncertainty with thresholds"""
        if thresholds is None:
            return

        oks_values = thresholds['oks_values']
        uncertainty_values = thresholds['uncertainty_values']
        oks_threshold = thresholds['oks_threshold']
        uncertainty_threshold = thresholds['uncertainty_threshold']

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: OKS distribution
        n, bins, patches = ax1.hist(oks_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax1.axvline(oks_threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold = {oks_threshold:.3f}')

        # Add distribution curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(oks_values)
        x_vals = np.linspace(oks_values.min(), oks_values.max(), 100)
        ax1.plot(x_vals, kde(x_vals), 'b-', linewidth=2, alpha=0.8)

        ax1.set_xlabel('OKS Score', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Distribution of OKS Scores with Statistical Threshold', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        ax1.text(0.05, 0.95, f'Mean: {oks_values.mean():.3f}\nStd: {oks_values.std():.3f}\nN: {len(oks_values)}',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=10)

        # Plot 2: Uncertainty distribution
        n, bins, patches = ax2.hist(uncertainty_values, bins=20, alpha=0.7, color='lightcoral',
                                    edgecolor='black', density=True)
        ax2.axvline(uncertainty_threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold = {uncertainty_threshold:.3f}')

        # Add distribution curve
        kde = gaussian_kde(uncertainty_values)
        x_vals = np.linspace(uncertainty_values.min(), uncertainty_values.max(), 100)
        ax2.plot(x_vals, kde(x_vals), 'r-', linewidth=2, alpha=0.8)

        ax2.set_xlabel('Uncertainty', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Distribution of Uncertainty with Statistical Threshold', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        ax2.text(0.05, 0.95,
                 f'Mean: {uncertainty_values.mean():.3f}\nStd: {uncertainty_values.std():.3f}\nN: {len(uncertainty_values)}',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=10)

        plt.tight_layout()
        plt.savefig('threshold_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Saved distribution plots to 'threshold_distributions.png'")

    def run_analysis(self):
        """Run complete analysis"""
        print("Starting analysis...")
        self.analyze_all_files()

        print("\n" + "=" * 50)
        print("CALCULATING THRESHOLDS")
        print("=" * 50)

        thresholds = self.find_statistical_thresholds()

        if thresholds:
            print(f"\nFINAL THRESHOLDS:")
            print(f"OKS Threshold: {thresholds['oks_threshold']:.3f}")
            print(f"Uncertainty Threshold: {thresholds['uncertainty_threshold']:.3f}")

            # Create plots
            self.plot_distributions_with_thresholds(thresholds)

            return thresholds
        else:
            print("No thresholds could be calculated")
            return None


# Usage
if __name__ == "__main__":
    analyzer = ThresholdAnalyzer(
        annotations_path="/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/annotations/tomato_keypoints_val1.json",
        uncertainty_dir="output/json"
    )

    thresholds = analyzer.run_analysis()