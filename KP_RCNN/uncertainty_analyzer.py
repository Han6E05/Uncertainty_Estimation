import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind, ks_2samp
import os
from pathlib import Path


class EnhancedUncertaintyAnalyzer:
    def __init__(self, annotations_path, uncertainty_dir):
        self.annotations_path = annotations_path
        self.uncertainty_dir = Path(uncertainty_dir)
        self.results = []

        # OKS parameters
        self.oks_sigmas = np.array([0.035, 0.035, 0.035, 0.15, 0.15])
        self.oks_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        # Statistical thresholds (will be calculated)
        self.statistical_thresholds = None

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
        """Calculate Object Keypoint Similarity (OKS) between predicted and ground truth keypoints."""
        num_kps = 5
        oks_sum = 0.0
        valid_kps = 0
        per_kp_oks = []

        s = np.sqrt(bbox_area)

        for i in range(num_kps):
            pred_x, pred_y = pred_kps[i * 2], pred_kps[i * 2 + 1]
            gt_x, gt_y, visibility = gt_kps[i]

            if visibility == 0:
                per_kp_oks.append(0.0)
                continue

            d = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            sigma = self.oks_sigmas[i] if i < len(self.oks_sigmas) else 0.15
            kp_oks = np.exp(-d ** 2 / (2 * (s ** 2) * (sigma ** 2)))

            oks_sum += kp_oks
            valid_kps += 1
            per_kp_oks.append(kp_oks)

        oks = oks_sum / valid_kps if valid_kps > 0 else 0.0
        return oks, per_kp_oks

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
                    print(f"Warning: Could not find image ID for {image_name}")
                    continue

                with open(u_file, 'r') as f:
                    uncertainty_data = json.load(f)

                pred_keypoints = uncertainty_data['mean_keypoints']
                gt_keypoints = self.image_to_keypoints[image_id]
                pred_bbox = uncertainty_data['mean_bbox']
                gt_bbox = self.image_to_bbox[image_id]
                bbox_area = pred_bbox[2] * pred_bbox[3]

                iou = self.calculate_iou(pred_bbox, gt_bbox)
                oks, per_kp_oks = self.calculate_oks(pred_keypoints, gt_keypoints, bbox_area)

                kp_uncertainty = uncertainty_data['keypoint_uncertainty']
                bbox_uncertainty = uncertainty_data['bbox_uncertainty']

                result = {
                    'image_id': image_id,
                    'file_name': image_name,
                    'global_uncertainty': uncertainty_data['global_uncertainty'],
                    'mean_score': uncertainty_data['mean_score'],
                    'num_valid_passes': uncertainty_data['num_valid_passes'],
                    'overall_uncertainty': kp_uncertainty['overall_uncertainty'],
                    'epistemic_uncertainty': kp_uncertainty['epistemic_uncertainty'],
                    'aleatoric_uncertainty': kp_uncertainty['aleatoric_uncertainty'],
                    'mean_bbox': uncertainty_data['mean_bbox'],
                    'iou': iou,
                    'bbox_error': 1.0 - iou,
                    'bbox_uncertainty': bbox_uncertainty['overall_uncertainty'],
                    'oks': oks,
                    'oks_error': 1.0 - oks,
                }

                self.results.append(result)

            except Exception as e:
                print(f"Error processing {u_file}: {e}")
                continue

        print(f"Successfully processed {len(self.results)} files")

    def find_statistical_thresholds(self):
        """Find statistically significant thresholds for OKS and uncertainty"""
        if len(self.results) == 0:
            return None

        oks_values = np.array([r['oks'] for r in self.results])
        uncertainty_values = np.array([r['global_uncertainty'] for r in self.results])

        print("=" * 50)
        print("CALCULATING STATISTICAL THRESHOLDS")
        print("=" * 50)
        print(f"OKS range: [{oks_values.min():.3f}, {oks_values.max():.3f}]")
        print(f"Uncertainty range: [{uncertainty_values.min():.3f}, {uncertainty_values.max():.3f}]")

        def find_optimal_threshold(values, name):
            """Find threshold that gives most statistically significant separation"""
            best_pvalue = 1.0
            best_threshold = np.median(values)
            best_method = "median"

            test_thresholds = np.percentile(values, range(20, 81, 5))

            for threshold in test_thresholds:
                group1 = values[values <= threshold]
                group2 = values[values > threshold]

                if len(group1) < 10 or len(group2) < 10:
                    continue

                try:
                    ks_stat, ks_p = ks_2samp(group1, group2)

                    current_p = ks_p

                    if current_p < best_pvalue:
                        best_pvalue = current_p
                        best_threshold = threshold
                        best_method = f"p={current_p:.6f}"

                except:
                    continue

            print(f"{name} threshold: {best_threshold:.3f} ({best_method})")
            return best_threshold

        oks_threshold = find_optimal_threshold(oks_values, "OKS")
        uncertainty_threshold = find_optimal_threshold(uncertainty_values, "Uncertainty")

        self.statistical_thresholds = {
            'oks_threshold': oks_threshold,
            'uncertainty_threshold': uncertainty_threshold,
            'oks_values': oks_values,
            'uncertainty_values': uncertainty_values
        }

        return self.statistical_thresholds

    def classify_calibration_cases(self, use_statistical_thresholds=True):
        """Classify images into calibration categories using statistical thresholds"""
        if len(self.results) == 0:
            return {}

        if use_statistical_thresholds and self.statistical_thresholds is None:
            self.find_statistical_thresholds()

        if use_statistical_thresholds and self.statistical_thresholds:
            oks_threshold = self.statistical_thresholds['oks_threshold']
            uncertainty_threshold = self.statistical_thresholds['uncertainty_threshold']
            threshold_source = "statistical"
        else:
            # Fallback to median
            oks_values = [r['oks'] for r in self.results]
            uncertainty_values = [r['global_uncertainty'] for r in self.results]
            oks_threshold = np.median(oks_values)
            uncertainty_threshold = np.median(uncertainty_values)
            threshold_source = "median"

        dangerous_cases = []
        conservative_cases = []
        well_calibrated_high_error = []
        well_calibrated_low_error = []
        moderate_cases = []

        for result in self.results:
            oks = result['oks']
            uncertainty = result['global_uncertainty']

            if oks < oks_threshold and uncertainty < uncertainty_threshold:
                dangerous_cases.append(result)
            elif oks > oks_threshold and uncertainty > uncertainty_threshold:
                conservative_cases.append(result)
            elif oks < oks_threshold and uncertainty > uncertainty_threshold:
                well_calibrated_high_error.append(result)
            elif oks > oks_threshold and uncertainty < uncertainty_threshold:
                well_calibrated_low_error.append(result)
            else:
                moderate_cases.append(result)

        total_cases = len(self.results)

        return {
            'dangerous_cases': dangerous_cases,
            'conservative_cases': conservative_cases,
            'well_calibrated_high_error': well_calibrated_high_error,
            'well_calibrated_low_error': well_calibrated_low_error,
            'moderate_cases': moderate_cases,
            'thresholds': {
                'oks_threshold': oks_threshold,
                'uncertainty_threshold': uncertainty_threshold,
                'source': threshold_source
            },
            'counts': {
                'dangerous': len(dangerous_cases),
                'conservative': len(conservative_cases),
                'well_calibrated_high_error': len(well_calibrated_high_error),
                'well_calibrated_low_error': len(well_calibrated_low_error),
                'moderate': len(moderate_cases),
                'total': total_cases
            },
            'percentages': {
                'dangerous': len(dangerous_cases) / total_cases * 100,
                'conservative': len(conservative_cases) / total_cases * 100,
                'well_calibrated_high_error': len(well_calibrated_high_error) / total_cases * 100,
                'well_calibrated_low_error': len(well_calibrated_low_error) / total_cases * 100,
                'moderate': len(moderate_cases) / total_cases * 100,
            }
        }

    def calculate_oks_metrics(self):
        """Calculate comprehensive OKS-based evaluation metrics"""
        if len(self.results) == 0:
            return {}

        oks_scores = [result['oks'] for result in self.results]
        oks_errors = [result['oks_error'] for result in self.results]

        oks_metrics = {
            'mean_oks': np.mean(oks_scores),
            'median_oks': np.median(oks_scores),
            'std_oks': np.std(oks_scores),
            'min_oks': np.min(oks_scores),
            'max_oks': np.max(oks_scores),
            'num_detections': len(oks_scores),
            'oks_50': (np.array(oks_scores) >= 0.5).mean() * 100,
            'oks_75': (np.array(oks_scores) >= 0.75).mean() * 100,
            'oks_90': (np.array(oks_scores) >= 0.9).mean() * 100,
            'mean_oks_error': np.mean(oks_errors),
            'median_oks_error': np.median(oks_errors),
        }

        # Calculate mAP across multiple OKS thresholds
        oks_ap_scores = {}
        for threshold in self.oks_thresholds:
            oks_ap_scores[f'oks_{int(threshold * 100)}'] = (np.array(oks_scores) >= threshold).mean() * 100

        oks_metrics.update(oks_ap_scores)
        oks_metrics['mAP'] = np.mean(list(oks_ap_scores.values()))

        return oks_metrics

    def calculate_bbox_metrics(self):
        """Calculate comprehensive bounding box evaluation metrics"""
        if len(self.results) == 0:
            return {}

        ious = [result['iou'] for result in self.results]
        bbox_errors = [result['bbox_error'] for result in self.results]
        bbox_uncertainties = [result['bbox_uncertainty'] for result in self.results]

        bbox_metrics = {
            'mean_iou': np.mean(ious),
            'median_iou': np.median(ious),
            'std_iou': np.std(ious),
            'min_iou': np.min(ious),
            'max_iou': np.max(ious),
            'iou_50': (np.array(ious) >= 0.5).mean() * 100,
            'iou_75': (np.array(ious) >= 0.75).mean() * 100,
            'iou_90': (np.array(ious) >= 0.9).mean() * 100,
            'num_bboxes': len(ious),
            'mean_bbox_error': np.mean(bbox_errors),
            'median_bbox_error': np.median(bbox_errors),
            'mean_bbox_uncertainty': np.mean(bbox_uncertainties),
            'median_bbox_uncertainty': np.median(bbox_uncertainties),
        }

        return bbox_metrics

    def calculate_correlations(self):
        """Calculate correlation metrics focusing on OKS and uncertainty"""
        if len(self.results) == 0:
            return {}

        image_data = []
        for result in self.results:
            image_data.append({
                'image_id': result['image_id'],
                'oks': result['oks'],
                'oks_error': result['oks_error'],
                'global_uncertainty': result['global_uncertainty'],
                'overall_uncertainty': result['overall_uncertainty'],
                'mean_score': result['mean_score'],
                'bbox_uncertainty': result['bbox_uncertainty'],
                'iou': result['iou'],
                'bbox_error': result['bbox_error'],
            })

        image_df = pd.DataFrame(image_data)

        correlations = {
            'oks_vs_global_uncertainty_pearson': pearsonr(image_df['oks'], image_df['global_uncertainty'])[0],
            'oks_vs_global_uncertainty_spearman': spearmanr(image_df['oks'], image_df['global_uncertainty'])[0],
            'oks_vs_overall_uncertainty_pearson': pearsonr(image_df['oks'], image_df['overall_uncertainty'])[0],
            'oks_vs_overall_uncertainty_spearman': spearmanr(image_df['oks'], image_df['overall_uncertainty'])[0],
            'bbox_error_vs_uncertainty_pearson': pearsonr(image_df['bbox_error'], image_df['bbox_uncertainty'])[0],
            'bbox_error_vs_uncertainty_spearman': spearmanr(image_df['bbox_error'], image_df['bbox_uncertainty'])[0],
            'oks_vs_bbox_uncertainty': pearsonr(image_df['oks'], image_df['bbox_uncertainty'])[0],
            'oks_vs_score': pearsonr(image_df['oks'], image_df['mean_score'])[0],
            'oks_vs_iou': pearsonr(image_df['oks'], image_df['iou'])[0],
        }

        return correlations

    def assess_calibration_quality(self, correlations, oks_metrics, bbox_metrics, calibration_cases):
        """Assess calibration quality using statistical thresholds"""
        oks_uncertainty_corr = correlations['oks_vs_global_uncertainty_pearson']
        bbox_uncertainty_corr = correlations['bbox_error_vs_uncertainty_pearson']

        dangerous_percentage = calibration_cases['percentages']['dangerous']
        well_calibrated_percentage = (calibration_cases['percentages']['well_calibrated_high_error'] +
                                      calibration_cases['percentages']['well_calibrated_low_error'])

        # Combined calibration score
        correlation_score = max(0, -oks_uncertainty_corr)
        dangerous_penalty = dangerous_percentage / 100.0
        well_calibrated_bonus = well_calibrated_percentage / 100.0

        calibration_score = (correlation_score * 0.6 + well_calibrated_bonus * 0.3 - dangerous_penalty * 0.3)

        # Calibration quality assessment
        if calibration_score > 0.7:
            calibration_quality = "EXCELLENT"
            calibration_description = "Outstanding calibration - very few dangerous cases"
        elif calibration_score > 0.5:
            calibration_quality = "GOOD"
            calibration_description = "Good calibration - well calibrated cases dominate"
        elif calibration_score > 0.3:
            calibration_quality = "FAIR"
            calibration_description = "Fair calibration - room for improvement"
        elif calibration_score > 0.1:
            calibration_quality = "POOR"
            calibration_description = "Poor calibration - significant dangerous cases"
        else:
            calibration_quality = "FAILING"
            calibration_description = "Failing calibration - high risk of dangerous predictions"

        return {
            'quality': calibration_quality,
            'description': calibration_description,
            'score': calibration_score,
            'oks_uncertainty_correlation': oks_uncertainty_corr,
            'bbox_uncertainty_correlation': bbox_uncertainty_corr,
            'mean_oks': oks_metrics['mean_oks'],
            'mean_iou': bbox_metrics['mean_iou'],
            'dangerous_percentage': dangerous_percentage,
            'well_calibrated_percentage': well_calibrated_percentage,
            'calibration_cases': calibration_cases
        }

    def generate_report(self, use_statistical_thresholds=True):
        """Generate analysis report using statistical thresholds"""
        print("=" * 80)
        print("ENHANCED UNCERTAINTY CALIBRATION QUALITY REPORT")
        print("=" * 80)

        total_images = len(self.results)
        print(f"Total images analyzed: {total_images}")

        # Calculate statistical thresholds first
        if use_statistical_thresholds:
            self.find_statistical_thresholds()

        # Calculate metrics
        bbox_metrics = self.calculate_bbox_metrics()
        correlations = self.calculate_correlations()
        oks_metrics = self.calculate_oks_metrics()
        calibration_cases = self.classify_calibration_cases(use_statistical_thresholds)

        # Assess calibration quality
        calibration = self.assess_calibration_quality(correlations, oks_metrics, bbox_metrics, calibration_cases)

        print(f"\n--- CALIBRATION QUALITY ASSESSMENT ---")
        print(f"Quality: {calibration['quality']}")
        print(f"Assessment: {calibration['description']}")
        print(f"Calibration Score: {calibration['score']:.3f}")

        print(f"\n--- STATISTICAL THRESHOLDS ---")
        print(
            f"OKS Threshold: {calibration_cases['thresholds']['oks_threshold']:.3f} ({calibration_cases['thresholds']['source']})")
        print(
            f"Uncertainty Threshold: {calibration_cases['thresholds']['uncertainty_threshold']:.3f} ({calibration_cases['thresholds']['source']})")

        print(f"\n--- KEY CORRELATIONS ---")
        print(f"OKS vs Global Uncertainty Correlation: {calibration['oks_uncertainty_correlation']:.3f}")
        print(f"BBox Error vs Uncertainty Correlation: {calibration['bbox_uncertainty_correlation']:.3f}")

        print(f"\n--- PERFORMANCE METRICS ---")
        print(f"Mean OKS: {calibration['mean_oks']:.3f}")
        print(f"Mean IoU: {calibration['mean_iou']:.3f}")

        print(f"\n--- CALIBRATION CASE DISTRIBUTION ---")
        print(
            f"Dangerous cases (low OKS, low uncertainty): {calibration_cases['counts']['dangerous']} ({calibration_cases['percentages']['dangerous']:.1f}%)")
        print(
            f"Conservative cases (high OKS, high uncertainty): {calibration_cases['counts']['conservative']} ({calibration_cases['percentages']['conservative']:.1f}%)")
        print(
            f"Well-calibrated cases (low OKS, high uncertainty): {calibration_cases['counts']['well_calibrated_high_error']} ({calibration_cases['percentages']['well_calibrated_high_error']:.1f}%)")
        print(
            f"Well-calibrated cases (high OKS, low uncertainty): {calibration_cases['counts']['well_calibrated_low_error']} ({calibration_cases['percentages']['well_calibrated_low_error']:.1f}%)")
        print(
            f"Moderate cases: {calibration_cases['counts']['moderate']} ({calibration_cases['percentages']['moderate']:.1f}%)")

        print(f"\n--- DETAILED OKS PERFORMANCE ---")
        print(f"OKS AP@0.5: {oks_metrics['oks_50']:.1f}%")
        print(f"OKS AP@0.75: {oks_metrics['oks_75']:.1f}%")
        print(f"OKS mAP: {oks_metrics['mAP']:.1f}%")

        print(f"\n--- DETAILED BBOX PERFORMANCE ---")
        print(f"IoU AP@0.5: {bbox_metrics['iou_50']:.1f}%")
        print(f"IoU AP@0.75: {bbox_metrics['iou_75']:.1f}%")

        return calibration, correlations, oks_metrics, bbox_metrics, calibration_cases

    def plot_enhanced_calibration_analysis(self, calibration_cases):
        """Create enhanced calibration analysis plots with statistical thresholds"""
        if len(self.results) == 0:
            print("No data available for plotting")
            return

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Uncertainty Calibration Analysis with Statistical Thresholds',
                     fontsize=16, fontweight='bold')

        # Prepare data
        oks_scores = [r['oks'] for r in self.results]
        global_uncertainties = [r['global_uncertainty'] for r in self.results]
        bbox_errors = [r['bbox_error'] for r in self.results]
        bbox_uncertainties = [r['bbox_uncertainty'] for r in self.results]

        # Get thresholds
        oks_threshold = calibration_cases['thresholds']['oks_threshold']
        uncertainty_threshold = calibration_cases['thresholds']['uncertainty_threshold']

        # Colors for different case types
        colors = []
        for result in self.results:
            oks = result['oks']
            uncertainty = result['global_uncertainty']

            if oks < oks_threshold and uncertainty < uncertainty_threshold:
                colors.append('red')  # Dangerous
            elif oks > oks_threshold and uncertainty > uncertainty_threshold:
                colors.append('orange')  # Conservative
            elif oks < oks_threshold and uncertainty > uncertainty_threshold:
                colors.append('yellow')  # Well-calibrated (high error)
            elif oks > oks_threshold and uncertainty < uncertainty_threshold:
                colors.append('green')  # Well-calibrated (low error)
            else:
                colors.append('gray')  # Moderate

        # Plot 1: OKS vs Global Uncertainty with case classification
        oks_corr = pearsonr(oks_scores, global_uncertainties)[0]
        scatter1 = ax1.scatter(global_uncertainties, oks_scores, alpha=0.7, c=colors, s=50)
        ax1.axhline(y=oks_threshold, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax1.axvline(x=uncertainty_threshold, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Global Uncertainty')
        ax1.set_ylabel('OKS Score')
        ax1.set_title(f'OKS vs Global Uncertainty\nCorrelation: {oks_corr:.3f}', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label=f'Dangerous ({calibration_cases["counts"]["dangerous"]})'),
            Patch(facecolor='orange', label=f'Conservative ({calibration_cases["counts"]["conservative"]})'),
            Patch(facecolor='yellow',
                  label=f'Well-cal High Err ({calibration_cases["counts"]["well_calibrated_high_error"]})'),
            Patch(facecolor='green',
                  label=f'Well-cal Low Err ({calibration_cases["counts"]["well_calibrated_low_error"]})'),
            Patch(facecolor='gray', label=f'Moderate ({calibration_cases["counts"]["moderate"]})')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Plot 2: BBox Error vs BBox Uncertainty
        bbox_corr = pearsonr(bbox_uncertainties, bbox_errors)[0]
        ax2.scatter(bbox_uncertainties, bbox_errors, alpha=0.6, color='blue')
        z = np.polyfit(bbox_uncertainties, bbox_errors, 1)
        p = np.poly1d(z)
        ax2.plot(bbox_uncertainties, p(bbox_uncertainties), "r--", alpha=0.8, linewidth=2)
        ax2.set_xlabel('BBox Uncertainty')
        ax2.set_ylabel('BBox Error (1 - IoU)')
        ax2.set_title(f'BBox Error vs Uncertainty\nCorrelation: {bbox_corr:.3f}', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: OKS distribution with statistical threshold
        ax3.hist(oks_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax3.axvline(oks_threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Statistical Threshold = {oks_threshold:.3f}')
        ax3.set_xlabel('OKS Score')
        ax3.set_ylabel('Density')
        ax3.set_title('OKS Distribution with Statistical Threshold', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Uncertainty distribution with statistical threshold
        ax4.hist(global_uncertainties, bins=20, alpha=0.7, color='lightcoral',
                 edgecolor='black', density=True)
        ax4.axvline(uncertainty_threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Statistical Threshold = {uncertainty_threshold:.3f}')
        ax4.set_xlabel('Global Uncertainty')
        ax4.set_ylabel('Density')
        ax4.set_title('Uncertainty Distribution with Statistical Threshold', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('enhanced_calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Saved enhanced calibration plots to 'enhanced_calibration_analysis.png'")


# Usage example
if __name__ == "__main__":
    # Initialize enhanced analyzer
    analyzer = EnhancedUncertaintyAnalyzer(
        annotations_path="/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/annotations/tomato_keypoints_test1.json",
        uncertainty_dir="output/json"
    )

    # Run analysis
    analyzer.analyze_all_files()

    # Generate enhanced report with statistical thresholds
    print(f"\n{'=' * 80}")
    print(f"ENHANCED CALIBRATION QUALITY ANALYSIS WITH STATISTICAL THRESHOLDS")
    print(f"{'=' * 80}")

    calibration, correlations, oks_metrics, bbox_metrics, calibration_cases = analyzer.generate_report(
        use_statistical_thresholds=True)

    # Create enhanced calibration plots
    analyzer.plot_enhanced_calibration_analysis(calibration_cases)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL ENHANCED CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"Threshold Source: {calibration_cases['thresholds']['source']}")
    print(f"OKS Threshold: {calibration_cases['thresholds']['oks_threshold']:.3f}")
    print(f"Uncertainty Threshold: {calibration_cases['thresholds']['uncertainty_threshold']:.3f}")
    print(f"OKS vs Global Uncertainty Correlation: {calibration['oks_uncertainty_correlation']:.3f}")
    print(f"BBox Error vs Uncertainty Correlation: {calibration['bbox_uncertainty_correlation']:.3f}")
    print(f"Mean OKS: {calibration['mean_oks']:.3f}")
    print(f"Mean IoU: {calibration['mean_iou']:.3f}")
    print(f"Dangerous cases: {calibration['dangerous_percentage']:.1f}%")
    print(f"Well-calibrated cases: {calibration['well_calibrated_percentage']:.1f}%")
    print(f"Calibration Quality: {calibration['quality']}")