import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind, ks_2samp
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class EnhancedUncertaintyAnalyzer:
    def __init__(self, annotations_path, uncertainty_dir):
        self.annotations_path = annotations_path
        self.uncertainty_dir = Path(uncertainty_dir)
        self.results = []

        # OKS parameters (one per keypoint)
        self.oks_sigmas = np.array([0.035, 0.035, 0.15, 0.15, 0.15])
        self.oks_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        # Statistical thresholds (calculated)
        self.statistical_thresholds = None
        self.annotations_data = None

    # ----------------- loading / basic utils -----------------
    def load_annotations(self):
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
                per_kp_oks.append(None)  # Changed to None for invisible keypoints
                continue

            d = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            sigma = self.oks_sigmas[i] if i < len(self.oks_sigmas) else 0.15
            kp_oks = np.exp(-d ** 2 / (2 * (s ** 2) * (sigma ** 2)))

            oks_sum += kp_oks
            valid_kps += 1
            per_kp_oks.append(kp_oks)

        oks = oks_sum / valid_kps if valid_kps > 0 else 0.0
        return oks, per_kp_oks

    def _extract_per_keypoint_list(self, kp_uncertainty, candidate_names, num_kps=5):
        if kp_uncertainty is None:
            return None

        if isinstance(kp_uncertainty, dict):
            for name in candidate_names:
                if name in kp_uncertainty and isinstance(kp_uncertainty[name], (list, tuple, np.ndarray)):
                    arr = list(kp_uncertainty[name])
                    if len(arr) >= num_kps:
                        return arr[:num_kps]
                    else:
                        pad_val = arr[-1] if len(arr) > 0 else None
                        return arr + [pad_val] * (num_kps - len(arr))

        if isinstance(kp_uncertainty, (list, tuple, np.ndarray)):
            arr = list(kp_uncertainty)
            if len(arr) >= num_kps:
                return arr[:num_kps]
            else:
                pad_val = arr[-1] if len(arr) > 0 else None
                return arr + [pad_val] * (num_kps - len(arr))

        return None

    # ----------------- analysis -----------------
    def analyze_all_files(self):
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

                with open(u_file, 'r') as f:
                    uncertainty_data = json.load(f)

                pred_keypoints = uncertainty_data.get('mean_keypoints', None)
                if pred_keypoints is None:
                    pred_keypoints = uncertainty_data.get('pred_keypoints', None)
                if pred_keypoints is None:
                    continue

                gt_keypoints = self.image_to_keypoints.get(image_id, None)
                if gt_keypoints is None:
                    continue

                pred_bbox = uncertainty_data.get('mean_bbox', None)
                if pred_bbox is None:
                    pred_bbox = uncertainty_data.get('pred_bbox', None)
                if pred_bbox is None:
                    pred_bbox = self.image_to_bbox.get(image_id, [0, 0, 1, 1])

                gt_bbox = self.image_to_bbox[image_id]
                bbox_area = pred_bbox[2] * pred_bbox[3] if len(pred_bbox) >= 4 else 1.0

                iou = self.calculate_iou(pred_bbox, gt_bbox)
                oks, per_kp_oks = self.calculate_oks(pred_keypoints, gt_keypoints, bbox_area)

                kp_uncertainty = uncertainty_data.get('keypoint_uncertainty', {})
                bbox_uncertainty = uncertainty_data.get('bbox_uncertainty', {})

                num_kps = 5
                per_kp_overall = self._extract_per_keypoint_list(
                    kp_uncertainty,
                    candidate_names=['per_keypoint_overall', 'per_kp_overall', 'per_keypoints', 'per_kps', 'per_kp'],
                    num_kps=num_kps
                )
                per_kp_epistemic = self._extract_per_keypoint_list(
                    kp_uncertainty,
                    candidate_names=['per_keypoint_epistemic', 'per_kp_epistemic', 'per_kp_epi'],
                    num_kps=num_kps
                )
                per_kp_aleatoric = self._extract_per_keypoint_list(
                    kp_uncertainty,
                    candidate_names=['per_keypoint_aleatoric', 'per_kp_aleatoric', 'per_kp_alea'],
                    num_kps=num_kps
                )

                overall_unc_scalar = kp_uncertainty.get('overall_uncertainty', None) if isinstance(kp_uncertainty, dict) else None
                epi_unc_scalar = kp_uncertainty.get('epistemic_uncertainty', None) if isinstance(kp_uncertainty, dict) else None
                alea_unc_scalar = kp_uncertainty.get('aleatoric_uncertainty', None) if isinstance(kp_uncertainty, dict) else None

                if per_kp_overall is None:
                    per_kp_overall = [overall_unc_scalar if overall_unc_scalar is not None else np.nan] * num_kps
                if per_kp_epistemic is None:
                    per_kp_epistemic = [epi_unc_scalar if epi_unc_scalar is not None else np.nan] * num_kps
                if per_kp_aleatoric is None:
                    per_kp_aleatoric = [alea_unc_scalar if alea_unc_scalar is not None else np.nan] * num_kps

                result = {
                    'image_id': image_id,
                    'file_name': image_name,
                    'global_uncertainty': uncertainty_data.get('global_uncertainty', np.nan),
                    'mean_score': uncertainty_data.get('mean_score', np.nan),
                    'num_valid_passes': uncertainty_data.get('num_valid_passes', np.nan),
                    'overall_uncertainty': kp_uncertainty.get('overall_uncertainty', np.nan) if isinstance(kp_uncertainty, dict) else np.nan,
                    'epistemic_uncertainty': kp_uncertainty.get('epistemic_uncertainty', np.nan) if isinstance(kp_uncertainty, dict) else np.nan,
                    'aleatoric_uncertainty': kp_uncertainty.get('aleatoric_uncertainty', np.nan) if isinstance(kp_uncertainty, dict) else np.nan,
                    'mean_bbox': pred_bbox,
                    'iou': iou,
                    'bbox_error': 1.0 - iou,
                    'bbox_uncertainty': bbox_uncertainty.get('overall_uncertainty', np.nan) if isinstance(bbox_uncertainty, dict) else np.nan,
                    'oks': oks,
                    'oks_error': 1.0 - oks,
                    'per_kp_oks': per_kp_oks,
                    'mean_keypoints': pred_keypoints,
                    'per_kp_overall_uncertainty': per_kp_overall,
                    'per_kp_epistemic_uncertainty': per_kp_epistemic,
                    'per_kp_aleatoric_uncertainty': per_kp_aleatoric,
                    'gt_keypoints': gt_keypoints.tolist() if isinstance(gt_keypoints, np.ndarray) else gt_keypoints
                }

                self.results.append(result)

            except Exception as e:
                print(f"Skipping file {u_file.name} due to error: {e}")
                continue

        print(f"Successfully processed {len(self.results)} files")

    # ----------------- thresholds / classification -----------------
    def find_statistical_thresholds(self):
        if len(self.results) == 0:
            return None

        oks_values = np.array([r['oks'] for r in self.results])
        uncertainty_values = np.array([r['global_uncertainty'] for r in self.results])

        def find_optimal_threshold(values, name):
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
                    _, up = mannwhitneyu(group1, group2)
                    _, tp = ttest_ind(group1, group2)
                    ks_stat, ks_p = ks_2samp(group1, group2)
                    current_p = min(up, tp, ks_p)
                    if current_p < best_pvalue:
                        best_pvalue = current_p
                        best_threshold = threshold
                        best_method = f"p={current_p:.6f}"
                except Exception:
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
        if len(self.results) == 0:
            return {}

        if use_statistical_thresholds and self.statistical_thresholds is None:
            self.find_statistical_thresholds()

        if use_statistical_thresholds and self.statistical_thresholds:
            oks_threshold = self.statistical_thresholds['oks_threshold']
            uncertainty_threshold = self.statistical_thresholds['uncertainty_threshold']
            threshold_source = "statistical"
        else:
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

    # ----------------- metrics / correlations -----------------
    def calculate_oks_metrics(self):
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
        oks_ap_scores = {}
        for threshold in self.oks_thresholds:
            oks_ap_scores[f'oks_{int(threshold * 100)}'] = (np.array(oks_scores) >= threshold).mean() * 100
        oks_metrics.update(oks_ap_scores)
        oks_metrics['mAP'] = np.mean(list(oks_ap_scores.values()))
        return oks_metrics

    def calculate_bbox_metrics(self):
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
        oks_uncertainty_corr = correlations['oks_vs_global_uncertainty_pearson']
        bbox_uncertainty_corr = correlations['bbox_error_vs_uncertainty_pearson']
        dangerous_percentage = calibration_cases['percentages']['dangerous']
        well_calibrated_percentage = (calibration_cases['percentages']['well_calibrated_high_error'] +
                                      calibration_cases['percentages']['well_calibrated_low_error'])
        correlation_score = max(0, -oks_uncertainty_corr)
        dangerous_penalty = dangerous_percentage / 100.0
        well_calibrated_bonus = well_calibrated_percentage / 100.0
        calibration_score = (correlation_score * 0.6 + well_calibrated_bonus * 0.3 - dangerous_penalty * 0.3)

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
        if use_statistical_thresholds:
            self.find_statistical_thresholds()
        bbox_metrics = self.calculate_bbox_metrics()
        correlations = self.calculate_correlations()
        oks_metrics = self.calculate_oks_metrics()
        calibration_cases = self.classify_calibration_cases(use_statistical_thresholds)
        calibration = self.assess_calibration_quality(correlations, oks_metrics, bbox_metrics, calibration_cases)

        print(f"\n--- STATISTICAL THRESHOLDS ---")
        print(f"OKS Threshold: {calibration_cases['thresholds']['oks_threshold']:.3f} ({calibration_cases['thresholds']['source']})")
        print(f"Uncertainty Threshold: {calibration_cases['thresholds']['uncertainty_threshold']:.3f} ({calibration_cases['thresholds']['source']})")

        print(f"\n--- DETAILED OKS PERFORMANCE ---")
        print(f"OKS AP@0.5: {oks_metrics.get('oks_50', 0):.1f}%")
        print(f"OKS AP@0.75: {oks_metrics.get('oks_75', 0):.1f}%")
        print(f"OKS mAP: {oks_metrics.get('mAP', 0):.1f}%")

        print(f"\n--- DETAILED BBOX PERFORMANCE ---")
        print(f"IoU AP@0.5: {bbox_metrics.get('iou_50', 0):.1f}%")
        print(f"IoU AP@0.75: {bbox_metrics.get('iou_75', 0):.1f}%")

        return calibration, correlations, oks_metrics, bbox_metrics, calibration_cases


    def visualize_images(self, output_dir="annotated_images", image_root=".", keypoint_names=None,
                         show_bbox=True, font_path=None, kp_radius=6, use_statistical_thresholds=True):
        """
        Draw predictions and draw a footer under each image containing lines like:
        "KP1: OKS=0.83, U=0.12"

        - output_dir: where annotated images will be written
        - image_root: root directory to resolve image file names from annotations
        - keypoint_names: optional list of 5 names
        - use_statistical_thresholds: use thresholds computed earlier to determine safety coloring
        """
        if len(self.results) == 0:
            print("No results available. Run analyze_all_files() first.")
            return

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_root = Path(image_root)

        # thresholds for coloring per-keypoint safety
        if use_statistical_thresholds and self.statistical_thresholds is None:
            self.find_statistical_thresholds()

        if use_statistical_thresholds and self.statistical_thresholds:
            oks_thr = self.statistical_thresholds['oks_threshold']
            unc_thr = self.statistical_thresholds['uncertainty_threshold']
            thr_source = "statistical"
        else:
            oks_thr = 0.5
            unc_thr = 0.5
            thr_source = "0.5"

        if keypoint_names is None:
            keypoint_names = ['KP1', 'KP2', 'KP3', 'KP4', 'KP5']

        # Try to load a truetype font if provided, otherwise default
        try:
            font = ImageFont.truetype(font_path, size=14) if font_path else ImageFont.load_default()
            small_font = ImageFont.truetype(font_path, size=12) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        for res in self.results:
            file_name = res['file_name']
            img_path = image_root / file_name
            if not img_path.exists():
                # fallback: try relative path in same dir as annotations
                ann_dir = Path(self.annotations_path).parent
                alt = ann_dir / file_name
                if alt.exists():
                    img_path = alt
                else:
                    print(f"Image not found for {file_name}, skipping.")
                    continue

            img = Image.open(img_path).convert("RGBA")
            draw = ImageDraw.Draw(img)

            # Draw bounding boxes (pred / gt)
            pred_bbox = res.get('mean_bbox', None)
            gt_bbox = self.image_to_bbox.get(res['image_id'], None)

            def draw_rect(drawobj, bbox, outline=(0, 255, 0), width=3):
                if bbox and len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    drawobj.rectangle([x, y, x + w, y + h], outline=outline, width=width)

            if show_bbox and pred_bbox is not None:
                draw_rect(draw, pred_bbox, outline=(0, 255, 0), width=3)  # predicted bbox (green)
            if show_bbox and gt_bbox is not None:
                draw_rect(draw, gt_bbox, outline=(255, 0, 0), width=2)  # ground-truth bbox (red)

            # Draw keypoints markers (kept near their locations)
            pred_kps = list(res.get('mean_keypoints', []))
            gt_kps = np.array(res.get('gt_keypoints', np.zeros((5, 3))).copy()).reshape(-1, 3)
            per_kp_oks = res.get('per_kp_oks', [None] * 5)
            per_kp_overall_unc = res.get('per_kp_overall_uncertainty', [np.nan] * 5)
            per_kp_epi_unc = res.get('per_kp_epistemic_uncertainty', [np.nan] * 5)
            per_kp_alea_unc = res.get('per_kp_aleatoric_uncertainty', [np.nan] * 5)

            for i in range(5):
                px = pred_kps[i * 2] if len(pred_kps) > i * 2 else None
                py = pred_kps[i * 2 + 1] if len(pred_kps) > i * 2 + 1 else None
                gx, gy, gvis = (gt_kps[i, 0], gt_kps[i, 1], int(gt_kps[i, 2])) if i < gt_kps.shape[0] else (None, None,
                                                                                                            0)

                if px is None or py is None or np.isnan(px) or np.isnan(py):
                    continue

                # decide color for keypoint marker based on per-keypoint overall uncertainty vs threshold
                kp_unc = per_kp_overall_unc[i] if i < len(per_kp_overall_unc) else np.nan
                kp_oks = per_kp_oks[i] if i < len(per_kp_oks) else None

                # Default color: green (safe)
                color = (0, 255, 0)
                # If uncertainty high => yellow, if OKS low & uncertainty low => red (dangerous)
                try:
                    if not np.isnan(kp_unc) and kp_unc > unc_thr:
                        color = (255, 200, 0)  # yellow-ish uncertain
                    if (kp_oks is not None) and (kp_oks < oks_thr) and (not np.isnan(kp_unc)) and (kp_unc < unc_thr):
                        color = (255, 0, 0)  # red: low OKS but low reported uncertainty = dangerous
                except Exception:
                    color = (0, 255, 0)

                # Draw circle
                r = kp_radius
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(0, 0, 0))

                # Optionally draw GT keypoint if visible
                if gx is not None and gvis == 2:
                    rr = max(2, r // 2)
                    draw.ellipse([gx - rr, gy - rr, gx + rr, gy + rr], fill=(0, 0, 255, 180), outline=(0, 0, 0))

            # ------------------ Build footer text lines ------------------
            lines = []
            # include threshold source and values as a header line
            lines.append(f"Thresholds({thr_source}): OKS>={oks_thr:.2f}, U<={unc_thr:.2f}")

            for i in range(5):
                name = keypoint_names[i] if i < len(keypoint_names) else f"KP{i + 1}"
                kp_oks = per_kp_oks[i] if i < len(per_kp_oks) else None
                kp_unc = per_kp_overall_unc[i] if i < len(per_kp_overall_unc) else np.nan

                oks_s = f"{kp_oks:.2f}" if (
                            kp_oks is not None and not (isinstance(kp_oks, float) and np.isnan(kp_oks))) else "-"
                unc_s = f"{kp_unc:.2f}" if (not np.isnan(kp_unc)) else "-"

                lines.append(f"{name}: OKS={oks_s}, U={unc_s}")

            # compute footer height
            padding = 8
            line_spacing = 4
            # use small_font to measure
            # replaced getsize() with textbbox()
            bbox = small_font.getbbox("Ay")
            sample_h = (bbox[3] - bbox[1])  # height from getbbox()

            # ------------------ create footer and save image ------------------
            # measure each line width/height using getbbox()
            padding = 8
            line_spacing = 4
            line_heights = []
            line_widths = []
            for ln in lines:
                bb = small_font.getbbox(ln)
                w_line = bb[2] - bb[0]
                h_line = bb[3] - bb[1]
                line_widths.append(w_line)
                line_heights.append(h_line)

            # fallback when font measurement fails
            if len(line_heights) == 0:
                line_h = sample_h
                max_line_w = img.size[0] - padding * 2
            else:
                line_h = max(line_heights)
                max_line_w = int(max(line_widths))

            footer_h = padding * 2 + len(lines) * (line_h + line_spacing) - line_spacing

            # create a new image with extra space for the footer
            orig_w, orig_h = img.size
            new_w = max(orig_w, max_line_w + padding * 2)
            footer_bg = (25, 25, 25)  # dark footer background
            new_img = Image.new('RGB', (new_w, orig_h + footer_h), footer_bg)
            # paste original image at top-left (or centered horizontally if new_w > orig_w)
            paste_x = (new_w - orig_w) // 2 if new_w > orig_w else 0
            new_img.paste(img.convert('RGB'), (paste_x, 0))

            # draw footer text
            drawf = ImageDraw.Draw(new_img)
            x = padding
            y = orig_h + padding
            for li, line in enumerate(lines):
                drawf.text((x, y + li * (line_h + line_spacing)), line, font=small_font, fill=(255, 255, 255))

            # Save annotated image
            out_path = out_dir / file_name
            try:
                new_img.save(out_path, quality=90)
            except Exception as e:
                print(f"Failed to save annotated image {out_path}: {e}")

    # ----------------- per-keypoint table generation -----------------
    def generate_per_keypoint_predictions(self, output_json_path=None, output_csv_path=None, keypoint_names=None):
        if len(self.results) == 0:
            print("No results available. Run analyze_all_files() first.")
            return pd.DataFrame()

        if keypoint_names is None:
            keypoint_names = [f'kp_{i}' for i in range(5)]

        rows = []
        for res in self.results:
            pred_kps = res.get('mean_keypoints', None)
            if pred_kps is None:
                continue
            pred_kps = list(pred_kps)
            gt_kps = res.get('gt_keypoints', None)
            if gt_kps is None:
                continue
            gt_kps = np.array(gt_kps).reshape(-1, 3)

            per_kp_oks = res.get('per_kp_oks', [None] * 5)
            per_kp_overall_unc = res.get('per_kp_overall_uncertainty', [np.nan] * 5)
            per_kp_epi_unc = res.get('per_kp_epistemic_uncertainty', [np.nan] * 5)
            per_kp_alea_unc = res.get('per_kp_aleatoric_uncertainty', [np.nan] * 5)

            for i in range(5):
                pred_x = pred_kps[i * 2] if len(pred_kps) > i * 2 else np.nan
                pred_y = pred_kps[i * 2 + 1] if len(pred_kps) > i * 2 + 1 else np.nan
                gt_x, gt_y, gt_vis = (gt_kps[i, 0], gt_kps[i, 1], int(gt_kps[i, 2])) if i < gt_kps.shape[0] else (np.nan, np.nan, 0)
                row = {
                    'image_id': res['image_id'],
                    'file_name': res['file_name'],
                    'kp_index': i,
                    'kp_name': keypoint_names[i] if i < len(keypoint_names) else f'kp_{i}',
                    'pred_x': float(pred_x) if not pd.isna(pred_x) else np.nan,
                    'pred_y': float(pred_y) if not pd.isna(pred_y) else np.nan,
                    'gt_x': float(gt_x) if not pd.isna(gt_x) else np.nan,
                    'gt_y': float(gt_y) if not pd.isna(gt_y) else np.nan,
                    'gt_vis': int(gt_vis),
                    'kp_oks': float(per_kp_oks[i]) if (i < len(per_kp_oks) and per_kp_oks[i] is not None) else np.nan,
                    'kp_overall_uncertainty': float(per_kp_overall_unc[i]) if (i < len(per_kp_overall_unc) and per_kp_overall_unc[i] is not None) else np.nan,
                    'kp_epistemic_uncertainty': float(per_kp_epi_unc[i]) if (i < len(per_kp_epi_unc) and per_kp_epi_unc[i] is not None) else np.nan,
                    'kp_aleatoric_uncertainty': float(per_kp_alea_unc[i]) if (i < len(per_kp_alea_unc) and per_kp_alea_unc[i] is not None) else np.nan,
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        if output_json_path is not None:
            try:
                df.to_json(output_json_path, orient='records', indent=2)
                print(f"Saved per-keypoint JSON to {output_json_path}")
            except Exception as e:
                print(f"Failed to save JSON: {e}")

        if output_csv_path is not None:
            try:
                df.to_csv(output_csv_path, index=False)
                print(f"Saved per-keypoint CSV to {output_csv_path}")
            except Exception as e:
                print(f"Failed to save CSV: {e}")

        return df


# ----------------- usage example -----------------
if __name__ == "__main__":
    analyzer = EnhancedUncertaintyAnalyzer(
        annotations_path="/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/annotations/tomato_keypoints_test1.json",
        uncertainty_dir="yolonas_output/json"
    )

    # Run analysis (this will populate analyzer.results)
    analyzer.analyze_all_files()

    # Generate the report (also computes statistical thresholds)
    calibration, correlations, oks_metrics, bbox_metrics, calibration_cases = analyzer.generate_report(
        use_statistical_thresholds=True)

    # Create visualized images with overlays.
    # image_root is where your actual .png images live; change as needed.
    # This will write annotated images to 'annotated_images' directory.
    analyzer.visualize_images(output_dir="yolonas_output/annotated_images",
                              image_root="/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/test",
                              keypoint_names=['Top', 'Bottom', 'Cut', 'Further', 'End'],
                              show_bbox=True,
                              font_path=None,
                              kp_radius=6,
                              use_statistical_thresholds=False)

    # Optional: save per-keypoint table
    per_kp_df = analyzer.generate_per_keypoint_predictions(
        output_json_path="yolonas_output/per_keypoint_predictions.json",
        output_csv_path="yolonas_output/per_keypoint_predictions.csv",
        keypoint_names=['Top', 'Bottom', 'Cut', 'Further', 'End']
    )

    # Print summary
    print("\nFINAL SUMMARY")
    print(f"OKS vs Global Uncertainty Correlation: {calibration['oks_uncertainty_correlation']:.3f}")
    print(f"Mean OKS: {calibration['mean_oks']:.3f}")
    print(f"Annotated images saved to 'annotated_images/'")
