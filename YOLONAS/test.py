import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
from super_gradients.training import models



class YOLONASMCPredictor:
    """YOLO-NAS Pose predictor with MC Dropout uncertainty estimation."""

    def __init__(self, checkpoint_path: str, num_keypoints: int = 5,
                 dropout_rate: float = 0.10, conf_threshold: float = 0.20,
                 device: str = "cuda"):

        self.num_keypoints = num_keypoints
        self.dropout_rate = dropout_rate
        self.conf_threshold = conf_threshold
        self.device = device
        #change
        # Load model
        self.model = models.get(
            "yolo_nas_pose_l",
            num_classes=num_keypoints,
            checkpoint_path=checkpoint_path
        ).to(device)

        # Store original model for reference
        self.original_model = self.model
        self.model = self._wrap_with_mc_dropout(self.model, dropout_rate)

        print(f"YOLO-NAS MC Predictor ready on {device}")

    def _wrap_with_mc_dropout(self, model, dropout_rate):
        """Apply MC Dropout with regular Dropout instead of Dropout2d."""
        import copy
        model = copy.deepcopy(model)
        model.drop_head = True

        print(f"Applying MC Dropout (regular) to YOLO-NAS Pose heads...")

        #add dropout into backbone
        # backbone_modules = []
        # for name, module in model.backbone.named_children():
        #     if 'stage' in name and int(name[-1]) >= 3:  # Last 2 stages
        #         backbone_modules.append((name, module))
        #
        # for name, module in backbone_modules[-2:]:  # Last 2 stages
        #     if hasattr(module, 'blocks'):
        #         original_blocks = module.blocks
        #         module.blocks = nn.Sequential(
        #             original_blocks,
        #             nn.Dropout2d(p=0.03)
        #         )
        #         print(f"  - Added dropout to backbone.{name}.blocks")

        #neck start
        # if hasattr(model.neck, 'neck1'):
        #     original_blocks = model.neck.neck1.blocks
        #     model.neck.neck1.blocks = nn.Sequential(
        #         nn.Dropout2d(p=dropout_rate*3/2),
        #         original_blocks
        #     )
        # if hasattr(model.neck, 'neck2'):
        #     original_blocks = model.neck.neck2.blocks
        #     model.neck.neck2.blocks = nn.Sequential(
        #         nn.Dropout2d(p=dropout_rate*3/2),
        #         original_blocks
        #     )


        # neck end
        # neck2: YoloNASUpStage
        if hasattr(model.neck, 'neck2'):
            original_reduce = model.neck.neck2.reduce_after_concat
            model.neck.neck2.reduce_after_concat = nn.Sequential(
                nn.Dropout2d(p=dropout_rate / 2),
                original_reduce
            )

            original_blocks = model.neck.neck2.blocks
            model.neck.neck2.blocks = nn.Sequential(
                nn.Dropout2d(p=dropout_rate / 2),
                original_blocks
            )

        # neck3: YoloNASDownStage
        if hasattr(model.neck, 'neck3'):
            # Dropout before CSP blocks
            original_blocks = model.neck.neck3.blocks
            model.neck.neck3.blocks = nn.Sequential(
                nn.Dropout2d(p=dropout_rate),
                original_blocks
            )

        # neck4: YoloNASDownStage
        if hasattr(model.neck, 'neck4'):
            original_blocks = model.neck.neck4.blocks
            model.neck.neck4.blocks = nn.Sequential(
                nn.Dropout2d(p=dropout_rate),
                original_blocks
            )

        # #before predict head
        # if hasattr(model.heads, 'head1'):
        #     original_heads = model.heads.head1
        #     new_heads = nn.Sequential(
        #         nn.Dropout2d(dropout_rate),
        #         original_heads
        #     )
        #     model.heads.head1 = new_heads
        #
        # if hasattr(model.heads, 'head2'):
        #     original_heads = model.heads.head2
        #     new_heads = nn.Sequential(
        #         nn.Dropout2d(dropout_rate),
        #         original_heads
        #     )
        #     model.heads.head2 = new_heads
        #
        # if hasattr(model.heads, 'head3'):
        #     original_heads = model.heads.head3
        #     new_heads = nn.Sequential(
        #         nn.Dropout2d(dropout_rate),
        #         original_heads
        #     )
        #     model.heads.head3 = new_heads

        return model


    def predict_single_pass(self, image_path: str) -> Dict[str, Any]:
        with torch.no_grad():
            prediction = self.model.predict(image_path, conf=self.conf_threshold, fuse_model=False)
            prediction = self._get_top_confidence_prediction(prediction)

            if prediction.prediction is None or len(prediction.prediction.scores) == 0:
                return None

            pred = prediction.prediction

            # Extract predictions from the top detection
            score = float(pred.scores[0])
            bbox_xyxy = pred.bboxes_xyxy[0]
            poses = pred.poses[0]  # Shape: (5, 3) - [x, y, confidence] for each keypoint

            # Convert bbox to xywh format
            bbox_xywh = [
                float(bbox_xyxy[0]),
                float(bbox_xyxy[1]),
                float(bbox_xyxy[2] - bbox_xyxy[0]),
                float(bbox_xyxy[3] - bbox_xyxy[1])
            ]

            # Extract keypoint coordinates and scores
            keypoints_xy = poses[:, :2].flatten()
            keypoint_scores = poses[:, 2]

            return {
                'score': score,
                'bbox': bbox_xywh,
                'bbox_xyxy': bbox_xyxy.tolist(),
                'keypoints': poses,
                'keypoints_xy': keypoints_xy,
                'keypoint_scores': keypoint_scores,
                'keypoints_full': poses
            }

    def _get_top_confidence_prediction(self, prediction):
        """Filter prediction to show only the detection with highest confidence."""
        if hasattr(prediction, 'prediction') and prediction.prediction is not None:
            pred = prediction.prediction

            # Get scores/confidences
            if hasattr(pred, 'scores') and len(pred.scores) > 0:
                # Find index of highest confidence
                max_idx = np.argmax(pred.scores)

                # Keep only the top detection
                pred.scores = pred.scores[max_idx:max_idx + 1]

                if hasattr(pred, 'poses'):
                    pred.poses = pred.poses[max_idx:max_idx + 1]
                if hasattr(pred, 'bboxes_xyxy'):
                    pred.bboxes_xyxy = pred.bboxes_xyxy[max_idx:max_idx + 1]

        return prediction

    def _calculate_bbox_uncertainty(self, bboxes_array: np.ndarray, image_area: float) -> Dict[str, Any]:
        """Calculate comprehensive bbox uncertainty metrics."""
        if len(bboxes_array) == 0:
            return {
                'overall_uncertainty': 1.0,
                'position_uncertainty': 1.0,
                'size_uncertainty': 1.0,
                'iou_variance': 1.0,
                'area_variance': 1.0,
                'center_std': 100.0,
                'normalized_center_std': 1.0
            }

        # Convert to numpy for calculations
        bboxes = np.array(bboxes_array)

        # Calculate mean bbox
        mean_bbox = np.mean(bboxes, axis=0)
        mean_x, mean_y, mean_w, mean_h = mean_bbox

        # Calculate variances
        bbox_variance = np.var(bboxes, axis=0)
        var_x, var_y, var_w, var_h = bbox_variance

        # Position uncertainty (center point stability)
        centers = np.column_stack([bboxes[:, 0] + bboxes[:, 2] / 2,
                                   bboxes[:, 1] + bboxes[:, 3] / 2])
        center_std = np.mean(np.std(centers, axis=0))

        # Normalize by image dimensions for scale invariance
        normalized_center_std = center_std / np.sqrt(image_area)

        # Size uncertainty
        sizes = np.column_stack([bboxes[:, 2], bboxes[:, 3]])  # width, height
        size_std = np.mean(np.std(sizes, axis=0))
        normalized_size_std = size_std / np.sqrt(image_area)

        # Calculate IoU variance between samples and mean bbox
        ious = []
        for bbox in bboxes:
            iou = self._calculate_iou(bbox, mean_bbox)
            ious.append(iou)

        iou_variance = np.var(ious) if ious else 1.0
        mean_iou = np.mean(ious) if ious else 0.0

        # Area variance
        areas = bboxes[:, 2] * bboxes[:, 3]  # w * h
        area_variance = np.var(areas) / (image_area ** 2)  # Normalized

        # Combined bbox uncertainty
        position_uncertainty = min(normalized_center_std * 10, 1.0)
        size_uncertainty = min(normalized_size_std * 10, 1.0)
        iou_uncertainty = min(iou_variance * 5, 1.0)

        # Overall bbox uncertainty (weighted combination)
        overall_bbox_uncertainty = 0.4 * position_uncertainty + 0.4 * size_uncertainty + 0.2 * iou_uncertainty

        return {
            'overall_uncertainty': float(overall_bbox_uncertainty),
            'position_uncertainty': float(position_uncertainty),
            'size_uncertainty': float(size_uncertainty),
            'iou_variance': float(iou_variance),
            'mean_iou': float(mean_iou),
            'area_variance': float(area_variance),
            'center_std': float(center_std),
            'normalized_center_std': float(normalized_center_std),
            'bbox_variance': [float(var_x), float(var_y), float(var_w), float(var_h)],
            'num_bbox_samples': len(bboxes_array)
        }

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in [x, y, w, h] format."""
        # Convert to [x1, y1, x2, y2]
        box1_x1, box1_y1, box1_w, box1_h = box1
        box2_x1, box2_y1, box2_w, box2_h = box2

        box1_x2, box1_y2 = box1_x1 + box1_w, box1_y1 + box1_h
        box2_x2, box2_y2 = box2_x1 + box2_w, box2_y1 + box2_h

        # Calculate intersection
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union
        area1 = box1_w * box1_h
        area2 = box2_w * box2_h
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _calculate_keypoint_distances(self, mean_keypoints):
        """Calculate distances between all keypoint pairs."""
        distances = np.zeros((self.num_keypoints, self.num_keypoints))
        for i in range(self.num_keypoints):
            for j in range(self.num_keypoints):
                if i != j:
                    dist = np.linalg.norm(mean_keypoints[i] - mean_keypoints[j])
                    distances[i, j] = dist
        return distances

    def _calculate_reference_scales(self, keypoint_distances, bbox_reference_scale):
        """Calculate reference scales for each keypoint with bbox awareness."""
        reference_scales = []

        for kp_idx in range(self.num_keypoints):
            distances_to_others = [keypoint_distances[kp_idx, j] for j in range(self.num_keypoints) if j != kp_idx]

            if len(distances_to_others) >= 2:
                closest_distances = sorted(distances_to_others)[:2]
                local_scale = np.mean(closest_distances)
            elif len(distances_to_others) >= 1:
                local_scale = min(distances_to_others)
            else:
                local_scale = bbox_reference_scale * 0.3

            reference_scale = 0.7 * local_scale + 0.3 * bbox_reference_scale
            reference_scale = max(reference_scale, bbox_reference_scale * 0.1)
            reference_scale = min(reference_scale, bbox_reference_scale * 2.0)

            reference_scales.append(reference_scale)

        return reference_scales

    def calculate_comprehensive_uncertainty(self, image_path: str, num_passes: int = 30) -> Dict[str, Any]:
        """Calculate comprehensive uncertainty for both bbox and keypoints."""
        start_time = time.time()

        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            return self._create_empty_comprehensive_result(time.time() - start_time)

        image_area = image.shape[0] * image.shape[1]

        # Collect multiple predictions with MC Dropout
        all_scores = []
        all_bboxes = []
        all_bboxes_xyxy = []
        all_keypoints_xy = []
        all_keypoint_scores = []

        for i in range(num_passes):
            self.model.current_img = image_path
            result = self.predict_single_pass(image_path)
            if result is not None:
                all_scores.append(result['score'])
                all_bboxes.append(result['bbox'])
                all_bboxes_xyxy.append(result['bbox_xyxy'])
                all_keypoints_xy.append(result['keypoints_xy'])
                all_keypoint_scores.append(result['keypoint_scores'])
            self.model.last_img = image_path

        # Reset model to eval mode
        self.model.eval()

        if len(all_scores) == 0:
            return self._create_empty_comprehensive_result(time.time() - start_time)

        # Convert to arrays
        scores_array = np.array(all_scores)
        bboxes_array = np.array(all_bboxes)
        bboxes_xyxy_array = np.array(all_bboxes_xyxy)
        keypoints_array = np.array(all_keypoints_xy)
        keypoint_scores_array = np.array(all_keypoint_scores)

        # Calculate mean predictions
        mean_score = np.mean(scores_array)
        mean_bbox = np.mean(bboxes_array, axis=0)
        mean_bbox_xyxy = np.mean(bboxes_xyxy_array, axis=0)
        mean_keypoints = np.mean(keypoints_array, axis=0)
        mean_keypoint_scores = np.mean(keypoint_scores_array, axis=0)

        # 1. Calculate BBOX Uncertainty
        bbox_uncertainty = self._calculate_bbox_uncertainty(bboxes_array, image_area)

        # 2. Calculate KEYPOINT Uncertainty
        keypoints_reshaped = keypoints_array.reshape(len(keypoints_array), self.num_keypoints, 2)
        mean_keypoints_reshaped = mean_keypoints.reshape(self.num_keypoints, 2)

        # Calculate bbox size for normalization
        bbox_width = mean_bbox[2]
        bbox_height = mean_bbox[3]
        bbox_diagonal = np.sqrt(bbox_width ** 2 + bbox_height ** 2)
        bbox_area = bbox_width * bbox_height
        bbox_reference_scale = np.sqrt(bbox_area)

        # Keypoint uncertainty calculations
        keypoint_distances = self._calculate_keypoint_distances(mean_keypoints_reshaped)
        reference_scales = self._calculate_reference_scales(keypoint_distances, bbox_reference_scale)

        # Aleatoric uncertainty
        per_keypoint_aleatoric_uncertainties = []
        for kp_idx in range(self.num_keypoints):
            if len(keypoint_scores_array) > 1:
                kp_score_variance = np.var(keypoint_scores_array[:, kp_idx])
                aleatoric_unc = min(kp_score_variance * 4.0, 1.0)
                per_keypoint_aleatoric_uncertainties.append(aleatoric_unc)
            else:
                per_keypoint_aleatoric_uncertainties.append(1.0)

        overall_aleatoric_uncertainty = np.mean(per_keypoint_aleatoric_uncertainties)

        # Epistemic uncertainty
        keypoint_epistemic_uncertainties = []
        keypoint_determinants = []
        keypoint_spatial_stds = []

        for kp_idx in range(self.num_keypoints):
            kp_coords = keypoints_reshaped[:, kp_idx, :]
            if len(kp_coords) > 1:
                spatial_std_x = np.std(kp_coords[:, 0])
                spatial_std_y = np.std(kp_coords[:, 1])
                spatial_std = np.mean([spatial_std_x, spatial_std_y])
                keypoint_spatial_stds.append(spatial_std)

                reference_scale = reference_scales[kp_idx]
                normalized_spatial_std = spatial_std / reference_scale

                # if bbox_reference_scale < 50:
                #     uncertainty_threshold = 0.30
                # elif bbox_reference_scale < 120:
                #     uncertainty_threshold = 0.20
                # else:
                #     uncertainty_threshold = 0.15

                #epistemic_unc = min(normalized_spatial_std / uncertainty_threshold, 1.0)

                epistemic_unc = min(normalized_spatial_std, 1.0)

                cov_matrix = np.cov(kp_coords.T)
                det = np.linalg.det(cov_matrix)
                normalized_det = det / (bbox_reference_scale ** 4)

                keypoint_epistemic_uncertainties.append(epistemic_unc)
                keypoint_determinants.append(normalized_det)
            else:
                keypoint_spatial_stds.append(0.0)
                keypoint_epistemic_uncertainties.append(1.0)
                keypoint_determinants.append(1.0)

        # Combined per-keypoint uncertainty
        per_keypoint_uncertainty = []
        for kp_idx in range(self.num_keypoints):
            epistemic_unc = keypoint_epistemic_uncertainties[kp_idx]
            aleatoric_unc = per_keypoint_aleatoric_uncertainties[kp_idx]
            kp_uncertainty = min(0.9 * epistemic_unc + 0.1 * aleatoric_unc, 1.0)
            per_keypoint_uncertainty.append(float(kp_uncertainty))

        # Overall uncertainties
        overall_epistemic_uncertainty = np.mean(keypoint_epistemic_uncertainties)
        overall_keypoint_uncertainty = np.mean(per_keypoint_uncertainty)

        # 3. Calculate GLOBAL uncertainty (combined bbox and keypoint)
        global_uncertainty = 0.4 * bbox_uncertainty['overall_uncertainty'] + 0.6 * overall_keypoint_uncertainty

        return {
            # Global uncertainty
            'global_uncertainty': float(global_uncertainty),
            'mean_score': float(mean_score),
            'num_valid_passes': len(all_scores),
            'processing_time': time.time() - start_time,

            # BBOX uncertainty
            'bbox_uncertainty': bbox_uncertainty,
            'mean_bbox': [float(x) for x in mean_bbox],
            'mean_bbox_xyxy': [float(x) for x in mean_bbox_xyxy],

            # Keypoint uncertainty
            'keypoint_uncertainty': {
                'overall_uncertainty': float(overall_keypoint_uncertainty),
                'epistemic_uncertainty': float(overall_epistemic_uncertainty),
                'aleatoric_uncertainty': float(overall_aleatoric_uncertainty),
                'per_keypoint_uncertainty': per_keypoint_uncertainty,
                'per_keypoint_epistemic_uncertainty': [float(x) for x in keypoint_epistemic_uncertainties],
                'per_keypoint_aleatoric_uncertainty': [float(x) for x in per_keypoint_aleatoric_uncertainties],
                'per_keypoint_determinants': [float(x) for x in keypoint_determinants],
                'per_keypoint_spatial_stds': [float(x) for x in keypoint_spatial_stds],
                'per_keypoint_mean_scores': [float(x) for x in mean_keypoint_scores],
            },

            # Reference information
            'reference_scales': [float(x) for x in reference_scales],
            'bbox_size_info': {
                'width': float(bbox_width),
                'height': float(bbox_height),
                'diagonal': float(bbox_diagonal),
                'area': float(bbox_area),
                'reference_scale': float(bbox_reference_scale)
            },

            # Mean predictions
            'mean_keypoints': [float(x) for x in mean_keypoints],
        }

    def _create_empty_comprehensive_result(self, processing_time: float) -> Dict[str, Any]:
        """Create empty result when no detections are found."""
        empty_kp = [1.0] * self.num_keypoints
        empty_scales = [50.0] * self.num_keypoints

        return {
            'global_uncertainty': 1.0,
            'mean_score': 0.0,
            'num_valid_passes': 0,
            'processing_time': processing_time,

            'bbox_uncertainty': {
                'overall_uncertainty': 1.0,
                'position_uncertainty': 1.0,
                'size_uncertainty': 1.0,
                'iou_variance': 1.0,
                'mean_iou': 0.0,
                'area_variance': 1.0,
                'center_std': 100.0,
                'normalized_center_std': 1.0,
                'bbox_variance': [1.0, 1.0, 1.0, 1.0],
                'num_bbox_samples': 0
            },

            'keypoint_uncertainty': {
                'overall_uncertainty': 1.0,
                'epistemic_uncertainty': 1.0,
                'aleatoric_uncertainty': 1.0,
                'per_keypoint_uncertainty': empty_kp,
                'per_keypoint_epistemic_uncertainty': empty_kp,
                'per_keypoint_aleatoric_uncertainty': empty_kp,
                'per_keypoint_determinants': empty_kp,
                'per_keypoint_spatial_stds': empty_kp,
                'per_keypoint_mean_scores': [0.0] * self.num_keypoints,
            },

            'mean_bbox': [0.0, 0.0, 0.0, 0.0],
            'mean_bbox_xyxy': [0.0, 0.0, 0.0, 0.0],
            'mean_keypoints': [0.0] * (self.num_keypoints * 2),
            'reference_scales': empty_scales,
        }

    def _display_comprehensive_uncertainty(self, image_path: str, result: Dict[str, Any], output_path: str):
        """Display comprehensive uncertainty results with safety indicator."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot load image: {image_path}")
            return False

        h, w = image.shape[:2]

        # Global uncertainty indicator
        global_uncertainty = result['global_uncertainty']
        bbox_uncertainty = result['bbox_uncertainty']['overall_uncertainty']

        global_color = (0, int(255 * (1 - global_uncertainty)), int(255 * global_uncertainty))
        bbox_color = (0, int(255 * (1 - bbox_uncertainty)), int(255 * bbox_uncertainty))

        # Safety assessment based on keypoint uncertainties
        per_kp_uncertainty = result['keypoint_uncertainty']['per_keypoint_uncertainty']

        # Check safety condition: keypoints 0, 1, 2 must have uncertainty < 50%
        kp0_safe = per_kp_uncertainty[0] < 0.50
        kp1_safe = per_kp_uncertainty[1] < 0.50
        kp2_safe = per_kp_uncertainty[2] < 0.50

        is_safe_to_cut = kp0_safe == True and kp1_safe == True and kp2_safe == True

        # Safety indicator color
        safety_color = (0, 255, 0) if is_safe_to_cut else (0, 0, 255)  # Green if safe, Red if not
        safety_text = "SAFE TO CUT" if is_safe_to_cut else "NOT SAFE TO CUT"

        # Draw bounding box with bbox uncertainty
        if result['num_valid_passes'] > 0:
            bbox = result['mean_bbox']
            x, y, w_box, h_box = bbox

            # Draw bbox with uncertainty-based color
            cv2.rectangle(image,
                          (int(x), int(y)),
                          (int(x + w_box), int(y + h_box)),
                          bbox_color, 3)

            # Draw bbox uncertainty text
            bbox_unc_text = f'BBox Unc: {bbox_uncertainty:.2f}'
            cv2.putText(image, bbox_unc_text,
                        (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

            # Draw keypoints with individual uncertainty and safety status
            kps = result['mean_keypoints']

            for i in range(0, len(kps), 2):
                if i + 1 < len(kps):
                    kp_idx = i // 2
                    kp_unc = per_kp_uncertainty[kp_idx]

                    # Color coding: green if safe (<30%), yellow if borderline, red if unsafe
                    if kp_unc < 0.30:
                        kp_color = (0, 255, 0)  # Green - safe
                    elif kp_unc < 0.50:
                        kp_color = (0, 255, 255)  # Yellow - borderline
                    else:
                        kp_color = (0, 0, 255)  # Red - unsafe

                    # Highlight keypoint 2 with a different marker or size
                    marker_size = 8 if kp_idx == 2 else 6  # Larger for keypoint 2

                    # Draw keypoint
                    cv2.circle(image, (int(kps[i]), int(kps[i + 1])), marker_size, kp_color, -1)

                    # Draw keypoint number with safety indicator
                    kp_number = f'{kp_idx + 1}'
                    if kp_idx in [0, 1, 2]:  # Critical keypoints
                        kp_number += '*'  # Mark critical keypoints

                    cv2.putText(image, kp_number,
                                (int(kps[i]) - 4, int(kps[i + 1]) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Draw uncertainty value
                    uncertainty_text = f'{kp_unc:.2f}'
                    text_size = cv2.getTextSize(uncertainty_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                    text_x = int(kps[i]) - text_size[0] // 2
                    text_y = int(kps[i + 1]) + 20

                    # Background for text
                    cv2.rectangle(image,
                                  (text_x - 2, text_y - text_size[1] - 2),
                                  (text_x + text_size[0] + 2, text_y + 2),
                                  (255, 255, 255), -1)

                    cv2.putText(image, uncertainty_text,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Display comprehensive info with safety status
        info_lines = [
            f"SAFETY: {safety_text}",
        ]

        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 20
            color = safety_color if "SAFETY:" in line else global_color
            font_scale = 0.6 if "SAFETY:" in line else 0.5
            thickness = 2 if "SAFETY:" in line else 1

            cv2.putText(image, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Add detailed keypoint safety information
        safety_details_y = h - 80
        cv2.putText(image, "Keypoint Safety (Critical: 0,1,2*):",
                    (10, safety_details_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for kp_idx in range(self.num_keypoints):
            kp_unc = per_kp_uncertainty[kp_idx]
            safety_status = "SAFE" if kp_unc < 0.50 else "UNSAFE"
            color = (0, 255, 0) if kp_unc < 0.50 else (0, 0, 255)

            marker = "*" if kp_idx in [0, 1, 2] else " "
            kp_text = f"KP{kp_idx + 1}{marker}: {kp_unc:.2f} ({safety_status})"

            cv2.putText(image, kp_text,
                        (10, safety_details_y + 15 + (kp_idx * 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Save the image
        cv2.imwrite(output_path, image)
        print(f"Saved comprehensive uncertainty visualization to: {output_path}")
        print(f"SAFETY ASSESSMENT: {safety_text}")
        print(f"Keypoint 0: {per_kp_uncertainty[0]:.3f} ({'SAFE' if kp0_safe else 'UNSAFE'})")
        print(f"Keypoint 1: {per_kp_uncertainty[1]:.3f} ({'SAFE' if kp1_safe else 'UNSAFE'})")
        print(f"Keypoint 2: {per_kp_uncertainty[2]:.3f} ({'SAFE' if kp2_safe else 'UNSAFE'})")

        return is_safe_to_cut  # Return the safety status

    def verify_mc_dropout(self, image_path: str, num_passes: int = 10) -> bool:
        """Simple verification that MC dropout is working by checking if we get variation."""
        print("Verifying MC Dropout activation...")

        # Get predictions with dropout
        predictions_with_dropout = []
        bboxes_with_dropout = []
        keypoints_with_dropout = []

        for i in range(num_passes):
            result = self.predict_single_pass(image_path)
            if result is not None:
                predictions_with_dropout.append(result['score'])
                bboxes_with_dropout.append(result['bbox'])
                keypoints_with_dropout.append(result['keypoints_xy'])

        # Get predictions without dropout
        predictions_without_dropout = []
        for i in range(num_passes):
            result = self.predict_single_pass(image_path)
            if result is not None:
                predictions_without_dropout.append(result['score'])

        if len(predictions_with_dropout) > 1:
            var_with_dropout = np.var(predictions_with_dropout)
            bbox_var = np.var(bboxes_with_dropout, axis=0) if len(bboxes_with_dropout) > 1 else 0
            keypoint_var = np.var(keypoints_with_dropout, axis=0) if len(keypoints_with_dropout) > 1 else 0

            print(f"\n  Score variance: {var_with_dropout:.6f}")
            print(f"  BBox variance: {np.mean(bbox_var):.6f}")
            print(f"  Keypoint variance: {np.mean(keypoint_var):.6f}")

            # Check if we have meaningful variance
            has_variance = (var_with_dropout > 1e-6 or
                            np.any(bbox_var > 1e-6) or
                            np.any(keypoint_var > 1e-6))

            if has_variance:
                print("MC Dropout is working!")
                return True
            else:
                print("No variance detected - MC Dropout may not be working")
                return False
        else:
            print("Not enough valid predictions for verification")
            return False


def comprehensive_uncertainty_demo_yolonas(image_path, predictor):
    """Demo function for comprehensive uncertainty calculation with YOLO-NAS."""
    if not os.path.exists(image_path):
        print("Test image not found!")
        return False

    # Calculate comprehensive uncertainty
    result = predictor.calculate_comprehensive_uncertainty(image_path, num_passes=10)

    print(f"\nCOMPREHENSIVE UNCERTAINTY RESULTS:")
    print(f"Global Uncertainty: {result['global_uncertainty']:.4f}")
    print(f"Mean Score: {result['mean_score']:.4f}")
    print(f"Valid MC Passes: {result['num_valid_passes']}")

    print(f"\nBBOX UNCERTAINTY:")
    bbox_unc = result['bbox_uncertainty']
    print(f"   Overall: {bbox_unc['overall_uncertainty']:.4f}")
    print(f"   Position: {bbox_unc['position_uncertainty']:.4f}")
    print(f"   Size: {bbox_unc['size_uncertainty']:.4f}")
    print(f"   IoU Variance: {bbox_unc['iou_variance']:.4f}")
    print(f"   Mean IoU: {bbox_unc['mean_iou']:.4f}")
    print(f"   Center STD: {bbox_unc['center_std']:.2f} px")

    print(f"\nKEYPOINT UNCERTAINTY:")
    kp_unc = result['keypoint_uncertainty']
    print(f"   Overall: {kp_unc['overall_uncertainty']:.4f}")
    print(f"   Epistemic: {kp_unc['epistemic_uncertainty']:.4f}")
    print(f"   Aleatoric: {kp_unc['aleatoric_uncertainty']:.4f}")

    print(f"\nPER-KEYPOINT ANALYSIS:")
    for kp_idx in range(predictor.num_keypoints):
        print(f"  Keypoint {kp_idx + 1}:")
        print(f"    Overall: {kp_unc['per_keypoint_uncertainty'][kp_idx]:.4f}")
        print(f"    Epistemic: {kp_unc['per_keypoint_epistemic_uncertainty'][kp_idx]:.4f}")
        print(f"    Aleatoric: {kp_unc['per_keypoint_aleatoric_uncertainty'][kp_idx]:.4f}")
        print(f"    Mean Score: {kp_unc['per_keypoint_mean_scores'][kp_idx]:.4f}")


    # Create output directories
    os.makedirs('yolonas_output/visualize', exist_ok=True)
    os.makedirs('yolonas_output/json', exist_ok=True)

    image_name = os.path.basename(image_path)
    output_image_path = f'yolonas_output/visualize/{image_name}'

    # Display comprehensive results and get safety status
    is_safe = predictor._display_comprehensive_uncertainty(image_path, result, output_image_path)

    # Save detailed results to JSON
    filename = os.path.splitext(image_name)[0]
    json_output_path = f'yolonas_output/json/{filename}.json'
    with open(json_output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved comprehensive uncertainty results to: {json_output_path}")

    return is_safe

# def test(predictor):
#     print("Testing 3 DIFFERENT images:")
#     image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
#     VAL_IMAGES_DIR = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/test"
#     val_images = []
#     for ext in image_extensions:
#         val_images.extend(glob.glob(os.path.join(VAL_IMAGES_DIR, ext)))
#     val_images.sort()
#
#     for i in range(3):
#         img_path = val_images[i]
#         result = predictor.calculate_comprehensive_uncertainty(img_path, num_passes=10)
#
#         print(f"\nImage {i} ({os.path.basename(img_path)}):")
#         print(f"  BBox: {result['mean_bbox']}")
#         print(f"  First keypoint XY: {result['mean_keypoints']}")


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    VAL_IMAGES_DIR = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/test"
    OUTPUT_DIR = "yolonas_output"
    # Configuration
    CHECKPOINT_DIR = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/CheckPoint"

    # BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "YOLO_NAS_POSE_M/ckpt_best.pth")

    BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "Flip/ckpt_best.pth")
    config = {
        'checkpoint_path': BEST_CKPT_PATH,
        'dropout_rate': 0.02,
        'conf_threshold': 0,
        'num_keypoints': 5,
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }

    print("YOLO-NAS MC Dropout Uncertainty Calculator")
    print("=" * 60)

    # Initialize predictor
    predictor = YOLONASMCPredictor(**config)

    #test(predictor)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all validation images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    val_images = []
    for ext in image_extensions:
        val_images.extend(glob.glob(os.path.join(VAL_IMAGES_DIR, ext)))
    val_images.sort()

    print(f"Found {len(val_images)} validation images")

    safe_count = 0
    total_count = 0

    print(f"Starting YOLO-NAS uncertainty analysis of {len(val_images)} images...")

    for image_path in val_images:
        predictor.model.eval()
        for m in predictor.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
            elif isinstance(m, torch.nn.Dropout2d):
                m.train()
        current_time = time.time()
        image_name = os.path.basename(image_path)
        print(f"\n{'=' * 70}")
        print(f"Processing: {image_name} ({total_count + 1}/{len(val_images)})")
        print(f"{'=' * 70}")

        is_safe = comprehensive_uncertainty_demo_yolonas(image_path, predictor)
        total_count += 1
        print(time.time() - current_time)
        if is_safe:
            safe_count += 1
            print(f"ADDED TO SAFE COUNT: {safe_count}/{total_count}")
        else:
            print(f"NOT SAFE - Safe count remains: {safe_count}/{total_count}")


    # Final summary
    print(f"\n{'=' * 80}")
    print(f"YOLO-NAS FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total images processed: {total_count}")
    print(f"Images SAFE TO CUT: {safe_count}")
    print(f"Images NOT SAFE TO CUT: {total_count - safe_count}")
    print(f"Safety rate: {(safe_count / total_count) * 100:.1f}%")
    print(f"{'=' * 80}")
    #test(predictor)