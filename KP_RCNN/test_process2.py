import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from typing import List, Dict, Any, Tuple
import time
import torch.nn.functional as F

class InstantMCPredictor:
    """Instant MC Dropout predictor with per-keypoint AND bbox uncertainty estimation."""

    def __init__(self, config_path: str, weights_path: str,
                 dropout_rate: float = 0.10, score_threshold: float = 0,
                 num_keypoints: int = 5, device: str = "cuda"):

        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 1  # Only best detection

        # Build and wrap model with MC Dropout
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(weights_path)

        # Store original model for reference
        self.original_model = self.model
        self.dropout_rate = dropout_rate
        self.model = self._wrap_with_mc_dropout(self.model, dropout_rate)

        self.score_threshold = score_threshold
        self.num_keypoints = num_keypoints
        self.device = device
        print(f" Instant MC Predictor ready on {device}")

    def _wrap_with_mc_dropout(self, model, dropout_rate):
        """Strategic MC Dropout wrapper - focused on backbone, neck, bbox and keypoint prediction heads."""
        import copy
        model = copy.deepcopy(model)

        #Apply dropout to backbone
        #Add manually in fpn


        if hasattr(model, 'roi_heads'):
            roi_heads = model.roi_heads
            #add to ROI head == Neck
            class BoxPoolerWithDropout(nn.Module):
                def __init__(self, pooler, dropout_rate=dropout_rate):
                    super().__init__()
                    self.pooler = pooler
                    self.dropout = nn.Dropout2d(p=dropout_rate)

                def forward(self, features, boxes):
                    pooled = self.pooler(features, boxes)
                    return self.dropout(pooled)

            if hasattr(roi_heads, 'box_pooler'):
                roi_heads.box_pooler = BoxPoolerWithDropout(roi_heads.box_pooler, dropout_rate=dropout_rate)

            class KeypointPoolerWithDropout(nn.Module):
                def __init__(self, pooler, dropout_rate=dropout_rate):
                    super().__init__()
                    self.pooler = pooler
                    self.dropout = nn.Dropout2d(p=dropout_rate)

                def forward(self, features, boxes):
                    pooled = self.pooler(features, boxes)
                    return self.dropout(pooled)

            if hasattr(roi_heads, 'keypoint_pooler'):
                roi_heads.keypoint_pooler = KeypointPoolerWithDropout(roi_heads.keypoint_pooler, dropout_rate=dropout_rate)

            # # 1. Apply dropout to bbox head
            # if hasattr(roi_heads, 'box_head') and hasattr(roi_heads, 'box_predictor'):
            #     print(f" Applying MC Dropout to BBox Head...")
            #
            #     # Add dropout before the final bbox prediction layers
            #     box_predictor = roi_heads.box_predictor
            #     if hasattr(box_predictor, 'cls_score'):
            #         # Wrap the classification layer
            #         original_cls_score = box_predictor.cls_score
            #         box_predictor.cls_score = nn.Sequential(
            #             nn.Dropout(dropout_rate),
            #             original_cls_score
            #         )
            #         print(f"   Added Dropout to bbox cls_score (rate={dropout_rate})")
            #
            #     if hasattr(box_predictor, 'bbox_pred'):
            #         # Wrap the bbox regression layer
            #         original_bbox_pred = box_predictor.bbox_pred
            #         box_predictor.bbox_pred = nn.Sequential(
            #             nn.Dropout(dropout_rate),
            #             original_bbox_pred
            #         )
            #         print(f"   Added Dropout to bbox_pred (rate={dropout_rate})")
            #
            # # 2. Apply dropout to keypoint head
            # if hasattr(roi_heads, 'keypoint_head'):
            #     keypoint_head = roi_heads.keypoint_head
            #     print(f" Applying Strategic MC Dropout to Keypoint Head...")
            #
            #     # STRATEGIC CHOICE 1: Add dropout right before the final score_lowres layer
            #     if hasattr(keypoint_head, 'score_lowres'):
            #         original_score_lowres = keypoint_head.score_lowres
            #         keypoint_head.score_lowres = nn.Sequential(
            #             nn.Dropout2d(dropout_rate),
            #             original_score_lowres
            #         )
            #         print(f"   Added Dropout2d (rate={dropout_rate}) before score_lowres")

        return model

    def _enable_dropout(self):
        """Activate only the dropout layers we intentionally added."""
        dropout_count = 0

        def set_dropout_mode(module):
            nonlocal dropout_count
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()  # Keep dropout active in eval mode
                dropout_count += 1

        self.model.apply(set_dropout_mode)

    def predict_single_pass(self, image: np.ndarray, use_dropout: bool = False) -> Dict[str, Any]:
        """Single forward pass prediction with optional dropout."""
        if use_dropout:
            self._enable_dropout()
        else:
            # Ensure all dropout layers are disabled
            self.model.eval()

        with torch.no_grad():
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
                image_tensor = torch.as_tensor(
                    np.ascontiguousarray(image.transpose(2, 0, 1)),
                    device=self.device
                ).float()
                inputs = {"image": image_tensor, "height": height, "width": width}
            else:
                inputs = image

            predictions = self.model([inputs])
            instances = predictions[0]["instances"]

            if len(instances) == 0:
                return None

            # Get best detection
            best_idx = torch.argmax(instances.scores).item()
            instance = instances[best_idx]

            # Extract predictions
            score = float(instance.scores.cpu())
            if score < self.score_threshold:
                return None

            bbox = instance.pred_boxes.tensor.cpu().numpy()[0]
            keypoints = instance.pred_keypoints.cpu().numpy()[0]

            # Extract keypoint coordinates and scores
            keypoints_xy = keypoints[:, :2]
            keypoint_scores = keypoints[:, 2]

            # Convert bbox to xywh
            bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            return {
                'score': score,
                'bbox': bbox_xywh,
                'bbox_xyxy': bbox.tolist(),  # Keep original format for variance calculation
                'keypoints': keypoints,
                'keypoints_xy': keypoints_xy.flatten(),
                'keypoint_scores': keypoint_scores,
                'keypoints_full': keypoints
            }

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

    def calculate_comprehensive_uncertainty(self, image: np.ndarray, num_passes: int = 30) -> Dict[str, Any]:
        """
        Calculate comprehensive uncertainty for both bbox and keypoints.
        """
        start_time = time.time()
        image_area = image.shape[0] * image.shape[1]

        # Collect multiple predictions with MC Dropout
        all_scores = []
        all_bboxes = []
        all_bboxes_xyxy = []
        all_keypoints_xy = []
        all_keypoint_scores = []

        for i in range(num_passes):
            result = self.predict_single_pass(image, use_dropout=True)
            if result is not None:
                all_scores.append(result['score'])
                all_bboxes.append(result['bbox'])
                all_bboxes_xyxy.append(result['bbox_xyxy'])
                all_keypoints_xy.append(result['keypoints_xy'])
                all_keypoint_scores.append(result['keypoint_scores'])

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

        # 2. Calculate KEYPOINT Uncertainty (existing logic)
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

                if bbox_reference_scale < 50:
                    uncertainty_threshold = 0.30
                elif bbox_reference_scale < 120:
                    uncertainty_threshold = 0.28
                else:
                    uncertainty_threshold = 0.25

                epistemic_unc = min(normalized_spatial_std / uncertainty_threshold, 1.0)
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
            kp_uncertainty = 0.9 * epistemic_unc + 0.1 * aleatoric_unc
            #kp_uncertainty = min(0.8 * epistemic_unc + 1 * aleatoric_unc, 1)

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

    def _display_comprehensive_uncertainty(self, image: np.ndarray, result: Dict[str, Any], image_name: str):
        """Display comprehensive uncertainty results with safety indicator for cutting."""
        h, w = image.shape[:2]

        # Global uncertainty indicator
        global_uncertainty = result['global_uncertainty']
        bbox_uncertainty = result['bbox_uncertainty']['overall_uncertainty']

        global_color = (0, int(255 * (1 - global_uncertainty)), int(255 * global_uncertainty))
        bbox_color = (0, int(255 * (1 - bbox_uncertainty)), int(255 * bbox_uncertainty))

        # Safety assessment based on keypoint uncertainties
        per_kp_uncertainty = result['keypoint_uncertainty']['per_keypoint_uncertainty']

        # Check safety condition: keypoints 0, 1, 2 must have uncertainty < 30%
        # Keypoint 2 has higher importance (you can adjust weights if needed)
        kp0_safe = per_kp_uncertainty[0] < 0.50
        kp1_safe = per_kp_uncertainty[1] < 0.50
        kp2_safe = per_kp_uncertainty[2] < 0.50  # Higher importance

        overall_safe = 0.20 * per_kp_uncertainty[0] + 0.20 * per_kp_uncertainty[1] + 0.60 * per_kp_uncertainty[2]
        is_safe_to_cut = overall_safe < 0.5

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
        output_path = f'output/visualize/{image_name}'
        cv2.imwrite(output_path, image)
        print(f" Saved comprehensive uncertainty visualization to: {output_path}")
        print(f"   SAFETY ASSESSMENT: {safety_text}")
        print(f"   Keypoint 0: {per_kp_uncertainty[0]:.3f} ({'SAFE' if kp0_safe else 'UNSAFE'})")
        print(f"   Keypoint 1: {per_kp_uncertainty[1]:.3f} ({'SAFE' if kp1_safe else 'UNSAFE'})")
        print(f"   Keypoint 2: {per_kp_uncertainty[2]:.3f} ({'SAFE' if kp2_safe else 'UNSAFE'})")

        return is_safe_to_cut  # Return the safety status

    def verify_mc_dropout(self, image: np.ndarray, num_passes: int = 30) -> bool:
        """Simple verification that MC dropout is working by checking if we get variation."""
        print(" Verifying MC Dropout activation...")

        # Get predictions with dropout
        predictions_with_dropout = []
        bboxes_with_dropout = []

        for i in range(num_passes):
            result = self.predict_single_pass(image, use_dropout=True)
            if result is not None:
                predictions_with_dropout.append(result['score'])
                bboxes_with_dropout.append(result['bbox'])

        # Get predictions without dropout
        predictions_without_dropout = []
        for i in range(num_passes):
            result = self.predict_single_pass(image, use_dropout=False)
            if result is not None:
                predictions_without_dropout.append(result['score'])

        if len(predictions_with_dropout) > 1:
            var_with_dropout = np.var(predictions_with_dropout)
            bbox_var = np.var(bboxes_with_dropout, axis=0) if len(bboxes_with_dropout) > 1 else 0

            # If we have variance with dropout, it's working
            if var_with_dropout > 0 or np.any(bbox_var > 0):
                return True
            else:
                print(" No variance with dropout - MC Dropout may not be working")
                return False
        else:
            print(" Not enough valid predictions for verification")
            return False


def comprehensive_uncertainty_demo(image_path):
    """Demo function for comprehensive uncertainty calculation (bbox + keypoints)."""
    # Configuration
    config = {
        'config_path': "my_detectron2/configs/COCO-Keypoints/my_keypoint-rcnn.yaml",
        'weights_path': "output/model_final.pth",
        'dropout_rate': 0.02,
        'score_threshold': 0.0,
        'num_keypoints': 5,
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }

    print(" Comprehensive MC Dropout Uncertainty Calculator")
    print("=" * 60)

    # Initialize predictor
    predictor = InstantMCPredictor(**config)

    if not os.path.exists(image_path):
        print(" Test image not found!")
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(" Cannot load image!")
        return False

    # Simple verification
    print("\n Verifying MC Dropout...")
    if predictor.verify_mc_dropout(image):
        print(" Proceeding with comprehensive uncertainty calculation...")

        # Calculate comprehensive uncertainty
        result = predictor.calculate_comprehensive_uncertainty(image, num_passes=10)

        print(f"\n COMPREHENSIVE UNCERTAINTY RESULTS:")
        print(f" Global Uncertainty: {result['global_uncertainty']:.4f}")
        print(f" Mean Score: {result['mean_score']:.4f}")
        print(f" Valid MC Passes: {result['num_valid_passes']}")

        print(f"\n BBOX UNCERTAINTY:")
        bbox_unc = result['bbox_uncertainty']
        print(f"   Overall: {bbox_unc['overall_uncertainty']:.4f}")
        print(f"   Position: {bbox_unc['position_uncertainty']:.4f}")
        print(f"   Size: {bbox_unc['size_uncertainty']:.4f}")
        print(f"   IoU Variance: {bbox_unc['iou_variance']:.4f}")
        print(f"   Mean IoU: {bbox_unc['mean_iou']:.4f}")
        print(f"   Center STD: {bbox_unc['center_std']:.2f} px")

        print(f"\n KEYPOINT UNCERTAINTY:")
        kp_unc = result['keypoint_uncertainty']
        print(f"   Overall: {kp_unc['overall_uncertainty']:.4f}")
        print(f"   Epistemic: {kp_unc['epistemic_uncertainty']:.4f}")
        print(f"   Aleatoric: {kp_unc['aleatoric_uncertainty']:.4f}")

        print(f"\n PER-KEYPOINT ANALYSIS:")
        for kp_idx in range(predictor.num_keypoints):
            print(f"  Keypoint {kp_idx + 1}:")
            print(f"    Overall: {kp_unc['per_keypoint_uncertainty'][kp_idx]:.4f}")
            print(f"    Epistemic: {kp_unc['per_keypoint_epistemic_uncertainty'][kp_idx]:.4f}")
            print(f"    Aleatoric: {kp_unc['per_keypoint_aleatoric_uncertainty'][kp_idx]:.4f}")
            print(f"    Mean Score: {kp_unc['per_keypoint_mean_scores'][kp_idx]:.4f}")


        print(f"Processing_Time: {result['processing_time']}")
        image_name = os.path.basename(image_path)
        # Display comprehensive results and get safety status
        is_safe = predictor._display_comprehensive_uncertainty(image, result, image_name)

        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Create folder if it doesn't exist
        output_dir = 'output/json'
        os.makedirs(output_dir, exist_ok=True)

        json_output_path = f'{output_dir}/{filename}.json'

        with open(json_output_path, 'w') as f:
            json.dump(result, f, indent=2)

        return is_safe

    else:
        print(" MC Dropout not working effectively.")
        return False


if __name__ == "__main__":
    images_path = '/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/test'
    safe_count = 0
    total_count = 0

    # Get list of image files
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    total_images = len(image_files)

    print(f" Starting analysis of {total_images} images...")

    for image in image_files:
        if image.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"\n{'=' * 70}")
            print(f" Processing: {image} ({total_count + 1}/{total_images})")
            print(f"{'=' * 70}")
            is_safe = comprehensive_uncertainty_demo(os.path.join(images_path, image))
            total_count += 1
            if is_safe:
                safe_count += 1
                print(f" ADDED TO SAFE COUNT: {safe_count}/{total_count}")
            else:
                print(f" NOT SAFE - Safe count remains: {safe_count}/{total_count}")

    # Final summary
    print(f"\n{'=' * 80}")
    print(f" FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f" Total images processed: {total_count}")
    print(f" Images SAFE TO CUT: {safe_count}")
    print(f" Images NOT SAFE TO CUT: {total_count - safe_count}")
    print(f" Safety rate: {(safe_count / total_count) * 100:.1f}%")
    print(f"{'=' * 80}")