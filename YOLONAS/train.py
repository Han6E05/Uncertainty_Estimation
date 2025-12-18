import json
import os
from typing import List, Union, Tuple

import torch
from super_gradients.training import Trainer
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform
from super_gradients.training.samples import PoseEstimationSample

from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import AbstractPoseEstimationDataset

from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN



def open_file(file_path: str) -> Union[dict, list, None]:
    """
    Opens and reads the content of a JSON or YAML file.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    Union[dict, list, None]: The content of the file parsed to a dictionary or a list,
                             or None if an error occurs.
    """
    with open(file_path, "r") as file:
        if file_path.endswith(".json"):
            return json.load(file)
        elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
            return yaml.safe_load(file)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

class TomatoPoseEstimationDataset(AbstractPoseEstimationDataset):
    @classmethod
    def split_tomato_pose_dataset(cls, annotation_file: str, train_annotation_file: str, val_annotation_file: str,
                                  val_fraction: float):
        with open(annotation_file, "r") as f:
            annotation = json.load(f)

        image_ids = [img["id"] for img in annotation["images"]]
        image_id_to_image = {img["id"]: img for img in annotation["images"]}

        labels = [[ann["category_id"] for ann in annotation["annotations"] if ann["image_id"] == image_id] for image_id
                  in image_ids]
        labels = [label[0] if len(label) else -1 for label in labels]

        train_ids, val_ids = train_test_split(image_ids, test_size=val_fraction, random_state=42, stratify=labels)

        train_annotations = {
            "info": annotation.get("info", {}),
            "categories": annotation["categories"],
            "images": [image_id_to_image[image_id] for image_id in train_ids],
            "annotations": [ann for ann in annotation["annotations"] if ann["image_id"] in train_ids],
        }

        val_annotations = {
            "info": annotation.get("info", {}),
            "categories": annotation["categories"],
            "images": [image_id_to_image[image_id] for image_id in val_ids],
            "annotations": [ann for ann in annotation["annotations"] if ann["image_id"] in val_ids],
        }

        with open(train_annotation_file, "w") as f:
            json.dump(train_annotations, f)
            print("Train annotations saved to", train_annotation_file)
            print("Train images:", len(train_ids))
            print("Train annotations:", len(train_annotations["annotations"]))

        with open(val_annotation_file, "w") as f:
            json.dump(val_annotations, f)
            print("Val annotations saved to", val_annotation_file)
            print("Val images:", len(val_ids))
            print("Val annotations:", len(val_annotations["annotations"]))

    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        json_file: str,
        transforms: List[AbstractKeypointTransform],
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
    ):
        split_json_file = os.path.join(data_dir, json_file)
        with open(split_json_file, "r") as f:
            json_annotations = json.load(f)

        joints = json_annotations["categories"][0]["keypoints"]
        num_joints = len(joints)

        super().__init__(
            transforms=transforms,
            num_joints=num_joints,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )

        self.num_joints = num_joints

        images_and_ids = [(img["id"], os.path.join(data_dir, images_dir, img["file_name"])) for img in
                          json_annotations["images"]]
        self.image_ids, self.image_files = zip(*images_and_ids)

        self.annotations = []

        for image_id in self.image_ids:
            keypoints_per_image = []
            bboxes_per_image = []

            image_annotations = [ann for ann in json_annotations["annotations"] if str(ann["image_id"]) == str(image_id)]
            for ann in image_annotations:
                keypoints = np.array(ann["keypoints"]).reshape(self.num_joints, 3)
                x, y, w, h = ann["bbox"]

                bbox_xywh = np.array([x, y, w, h])
                keypoints_per_image.append(keypoints)
                bboxes_per_image.append(bbox_xywh)

            keypoints_per_image = np.array(keypoints_per_image, dtype=np.float32).reshape(-1, self.num_joints, 3)
            bboxes_per_image = np.array(bboxes_per_image, dtype=np.float32).reshape(-1, 4)
            annotation = keypoints_per_image, bboxes_per_image
            self.annotations.append(annotation)

    def __len__(self):
        return len(self.image_ids)

    def load_sample(self, index) -> PoseEstimationSample:
        file_path = self.image_files[index]
        gt_joints, gt_bboxes = self.annotations[index]  # boxes in xywh format

        gt_areas = np.array([box[2] * box[3] for box in gt_bboxes], dtype=np.float32)
        gt_iscrowd = np.array([0] * len(gt_joints), dtype=bool)

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        mask = np.ones(image.shape[:2], dtype=np.float32)

        return PoseEstimationSample(
            image=image, mask=mask, joints=gt_joints, areas=gt_areas, bboxes_xywh=gt_bboxes, is_crowd=gt_iscrowd, additional_samples=None
        )

KEYPOINT_NAMES = [
    "Cutting_top",
    "Cutting_bottom",
    "Cutting_point",
    "Peduncle_futher",
    "Peduncle_last"
]

NUM_JOINTS = len(KEYPOINT_NAMES)

KEYPOINT_COLORS = [[ 0, 102, 204 ], [ 204, 0, 102 ], [ 255, 204, 0 ], [ 255, 102, 0 ], [ 0, 153, 76 ]]

OKS_SIGMAS = [0.035, 0.035, 0.035, 0.15, 0.15]

EDGE_LINKS = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]]
EDGE_COLORS = [[ 51, 153, 255 ], [ 255, 51, 153 ], [ 153, 51, 255 ], [ 255, 153, 51 ], [ 51, 255, 153 ]]
FLIP_INDEXES = [0, 1, 2, 3, 4]

IMAGE_SIZE = (640, 640)

def main():
    # Import transforms
    from super_gradients.training.transforms.keypoints import (
        KeypointsRandomHorizontalFlip,
        KeypointsHSV,
        KeypointsBrightnessContrast,
        KeypointsMosaic,
        KeypointsRandomAffineTransform,
        KeypointsLongestMaxSize,
        KeypointsPadIfNeeded,
        KeypointsImageStandardize,
        KeypointsRemoveSmallObjects,
    )

    # Define transforms
    keypoints_random_horizontal_flip = KeypointsRandomHorizontalFlip(flip_index=FLIP_INDEXES, prob=0.5)
    keypoints_hsv = KeypointsHSV(prob=0.7, hgain=15, sgain=15, vgain=15)
    keypoints_brightness_contrast = KeypointsBrightnessContrast(prob=0.3, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4))
    keypoints_mosaic = KeypointsMosaic(prob=0.3)
    keypoints_random_affine_transform = KeypointsRandomAffineTransform(
        max_rotation=30, min_scale=0.5, max_scale=1.5, max_translate=0.15, image_pad_value=127, mask_pad_value=1, prob=0.75, interpolation_mode=[0, 1, 2, 3, 4]
    )
    keypoints_longest_max_size = KeypointsLongestMaxSize(max_height=IMAGE_SIZE[0], max_width=IMAGE_SIZE[1])
    keypoints_pad_if_needed = KeypointsPadIfNeeded(
        min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], image_pad_value=127, mask_pad_value=1, padding_mode="bottom_right"
    )
    keypoints_image_standardize = KeypointsImageStandardize(max_value=255)

    train_transforms = [
        keypoints_random_horizontal_flip,
        keypoints_hsv,
        keypoints_brightness_contrast,
        keypoints_random_affine_transform,
        keypoints_longest_max_size,
        keypoints_pad_if_needed,
        keypoints_image_standardize,
    ]

    val_transforms = [
        keypoints_longest_max_size,
        keypoints_pad_if_needed,
        keypoints_image_standardize,
    ]

    # Create datasets
    train_dataset = TomatoPoseEstimationDataset(
        data_dir="/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset",
        images_dir="images/train",
        json_file="annotations/tomato_keypoints_train1.json",
        transforms=train_transforms,
        edge_links=EDGE_LINKS,
        edge_colors=EDGE_COLORS,
        keypoint_colors=KEYPOINT_COLORS,
    )

    val_dataset = TomatoPoseEstimationDataset(
        data_dir="/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset",
        images_dir="images/val",
        json_file="annotations/tomato_keypoints_val1.json",
        transforms=val_transforms,
        edge_links=EDGE_LINKS,
        edge_colors=EDGE_COLORS,
        keypoint_colors=KEYPOINT_COLORS,
    )

    test_dataset = TomatoPoseEstimationDataset(
        data_dir="/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset",
        images_dir="images/test",
        json_file="annotations/tomato_keypoints_test1.json",
        transforms=val_transforms,
        edge_links=EDGE_LINKS,
        edge_colors=EDGE_COLORS,
        keypoint_colors=KEYPOINT_COLORS,
    )

    from torch.utils.data import DataLoader

    # Create dataloaders
    train_dataloader_params = {"shuffle": True, "batch_size": 12, "drop_last": True, "pin_memory": False,
                               "collate_fn": YoloNASPoseCollateFN()}

    val_dataloader_params = {"shuffle": True, "batch_size": 12, "drop_last": True, "pin_memory": False,
                             "collate_fn": YoloNASPoseCollateFN()}

    train_dataloader = DataLoader(train_dataset, **train_dataloader_params)
    val_dataloader = DataLoader(val_dataset, **val_dataloader_params)
    test_dataloader = DataLoader(test_dataset, **val_dataloader_params)

    # Load model
    #yolo_nas_pose = models.get(Models.YOLO_NAS_POSE_M, num_classes=NUM_JOINTS, pretrained_weights="coco_pose").cuda()
    yolo_nas_pose = models.get(Models.YOLO_NAS_POSE_L, num_classes=NUM_JOINTS, pretrained_weights="coco_pose").cuda()

    # Training setup
    from super_gradients.training.models.pose_estimation_models.yolo_nas_pose import YoloNASPosePostPredictionCallback
    from super_gradients.training.utils.callbacks import ExtremeBatchPoseEstimationVisualizationCallback, Phase
    from super_gradients.training.utils.early_stopping import EarlyStop
    from super_gradients.training.metrics import PoseEstimationMetrics

    post_prediction_callback = YoloNASPosePostPredictionCallback(
        pose_confidence_threshold=0.01,
        nms_iou_threshold=0.7,
        pre_nms_max_predictions=300,
        post_nms_max_predictions=10,
    )

    metrics = PoseEstimationMetrics(
        num_joints=NUM_JOINTS,
        oks_sigmas=OKS_SIGMAS,
        max_objects_per_image=5,
        post_prediction_callback=post_prediction_callback,
    )

    visualization_callback = ExtremeBatchPoseEstimationVisualizationCallback(
        keypoint_colors=KEYPOINT_COLORS,
        edge_colors=EDGE_COLORS,
        edge_links=EDGE_LINKS,
        loss_to_monitor="YoloNASPoseLoss/loss",
        max=True,
        freq=1,
        max_images=16,
        enable_on_train_loader=True,
        enable_on_valid_loader=True,
        post_prediction_callback=post_prediction_callback,
    )

    early_stop = EarlyStop(
        phase=Phase.VALIDATION_EPOCH_END,
        monitor="AP",
        mode="max",
        min_delta=0.0001,
        patience=100,
        verbose=True,
    )

    train_params = {
        "warmup_mode": "LinearBatchLRWarmup",
        "warmup_initial_lr": 1e-4,
        "lr_warmup_epochs": 2,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.05,
        "max_epochs": 60,
        "zero_weight_decay_on_bias_and_bn": True,
        "batch_accumulate": 1,
        "average_best_models": False,
        "save_ckpt_epoch_list": [],
        "loss": "yolo_nas_pose_loss",
        "criterion_params": {
            "oks_sigmas": OKS_SIGMAS,
            "classification_loss_weight": 1.0,
            "classification_loss_type": "focal",
            "regression_iou_loss_type": "ciou",
            "iou_loss_weight": 2.5,
            "dfl_loss_weight": 0.01,
            "pose_cls_loss_weight": 1.0,
            "pose_reg_loss_weight": 34.0,
            "pose_classification_loss_type": "focal",
            "rescale_pose_loss_with_assigned_score": True,
            "assigner_multiply_by_pose_oks": True,
        },
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 0.0001},
        "ema": True,
        "ema_params": {"decay": 0.997, "decay_type": "threshold"},
        "mixed_precision": True,
        "sync_bn": False,
        "valid_metrics_list": [metrics],
        "phase_callbacks": [visualization_callback, early_stop],
        "pre_prediction_callback": None,
        "metric_to_watch": "AP",
        "greater_metric_to_watch_is_better": True,
    }

    # Test if data is correct
    sample = train_dataset[0]
    print("Image shape:", sample.image.shape)
    print("Joints:", sample.joints)
    print("BBoxes:", sample.bboxes_xywh)

    CHECKPOINT_DIR = "checkpoints"
    trainer = Trainer(experiment_name="Tomato_pose", ckpt_root_dir=CHECKPOINT_DIR)

    # Train the model
    trainer.train(model=yolo_nas_pose, training_params=train_params, train_loader=train_dataloader, valid_loader=val_dataloader)

    # best_model = models.get("yolo_nas_pose_m", num_classes=NUM_JOINTS,
    #                         checkpoint_path=os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth"))
    #
    # print(trainer.test(model=best_model, test_loader=test_dataloader, test_metrics_list=metrics))

if __name__ == "__main__":
    main()
