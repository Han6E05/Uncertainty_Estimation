import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import detectron2.data.detection_utils as utils

def build_train_augmentation(cfg):
    """
    Create a list of training augmentations
    """
    IMAGE_SIZE = (640, 640)  # Adjust based on your needs

    augmentation = [
        # 1. Rotation ±30 degrees - wrap with RandomApply for probability control
        T.RandomApply(T.RandomRotation(angle=[-30, 30], expand=True), prob=0.5),

        # 2. Color jitter (similar to HSV gain)
        T.RandomApply(
            T.AugmentationList([
                T.RandomBrightness(0.15, 0.15),  # brightness
                T.RandomSaturation(0.15, 0.15),  # saturation
                T.RandomContrast(0.15, 0.15),  # contrast
            ]),
            prob=0.5
        ),

        # 3. Additional brightness/contrast
        T.RandomApply(
            T.AugmentationList([
                T.RandomBrightness(0.4, 0.4),
                T.RandomContrast(0.4, 0.4),
            ]),
            prob=0.2
        ),

        # 4. Resize to fixed size
        T.ResizeScale(
            min_scale=0.1, max_scale=2.0, target_height=IMAGE_SIZE[0], target_width=IMAGE_SIZE[1]
        ),
        T.FixedSizeCrop(crop_size=IMAGE_SIZE),

        # Random flip
        #T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    ]

    return augmentation


class CustomDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        # Initialize the parent class
        if is_train:
            # Build augmentations first
            augmentations = build_train_augmentation(cfg)
            super().__init__(cfg, is_train=is_train, augmentations=augmentations)
        else:
            super().__init__(cfg, is_train=is_train)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = CustomDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


# Your existing dataset registration code remains the same
register_coco_instances(
    "train", {},
    "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/annotations/tomato_keypoints_train1.json",
    "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/train"
)

register_coco_instances(
    "val", {},
    "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/annotations/tomato_keypoints_val1.json",
    "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/val"
)

# Define keypoints (your existing code)
keypoints = [
    "Cutting_top", "Cutting_bottom", "Cutting_point",
    "Peduncle_futher", "Peduncle_last"
]

MetadataCatalog.get("train").keypoint_flip_map = [
    ("Cutting_top", "Cutting_top"),
    ("Cutting_bottom", "Cutting_bottom"),
    ("Cutting_point", "Cutting_point"),
    ("Peduncle_futher", "Peduncle_futher"),
    ("Peduncle_last", "Peduncle_last")
]

MetadataCatalog.get("val").keypoint_flip_map = [
    ("Cutting_top", "Cutting_top"),
    ("Cutting_bottom", "Cutting_bottom"),
    ("Cutting_point", "Cutting_point"),
    ("Peduncle_futher", "Peduncle_futher"),
    ("Peduncle_last", "Peduncle_last")
]

MetadataCatalog.get("train").keypoint_names = keypoints
MetadataCatalog.get("val").keypoint_names = keypoints

# Load and customize config
cfg = get_cfg()
cfg.merge_from_file("my_detectron2/configs/COCO-Keypoints/my_keypoint-rcnn.yaml")
cfg.DATASETS.TRAIN = ["train"]
cfg.DATASETS.TEST = ["val"]
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 10000
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
cfg.OUTPUT_DIR = "output"
cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.035, 0.035, 0.035, 0.15, 0.15]

# Non-maximum suppression settings
cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.5  # Standard NMS threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # During inference

# Image size configuration (used in augmentations)
cfg.INPUT.MIN_SIZE_TRAIN = (640,)
cfg.INPUT.MAX_SIZE_TRAIN = 640
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 640

# ========== LOSS WEIGHT CONFIGURATION ==========
# Increase keypoint loss weight to emphasize keypoint detection
cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 10.0  # Default is 1.0

# Create output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Start training
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

#
# #  Register your tomato dataset
# register_coco_instances(
#     "train", {},
#     "dataset/annotations/tomato_keypoints_train1.json",
#     "dataset/images/train"
# )
#
# register_coco_instances(
#     "val", {},
#     "dataset/annotations/tomato_keypoints_val1.json",
#     "dataset/images/val"
# )
#
# keypoints = [
#     "Cutting_top",
#     "Cutting_bottom",
#     "Cutting_point",
#     "Peduncle_futher",
#     "Peduncle_last"
# ]
# MetadataCatalog.get("train").keypoint_flip_map = [
#     ("Cutting_top", "Cutting_top"),
#     ("Cutting_bottom", "Cutting_bottom"),
#     ("Cutting_point", "Cutting_point"),
#     ("Peduncle_futher", "Peduncle_futher"),
#     ("Peduncle_last", "Peduncle_last")
# ]
#
# MetadataCatalog.get("val").keypoint_flip_map = [
#     ("Cutting_top", "Cutting_top"),
#     ("Cutting_bottom", "Cutting_bottom"),
#     ("Cutting_point", "Cutting_point"),
#     ("Peduncle_futher", "Peduncle_futher"),
#     ("Peduncle_last", "Peduncle_last")
# ]
#
# MetadataCatalog.get("train").keypoint_names = keypoints
# MetadataCatalog.get("val").keypoint_names = keypoints
#
# # ️ Load and customize config
# cfg = get_cfg()
# cfg.merge_from_file("my_detectron2/configs/COCO-Keypoints/my_keypoint-rcnn.yaml")
# cfg.DATASETS.TRAIN = ["train"]
# cfg.DATASETS.TEST = ["val"]
# cfg.DATALOADER.NUM_WORKERS = 2
#
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.001
# cfg.SOLVER.MAX_ITER = 10000
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5  # Adjust to your keypoints
# cfg.OUTPUT_DIR = "output"
# cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.035, 0.035, 0.035, 0.15, 0.15]
#
# #  Add dropout configuration
# #cfg.MODEL.ROI_KEYPOINT_HEAD.DROPOUT_RATE = 0.10
#
# #  Create output directory
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#
# #  Start training
# class MyTrainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         return COCOEvaluator(dataset_name, cfg, True, output_folder)
#
# trainer = MyTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
# #trainer.test()