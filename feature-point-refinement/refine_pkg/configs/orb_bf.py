from yacs.config import CfgNode as CN
_CN = CN()

##############  Dataset  ##############
_CN.DATASET = CN()
# train
TRAIN_BASE_PATH = "data/megadepth/index"
_CN.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = "data/megadepth/train"
_CN.DATASET.TRAIN_POSE_ROOT = None  
_CN.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
_CN.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/trainvaltest_list/train_list.txt"
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
# val
VAL_BASE_PATH = TEST_BASE_PATH = "data/megadepth/index"
_CN.DATASET.TEST_DATA_SOURCE = "MegaDepth"

_CN.DATASET.VAL_DATA_ROOT = "data/megadepth/test"
_CN.DATASET.VAL_POSE_ROOT = None  
_CN.DATASET.VAL_NPZ_ROOT = f"{VAL_BASE_PATH}/scene_info_val_1500"
_CN.DATASET.VAL_LIST_PATH = f"{VAL_BASE_PATH}/trainvaltest_list/val_list.txt"
_CN.DATASET.VAL_INTRINSIC_PATH = None
# test
_CN.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
_CN.DATASET.TEST_POSE_ROOT = None  
_CN.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500"
_CN.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/val_list.txt"
_CN.DATASET.TEST_INTRINSIC_PATH = None

_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# 2. dataset config
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']
# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 1200
_CN.DATASET.MGDPT_IMG_PAD = True
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = None

##############  Model  ##############
_CN.MODEL = CN()
_CN.MODEL.INPUT_CHANNELS = 1
_CN.MODEL.ERR_THRESHOLD = 8 # 仅训练网络调整 ERR_THRESHOLD 个像素
_CN.MODEL.PATCH_SIZE = 33 # 实际给网络多少的感受野去调整特征点
_CN.MODEL.N_FEATURES = 2048 # 一个图像上检测多少个特征点
_CN.MODEL.HUBER_DELTA = 4 # 对应到向量的模长, 0.5 = 4像素 / 8像素
_CN.MODEL.MATCHER = "orb_bf"

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.EXP_NAME = "orb_bf"
_CN.TRAINER.EPOCHS = 30
_CN.TRAINER.BATCH_SIZE = 4
_CN.TRAINER.LR = 3e-4
_CN.TRAINER.NUM_WORKERS = 4

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 100
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# reproducibility
_CN.TRAINER.SEED = 66

def get_cfg():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()