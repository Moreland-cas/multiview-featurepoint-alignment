from .orb_bf import get_cfg as get_orb_bf_cfg
from .sift_bf import get_cfg as get_sift_bf_cfg
from .akaze_bf import get_cfg as get_akaze_bf_cfg
from .brisk_bf import get_cfg as get_brisk_bf_cfg
from .sp_sg import get_cfg as get_sp_sg_cfg
from .sp_lg import get_cfg as get_sp_lg_cfg
from .disk_lg import get_cfg as get_disk_lg_cfg
from .sift_lg import get_cfg as get_sift_lg_cfg
from .dog_lg import get_cfg as get_dog_lg_cfg
from .aliked_lg import get_cfg as get_aliked_lg_cfg

def get_cfg(model_choice):
    if model_choice == "orb_bf":
        return get_orb_bf_cfg()
    if model_choice == "sift_bf":
        return get_sift_bf_cfg()
    if model_choice == "akaze_bf":
        return get_akaze_bf_cfg()
    if model_choice == "brisk_bf":
        return get_brisk_bf_cfg()
    if model_choice == "sp_sg":
        return get_sp_sg_cfg()
    if model_choice == "sp_lg":
        return get_sp_lg_cfg()
    if model_choice == "disk_lg":
        return get_disk_lg_cfg()
    if model_choice == "sift_lg":
        return get_sift_lg_cfg()
    if model_choice == "dog_lg":
        return get_dog_lg_cfg()
    if model_choice == "aliked_lg":
        return get_aliked_lg_cfg()
    print(f"Error! {model_choice} not suported!")