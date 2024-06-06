from .orb_bf import ORBMatcher
from .sift_bf import SIFTMatcher
from .akaze_bf import AKAZEMatcher
from .brisk_bf import BRISKMatcher
from .superpoint_superglue import SPSGMatcher
from .superpoint_lightglue import SPLGMatcher
from .disk_lightglue import DISKLGMatcher
from .sift_lightglue import SIFTLGMatcher
from .aliked_lightglue import ALLGMatcher
from .dog_lightglue import DOGLGMatcher


def load_detector_and_matcher(matcher_type, n_features):
    """
    matcher_type: string, like orb_bf
    """
    if matcher_type == "orb_bf":
        return ORBMatcher(n_features)
    if matcher_type == "sift_bf":
        return SIFTMatcher(n_features)
    if matcher_type == "akaze_bf":
        return AKAZEMatcher(n_features)
    if matcher_type == "brisk_bf":
        return BRISKMatcher(n_features)
    if matcher_type == "sp_sg":
        return SPSGMatcher(n_features)
    if matcher_type == "sp_lg":
        return SPLGMatcher(n_features)
    if matcher_type == "disk_lg":
        return DISKLGMatcher(n_features)
    if matcher_type == "sift_lg":
        return SIFTLGMatcher(n_features)
    if matcher_type == "aliked_lg":
        return ALLGMatcher(n_features)
    if matcher_type == "dog_lg":
        return DOGLGMatcher(n_features)
    
    print(f"{matcher_type} is NOT supported!")