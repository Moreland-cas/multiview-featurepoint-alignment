import torch
from refine_pkg.third_party.LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from refine_pkg.third_party.LightGlue.lightglue.utils import load_image, rbd

class DISKLGMatcher:
    def __init__(self, n_features):
        self.n_features = n_features
        # Disk+LightGlue
        self.detector = DISK(max_num_keypoints=n_features).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='disk').eval().cuda()  # load the matcher
    
    @torch.no_grad()
    def get_matched_points(self, images0, images1):
        """
        输入:
            images0: B, C, H, W
        返回:
            kpts0: B, N, 2
            kpts1: B, N, 2
            matches_mask: B, N
        """
        B, _, _, _ = images0.shape  # 图像批次大小
        # 初始化特征点坐标和匹配掩码
        kpts0 = torch.full((B, self.n_features, 2), 0.0, device=images0.device) 
        kpts1 = torch.full((B, self.n_features, 2), 0.0, device=images1.device) 
        matches_mask = torch.zeros((B, self.n_features), dtype=torch.bool, device=images0.device)

        for i in range(B):
            # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
            feats0 = self.detector.extract(images0[i])  # auto-resize the image, disable with resize=None
            feats1 = self.detector.extract(images1[i])

            # match the features
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)
            points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
            
            # import pdb;pdb.set_trace()
            if matches is not None:
                valid_num = matches.shape[0]
            else:
                valid_num = 0
            kpts0[i, :valid_num] = points0
            kpts1[i, :valid_num] = points1
            matches_mask[i, :valid_num] = True
        
        return kpts0, kpts1, matches_mask
    