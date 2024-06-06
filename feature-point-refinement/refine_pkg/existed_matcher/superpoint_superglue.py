import torch
from refine_pkg.third_party.SuperGluePretrainedNetwork.models.matching import Matching

class SPSGMatcher:
    def __init__(self, n_features):
        self.n_features = n_features
        self.spsg_config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': n_features
            },
            'superglue': {
                'weights': "outdoor",
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matcher = Matching(self.spsg_config).eval().to("cuda")
    
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
            pred = self.matcher({'image0': images0[i][None], 'image1': images1[i][None]})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            sg_kpts0, sg_kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            
            # Keep the matching keypoints.
            valid = matches > -1
            valid_num = valid.sum()
            mkpts0 = sg_kpts0[valid] # valid, 2
            mkpts1 = sg_kpts1[matches[valid]] # valid, 2
            # mconf = conf[valid] # valid
        
            kpts0[i, :valid_num] = torch.from_numpy(mkpts0)
            kpts1[i, :valid_num] = torch.from_numpy(mkpts1)
            matches_mask[i, :valid_num] = True
        
        return kpts0, kpts1, matches_mask
    
if __name__ == '__main__':
    pass   
