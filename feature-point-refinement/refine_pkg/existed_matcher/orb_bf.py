import cv2
import torch
import numpy as np

class ORBMatcher:
    def __init__(self, n_features):
        self.n_features = n_features
        self.detector = cv2.ORB_create(nfeatures=self.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
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
            # 转换图像格式
            img0 = (images0[i][0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (images1[i][0].cpu().numpy() * 255).astype(np.uint8)

            # 检测ORB特征点和计算描述子
            kp0, des0 = self.detector.detectAndCompute(img0, None)
            kp1, des1 = self.detector.detectAndCompute(img1, None)
            
            # 确保数目不会超出
            kp0, des0 = kp0[:self.n_features], des0[:self.n_features]
            kp1, des1 = kp1[:self.n_features], des1[:self.n_features]

            # 确保检测到了特征点
            if des0 is not None and des1 is not None:
                # BF匹配
                matches = self.matcher.match(des0, des1)
                # 选取前n_features个匹配（如果有的话）
                matches = sorted(matches, key=lambda x: x.distance)

                # 填充特征点坐标和匹配掩码
                for j, match in enumerate(matches):
                    # 获取匹配特征点的坐标
                    pt0 = kp0[match.queryIdx].pt
                    pt1 = kp1[match.trainIdx].pt
                    # 转换为torch张量
                    kpts0[i, j] = torch.tensor(pt0, device=images0.device)
                    kpts1[i, j] = torch.tensor(pt1, device=images1.device)
                    # 标记此位置有有效匹配
                    matches_mask[i, j] = True
        
        return kpts0, kpts1, matches_mask