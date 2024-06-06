import torch
from refine_pkg.third_party.LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from refine_pkg.third_party.LightGlue.lightglue.utils import load_image, rbd

class DOGLGMatcher:
    def __init__(self, n_features):
        self.n_features = n_features
        # DoGHardNet+LightGlue
        self.detector = DoGHardNet(max_num_keypoints=n_features).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='doghardnet').eval().cuda()  # load the matcher
    
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

def test_splg_matcher():
    # 初始化参数
    n_features = 512  # 假设我们想要提取512个特征点
    B, C, H, W = 4, 1, 240, 320  # 假设有4张3通道的240x320的图像
    
    # 初始化匹配器
    matcher = SPLGMatcher(n_features=n_features)
    
    # 生成随机图像数据
    images0 = torch.rand(B, C, H, W).cuda()  # 假设输入数据已经是归一化到[0, 1]区间的
    images1 = torch.rand(B, C, H, W).cuda()
    
    # 调用get_matched_points方法
    kpts0, kpts1, matches_mask = matcher.get_matched_points(images0, images1)
    
    # 输出检查
    print("kpts0 shape:", kpts0.shape)
    print("kpts1 shape:", kpts1.shape)
    print("matches_mask shape:", matches_mask.shape)
    
    # 确保形状是我们预期的
    assert kpts0.shape == (B, n_features, 2), "Unexpected shape for kpts0"
    assert kpts1.shape == (B, n_features, 2), "Unexpected shape for kpts1"
    assert matches_mask.shape == (B, n_features), "Unexpected shape for matches_mask"
    
    print("Test passed!")
    
if __name__ == "__main__":
    test_splg_matcher()
    