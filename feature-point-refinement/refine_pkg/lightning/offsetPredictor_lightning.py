import cv2
import os
import torch
import numpy as np
from einops import rearrange
from pytorch_lightning import LightningModule
from refine_pkg.model.offsetPredictor import create_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from refine_pkg.utils.utils import (
    warp_kpts, 
    visualize_refine, 
    crop_patches, 
    filter_matches_below_err,
    estimate_pose,
    compute_pose_error,
    pose_auc
)
from refine_pkg.utils.utils import make_matching_plot_new

class OffsetPredictorLightning(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.input_channels = config.MODEL.INPUT_CHANNELS
        self.err_threshold = config.MODEL.ERR_THRESHOLD
        self.patch_size = config.MODEL.PATCH_SIZE
        self.lr = config.TRAINER.LR
        self.n_features = config.MODEL.N_FEATURES
        self.huber_delta = config.MODEL.HUBER_DELTA
        self.matcher_type = config.MODEL.MATCHER
        
        self.model = create_model(self.input_channels, self.patch_size)
        self.load_matcher()
        self.save_hyperparameters()
        
    def load_matcher(self):
        from refine_pkg.existed_matcher import load_detector_and_matcher
        self.matcher = load_detector_and_matcher(self.matcher_type, self.n_features)
        print(f"Matcher loaded, using {self.matcher_type} now.")
    
    def process_batch_data(self, data):
        # transform data from dataloader to supervison signal for training offset predictor
        # 这里的 data 是以 batch 的形式返回
        
        # 如果是对匹配后的特征点微调，这里的 kpts0 和 kpts1 是对应的
        kpts0, kpts1, kpts_mask = self.matcher.get_matched_points(data["image0"], data["image1"])
        
        # 先把 kpts0 scale 回原本的分辨率 scale: [w/w_new, h/h_new]   
        kpts0_orign = kpts0 * data["scale0"].unsqueeze(1)
        warp_valid_mask, w_kpts0_origin = warp_kpts(
            kpts0_orign, 
            data["depth0"],
            data["depth1"],
            data["T_0to1"],
            data["K0"],
            data["K1"]
        )
        # 再把 w_kpts0 scale 到图1现在的分辨率 warp_valid_mask: B, N
        w_kpts0 = w_kpts0_origin / data["scale1"].unsqueeze(1) # B, N, 2
        
        # 找到最近的匹配点, 返回 B, N, 2; B, N。对于初步匹配的特征点:
        nearest_valid_mask = filter_matches_below_err(kpts0, kpts1, w_kpts0, self.err_threshold)
        
        # 筛选有效的特征点
        valid_mask = kpts_mask.bool() & warp_valid_mask.bool() & nearest_valid_mask.bool() # 仅在这些位置计算损失
        
        # 返回对应的patch
        patch0 = crop_patches(data["image0"], kpts0, self.patch_size) # B N C p p
        patch1 = crop_patches(data["image1"], kpts1, self.patch_size) # B N C p p
        
        return kpts0, kpts1, w_kpts0, patch0, patch1, kpts_mask, valid_mask
    
    def forward(self, x1, x2):
        out = self.model(x1, x2)
        return out

    def training_step(self, batch, batch_idx):
        kpts0, kpts1, w_kpts0, patch0, patch1, kpts_mask, valid_mask = self.process_batch_data(batch)
        
        # reshape
        patch0 = rearrange(patch0, 'B n C p1 p2 -> (B n) C p1 p2')
        patch1 = rearrange(patch1, 'B n C p1 p2 -> (B n) C p1 p2')
        
        kpts1 = kpts1.reshape(-1, 2) # B*n, 2
        w_kpts0 = w_kpts0.reshape(-1, 2) # B*n, 2
        valid_mask = valid_mask.reshape(-1) # B*n
        
        # filter
        patch0 = patch0[valid_mask] # valid, C, p, p
        patch1 = patch1[valid_mask] # valid, C, p, p
        kpts1 = kpts1[valid_mask] # valid, 2
        w_kpts0 = w_kpts0[valid_mask] # valid, 2
        
        pred_offset = self(patch0, patch1) # valid, 2, (-1, 1)
        gt_offset = (w_kpts0 - kpts1) / self.err_threshold # valid, 2 
        diff_vec = gt_offset - pred_offset # valid, 2
        
        diff_norm = (diff_vec ** 2).sum(dim=-1)**0.5 # valid, 模长最大为2根2
        target_norm = torch.zeros_like(diff_norm)
        
        huber_loss_func = torch.nn.HuberLoss(reduction='mean', delta=self.huber_delta / self.err_threshold)  
        loss = huber_loss_func(diff_norm, target_norm)
        
        if torch.isnan(loss).any():
            return {"loss": torch.zeros_like(loss, requires_grad=True)} 
            
        # 记录数据
        self.log("train/loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train/before", before.detach(), on_step=True, prog_bar=False, logger=True)
        # self.log("train/after", after.detach(), on_step=True, prog_bar=False, logger=True)
        return {"loss": loss} 

    def evaluate(self, batch, batch_idx):
        # 读取数据
        kpts0, kpts1, w_kpts0, patch0, patch1, kpts_mask, valid_mask = self.process_batch_data(batch)
        K0 = batch['K0'][0].cpu().numpy()
        K1 = batch['K1'][0].cpu().numpy()
        thresh = 1.0 # 默认一个像素，其实可以调整
        T_0to1 = batch['T_0to1'][0].cpu().numpy() # 4, 4
        
        # reshape
        patch0 = rearrange(patch0, 'B n C p1 p2 -> (B n) C p1 p2')
        patch1 = rearrange(patch1, 'B n C p1 p2 -> (B n) C p1 p2')
        
        kpts0 = kpts0.reshape(-1, 2) # B*n, 2
        kpts1 = kpts1.reshape(-1, 2) # B*n, 2
        w_kpts0 = w_kpts0.reshape(-1, 2) # B*n, 2
        valid_mask = valid_mask.reshape(-1) # B*n
        kpts_mask = kpts_mask.reshape(-1) # B*n
        # 计算匹配网络返回的有效匹配点对中，哪些的gt距离小于误差范围，仅在这些点对上看微调网络做的如何
        valid_among_matches_mask = valid_mask[kpts_mask] # matches
        
        # 筛选出合法的匹配点
        patch0_pose = patch0[kpts_mask] # matches, C, p, p
        patch1_pose = patch1[kpts_mask] # matches, C, p, p
        kpts0_pose = kpts0[kpts_mask] # matches, 2
        kpts1_pose = kpts1[kpts_mask] # matches, 2
        
        gt_offset = w_kpts0[kpts_mask] - kpts1_pose # matches, 2 
        pred_offset = self(patch0_pose, patch1_pose) * self.err_threshold # matches, 2
        kpts1_pose_ = kpts1_pose + pred_offset # matches, 2
        
        # 对可视化数据进行保存
        draw_kpts0 = kpts0_pose
        draw_kpts1 = kpts1_pose
        draw_kpts1_ = kpts1_pose_
        
        # 可视化 Refine 过程
        visualize_patch = False
        if visualize_patch:
            method_name = self.matcher_type
            save_dir = f"./results_filtered/{method_name}/"
            # 对一张图上所有 patch 进行绘画
            visualize_refine(
                patch_left = patch0_pose[valid_among_matches_mask],
                patch_right = patch1_pose[valid_among_matches_mask],
                gt_offset = gt_offset[valid_among_matches_mask],
                pred_offset = pred_offset[valid_among_matches_mask], 
                save_dir = save_dir,
                prefix = batch_idx,
            )
            
        # 放缩回与初始内参相同的比例
        kpts0_pose = kpts0_pose * batch["scale0"] # matches, 2
        kpts1_pose = kpts1_pose * batch["scale1"]
        kpts1_pose_ = kpts1_pose_ * batch["scale1"]
        
        # 用修正前的匹配点对计算位姿
        ret_before = estimate_pose(
            kpts0_pose.cpu().numpy(), 
            kpts1_pose.cpu().numpy(), 
            K0, K1, thresh
        )
        if ret_before is None:
            inliers_before = 0
            err_t_before, err_R_before = np.inf, np.inf
        else:
            (R_before, t_before, inliers_before) = ret_before
            num_inliers_before = inliers_before.sum()
            err_t_before, err_R_before = compute_pose_error(T_0to1, R_before, t_before)
        
        # 用修正后的匹配点对计算位姿
        ret_after = estimate_pose(
            kpts0_pose.cpu().numpy(), 
            kpts1_pose_.cpu().numpy(), 
            K0, K1, thresh
        )
        if ret_after is None:
            inliers_after = 0
            err_t_after, err_R_after = np.inf, np.inf
        else:
            (R_after, t_after, inliers_after) = ret_after
            num_inliers_after = inliers_after.sum()
            err_t_after, err_R_after = compute_pose_error(T_0to1, R_after, t_after)
        
        # 在这里可视化整体的有效匹配结果
        if (num_inliers_before > 6) and (num_inliers_after > num_inliers_before * 1.3): 
            draw_img0 = batch['image0'].squeeze().cpu().numpy()
            draw_img1 = batch['image1'].squeeze().cpu().numpy()
            draw_img0 = (draw_img0 * 255).astype(np.uint8)
            draw_img1 = (draw_img1 * 255).astype(np.uint8)
            
            draw_kpts0 = draw_kpts0.cpu().numpy()
            draw_kpts1 = draw_kpts1.cpu().numpy()
            draw_kpts1 = draw_kpts1_.cpu().numpy()
            save_before = make_matching_plot_new(draw_img0, draw_img1, mkpts0=draw_kpts0[inliers_before], mkpts1=draw_kpts1[inliers_before])
            save_after = make_matching_plot_new(draw_img0, draw_img1, mkpts0=draw_kpts0[inliers_after], mkpts1=draw_kpts1[inliers_after])
            
            save_prefix = f"./match_results/{self.matcher_type}/"
            os.makedirs(save_prefix, exist_ok=True)
            
            cv2.imwrite(save_prefix + f"{batch_idx}_before_{num_inliers_before}.png", save_before)
            cv2.imwrite(save_prefix + f"{batch_idx}_after_{num_inliers_after}.png", save_after)
        
        # 看看 refine 的怎么样,看在原始分辨率上提升了多少
        w_kpts0 = w_kpts0 * batch["scale1"]
        gt_distance = (w_kpts0[kpts_mask] - kpts1_pose)[valid_among_matches_mask] # valid, 2 
        refined_distance = (w_kpts0[kpts_mask] - kpts1_pose_)[valid_among_matches_mask] # valid, 2 
        
        def tmp(distance):
            distance = distance ** 2        # valid, 2
            distance = distance.sum(dim=1)  # valid
            distance = distance ** 0.5
            distance = distance.mean(dim=0)
            return distance
        
        # 计算修正前具有的平均像素误差
        dist_before = tmp(gt_distance)
        # 计算修正后具有的误差
        dist_after = tmp(refined_distance)
            
        return {
            "before": {
                "err_R": err_R_before,
                "err_t": err_t_before,
                "num_valid_gt": valid_mask.sum().cpu().numpy(), # 有多少对 gt 得 valid 得匹配，用这些来计算 refine 得好坏
                "num_inliers": num_inliers_before, # 经过 ransac 筛选后还剩多少
                "pixel_dist": dist_before.cpu().numpy()
            },
            "after": {
                "err_R": err_R_after,
                "err_t": err_t_after,
                "num_valid_gt": valid_mask.sum().cpu().numpy(), # 有多少对 gt 得 valid 得匹配，用这些来计算 refine 得好坏
                "num_inliers": num_inliers_after,
                "pixel_dist": dist_after.cpu().numpy()
            }
        }

    def validation_step(self, batch, batch_idx):
        ret_dict = self.evaluate(batch, batch_idx)
        return ret_dict

    def validation_epoch_end(self, outputs):
        status = "before"
        if "before":
            pixel_dist_sum = 0
            valid_matches_sum = 0
            inliers_sum = 0
            pose_errors = []
            for x in outputs:
                pixel_dist = np.nan_to_num(x[status]["pixel_dist"])
                valid_matches = x[status]["num_valid_gt"]
                num_inliers = x[status]["num_inliers"]
                
                pixel_dist_sum += pixel_dist * valid_matches
                valid_matches_sum += valid_matches
                inliers_sum += num_inliers

                pose_error = np.maximum(x[status]['err_R'], x[status]['err_t'])
                pose_errors.append(pose_error)
            
            aucs = pose_auc(pose_errors)
            aucs = [100.*yy for yy in aucs]
            
            self.log(f"val/{status}/pixel_dist", pixel_dist_sum / valid_matches_sum, prog_bar=False, logger=True)
            self.log(f"val/{status}/num_inliers", inliers_sum / len(outputs), prog_bar=False, logger=True)
            self.log(f"val/{status}/AUC5", aucs[0], prog_bar=False, logger=True)
            self.log(f"val/{status}/AUC10", aucs[1], prog_bar=False, logger=True)
            self.log(f"val/{status}/AUC20", aucs[2], prog_bar=False, logger=True)
        
        status = "after"
        if "after":
            pixel_dist_sum = 0
            valid_matches_sum = 0
            inliers_sum = 0
            pose_errors = []
            for x in outputs:
                pixel_dist = np.nan_to_num(x[status]["pixel_dist"])
                valid_matches = x[status]["num_valid_gt"]
                num_inliers = x[status]["num_inliers"]
                
                pixel_dist_sum += pixel_dist * valid_matches
                valid_matches_sum += valid_matches
                inliers_sum += num_inliers

                pose_error = np.maximum(x[status]['err_R'], x[status]['err_t'])
                pose_errors.append(pose_error)
            
            aucs = pose_auc(pose_errors)
            aucs = [100.*yy for yy in aucs]
            
            self.log(f"val/{status}/pixel_dist", pixel_dist_sum / valid_matches_sum, prog_bar=False, logger=True)
            self.log(f"val/{status}/num_inliers", inliers_sum / len(outputs), prog_bar=False, logger=True)
            self.log(f"val/{status}/AUC5", aucs[0], prog_bar=False, logger=True)
            self.log(f"val/{status}/AUC10", aucs[1], prog_bar=False, logger=True)
            self.log(f"val/{status}/AUC20", aucs[2], prog_bar=False, logger=True)
            
            self.log("AUC_sum", aucs[0] + aucs[1] + aucs[2], prog_bar=False, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                factor=0.25, 
                patience=3, 
                threshold=0.1, 
                threshold_mode='abs', 
                verbose=True
            ),
            'monitor': 'AUC_sum',  # 指定要监测的评估指标
            'interval': 'epoch',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
  