import io
import os
import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv
from PIL import Image, ImageDraw
import torch.nn.functional as F
from einops import rearrange

# 读取图片

def imread_gray(path, augment_fn=None):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), cv_type)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn)
    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask)

    return image, mask, scale


def read_megadepth_depth(path, pad_to=None):
    depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    if len(depth.shape) < 2:
        import pdb;pdb.set_trace()
    return depth


# --- ScanNet ---

def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


def read_scannet_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]

import torch


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0

def find_nearest_point(kpts1, kpts2, w_kpts1, threshold):
    """
    返回一个 N1, 2 的向量, 代表了对于图像1上每个特征点, 它在图像2上的对应点附近threshold距离里最近的图像2的特征点的坐标
    同时返回一个 N1 的布尔向量, 如果没找到足够近的对应点, 那就置为0
    """
    N1 = kpts1.size(0)
    N2 = kpts2.size(0)
    
    # 扩展kpts1_on_2和kpts2以便于广播计算所有配对的距离
    w_kpts1_expanded = w_kpts1.unsqueeze(1).expand(-1, N2, -1)
    kpts2_expanded = kpts2.unsqueeze(0).expand(N1, -1, -1)
    
    # 计算距离
    distances = torch.sqrt(((w_kpts1_expanded - kpts2_expanded) ** 2).sum(dim=2))
    
    # 找到每个点的最近距离和索引
    min_distances, min_indices = torch.min(distances, dim=1)
    
    # 判断最近的点是否在阈值内
    valid = min_distances < threshold
    
    # 提取最近点的坐标
    nearest_points = torch.zeros_like(kpts1)
    nearest_points[valid] = kpts2[min_indices[valid]]
    
    return nearest_points, valid

def find_nearest_point_batched(kpts1, kpts2, w_kpts1, threshold):
    """
    通过调用原始的 find_nearest_point_torch 函数来处理带有batch维度的数据。
    返回一个 B, N1, 2 的张量，代表了对于每个batch中图像1上每个特征点，
    它在图像2上的对应点附近threshold距离里最近的图像2的特征点的坐标。
    同时返回一个 B, N1 的布尔向量，如果没找到足够近的对应点，那就置为0。
    """
    B = kpts1.size(0)  # 获取batch的大小
    # 初始化返回值
    batch_nearest_points = torch.zeros_like(kpts1).to(kpts1.device)
    batch_valid = torch.zeros((B, kpts1.size(1)), dtype=torch.bool).to(kpts1.device)
    
    # 遍历每个batch，并调用原始函数
    for b in range(B):
        nearest_points, valid = find_nearest_point(kpts1[b], kpts2[b], w_kpts1[b], threshold)
        batch_nearest_points[b] = nearest_points
        batch_valid[b] = valid
    
    return batch_nearest_points, batch_valid

def filter_matches_below_err(kpts1, kpts2, w_kpts1, threshold):
    """
    kpts1: B, N, 2
    kpts2: B, N, 2
    w_kpts1: B, N, 2
    
    return: 
        batch_valid: B, N
    计算对应的 w_kpts1 和 kpts2 之间的像素间距离（欧几里得距离）,如果小于 threshold, 则令batch_valid对应位置的值为 True, 否则为False
    """
    batch_valid = torch.zeros((kpts1.size(0), kpts1.size(1)), dtype=torch.bool).to(kpts1.device)
    # 计算差异
    diffs = w_kpts1 - kpts2
    
    # 计算欧几里得距离（平方和的平方根）
    dists = torch.sqrt(torch.sum(diffs ** 2, dim=-1))
    
    # 根据阈值更新 batch_valid 的值
    batch_valid = dists < threshold
    return batch_valid

def draw_keypoints_on_patch(image_tensor, kpts_tensor, color):
    """
    - image_tensor: 归一化到0-1的灰度图像的PyTorch张量, 形状为(1, H, W)。
    - kpts_tensor: 特征点的图像坐标张量, 形状为 2。
    - color: 绘制点的颜色，格式为(R, G, B)，每个值应在[0, 1]范围内。
    """
    # 首先复制一份image_tensor以避免修改原始数据
    image_tensor = image_tensor.clone()
    
    # 转换为3通道RGB以绘制彩色点
    if image_tensor.shape[0] == 1:
        image_tensor_rgb = torch.cat([image_tensor, image_tensor, image_tensor], 0) # 3, h, w
    else:
        image_tensor_rgb = image_tensor
        
    # 绘制特征点
    x, y = round(kpts_tensor[0].item()), round(kpts_tensor[1].item())
    # 注意，这里假设color已经是归一化到[0, 1]的值
    image_tensor_rgb[:, y, x] = torch.tensor(color, dtype=torch.float32)
    
    # 返回与原始image_tensor形状相同的tensor，但需要转换回单通道灰度
    return image_tensor_rgb

def visualize_refine(patch_left, patch_right, gt_offset, pred_offset, save_dir, prefix):
    """
    patch: valid, C, p, p
    offset: valid, 2
    修改一下, 使得只显示效果明显的refine结果: 特指 gt_offset 不小且  gt_offset与pred_offset 相差很小的
    """
    os.makedirs(save_dir, exist_ok=True)
    r = (1, 0, 0)
    g = (0, 1, 0)
    patch_left, patch_right, gt_offset, pred_offset = patch_left.cpu(), patch_right.cpu(), gt_offset.cpu(), pred_offset.cpu()
    valid_num = patch_left.shape[0]
    if (valid_num < 1):
        return None
    
    def vector_magnitude(vector):
        # Destructure the vector into its components
        x, y = vector
        # Calculate the magnitude using the Pythagorean theorem
        magnitude = (x**2 + y**2)**0.5
        return magnitude
    
    for i in range(valid_num):
        # 只保留明显的匹配
        if (vector_magnitude(gt_offset[i]) <= 3) or (vector_magnitude(gt_offset[i] - pred_offset[i]) > 1):
            continue
        image_left = patch_left[i]
        image_right = patch_right[i]
        
        save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        patch_size = patch_left.shape[-1]
        center = torch.Tensor([patch_size / 2., patch_size / 2.])
        # import pdb;pdb.set_trace()
        initial_left = draw_keypoints_on_patch(image_left, center, r) # C, H, W
        initial_right = draw_keypoints_on_patch(image_right, center, r)
        refine_right = draw_keypoints_on_patch(image_right, center + pred_offset[i], r)
        gt_right = draw_keypoints_on_patch(image_right, center + gt_offset[i], r)
        
        # concate 四个返回的绘制图像，每个大小都为 C, H, W
        concate_img = torch.cat((initial_left, initial_right, refine_right, gt_right), dim=-1) # C, H, 4W
        # 将tensor转换回PIL图像以保存
        image_to_save = Image.fromarray((concate_img.permute(1, 2, 0).numpy() * 255).astype('uint8')) # H, 4W, C
        image_to_save.save(save_path)

def crop_patches(image, pixel_coords, patch_size):
    """
    使用grid_sample根据给定的像素坐标和patch大小从图像中裁剪出patches适应新的坐标定义。
    
    :param image: 输入的图像，形状为(B, C, H, W)的张量。
    :param pixel_coords: 一系列像素坐标，形状为(B, N, 2)的张量，坐标形式适应新定义。
    :param patch_size: 裁剪的patch大小应为奇数。
    :return: 裁剪出的图像patches形状为(B, N, C, patch_size, patch_size)的张量。
    """
    # assert patch_size % 2 == 1, "patch_size must be odd"
    patch_size = int(patch_size)
    
    B, N, _ = pixel_coords.shape
    C, H, W = image.shape[1], image.shape[2], image.shape[3]
    device = image.device
    half_patch = patch_size / 2.
    
    # 转换pixel_coords到归一化坐标系中，考虑到坐标定义从(0, 0)到(W, H)
    pixel_coords_normalized = pixel_coords.clone()
    pixel_coords_normalized[..., 0] = (pixel_coords[..., 0] / W) * 2 - 1
    pixel_coords_normalized[..., 1] = (pixel_coords[..., 1] / H) * 2 - 1
    
    # 创建patch_size大小的归一化网格
    # t.linspace(a, b, num)会返回num个点，首尾分别对应值a和b
    linspace_x = torch.linspace((-half_patch + 0.5) / W, (half_patch - 0.5) / W, patch_size, device=device) 
    linspace_y = torch.linspace((-half_patch + 0.5) / H, (half_patch - 0.5) / H, patch_size, device=device)
    grid_x, grid_y = torch.meshgrid(linspace_x, linspace_y, indexing='xy') # patch_size * patch_size
    
    grid_x = grid_x.expand(B, N, -1, -1)
    grid_y = grid_y.expand(B, N, -1, -1)
    
    grid = torch.stack((grid_x, grid_y), dim=-1) # B, N, p, p, 2
    grid += pixel_coords_normalized.unsqueeze(-2).unsqueeze(-2)
    
    # 使用grid_sample采样patches
    patches = F.grid_sample(image, grid.view(B, N * patch_size, patch_size, 2), padding_mode='zeros', align_corners=False)
    # B, C, np, p
    patches = rearrange(patches, 'B C (n p1) p2 -> B n C p1 p2', n=N)
    return patches

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3): # 当 E 是 3n, 3 的矩阵时，分为 n 个 3, 3 的矩阵的意思,按理说只会有一个E？？
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0) # R: 3, 3    t: 3   n (True/False)
    return ret

def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds=[5, 10, 20]):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

# make_matching_plot_fast(image0, image1, None, None, mkpts0, mkpts1, )
def make_matching_plot_new(image0, image1, mkpts0, mkpts1, margin=10):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        # cv2.circle(out, (x0, y0), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        # cv2.circle(out, (x1 + margin + W0, y1), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    return out

if __name__ == "__main__":
    def test_estimate_pose():
        # 模拟关键点，这里随机生成一些点作为例子
        np.random.seed(42)  # 确保结果可重现
        kpts0 = np.random.rand(100, 2) * 1000  # 假设有100个关键点
        kpts1 = kpts0 + np.random.rand(100, 2) * 5  # 第二图的关键点相对于第一图有小幅度偏移

        # 模拟相机内参矩阵，这里使用简化的单位矩阵假设相机焦距为1且主点在(0, 0)
        K0 = np.eye(3)
        K1 = np.eye(3)

        # 设置阈值和置信度
        thresh = 1.0
        conf = 0.99999

        # 调用函数
        ret = estimate_pose(kpts0, kpts1, K0, K1, thresh, conf)

        # 验证返回结果
        if ret is not None:
            R, t, mask = ret
            print("Rotation Matrix R:", R)
            print("Translation Vector t:", t)
            print("Inliers Mask:", mask)

            # 这里可以添加更多的断言来验证R, t等是否符合预期
            assert R is not None
            assert t is not None
            assert mask is not None
            print("Test passed!")
        else:
            print("No pose estimated, possibly not enough keypoints.")
            # 根据实际情况，这里可能需要一个断言来确保这是预期的行为
            assert len(kpts0) >= 5

    # 调用测试函数
    test_estimate_pose()