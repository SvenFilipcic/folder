import numpy as np
import torch
import copy

def _fps_indices(points_tensor, n_samples):
    """Core FPS — returns sample indices as a CPU long tensor. points_tensor: (N, 3) cuda float."""
    num_points = points_tensor.size(0)
    sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda()
    dists = torch.ones(num_points).cuda() * float('inf')
    selected = torch.randint(num_points, (1,), dtype=torch.long).cuda()
    sample_inds[0] = selected
    for i in range(1, n_samples):
        last_added = sample_inds[i - 1]
        dist_to_last = torch.sum((points_tensor[last_added] - points_tensor) ** 2, dim=-1)
        dists = torch.min(dist_to_last, dists)
        selected = torch.argmax(dists)
        sample_inds[i] = selected
    return sample_inds


def furthest_point_sampling(points, colors=None, n_samples=2048):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    pts = torch.Tensor(points).cuda()
    if colors is not None:
        colors = torch.Tensor(colors).cuda()
    sample_inds = _fps_indices(pts, n_samples)
    if colors is None:
        return pts[sample_inds].cpu().numpy()
    else:
        return pts[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()


def furthest_point_sampling_idx(points, n_samples):
    """
    Like furthest_point_sampling but returns the selected indices as a numpy int array (n_samples,).
    Useful when you need to apply the same selection to additional per-point data (e.g. normals).
    points: [N, 3] numpy array (only xyz used for FPS).
    """
    pts = torch.Tensor(points[:, :3]).cuda()
    sample_inds = _fps_indices(pts, n_samples)
    return sample_inds.cpu().numpy()

def normalize_pcd_points(pcd_points, x_range=(-1, 1), y_range=(-1, 1)):
    '''
    Normalize point cloud points to a given range.
    '''
    # calcaulate centroid
    centroid = np.mean(np.asarray(pcd_points), axis=0)

    # move centroid to origin
    normalized_points = np.asarray(pcd_points) - centroid

    # calculate scale factor
    min_coords = np.min(normalized_points[:, :2], axis=0)
    max_coords = np.max(normalized_points[:, :2], axis=0)
    scale_x = (x_range[1] - x_range[0]) / (max_coords[0] - min_coords[0]) if (max_coords[0] - min_coords[0]) != 0 else 1
    scale_y = (y_range[1] - y_range[0]) / (max_coords[1] - min_coords[1]) if (max_coords[1] - min_coords[1]) != 0 else 1
    scale = min(scale_x, scale_y)

    # scale
    normalized_points = normalized_points * scale

    return normalized_points, centroid, scale

def rotate_point_cloud_relative_to_origin_point(points, euler_angles):
        '''
        rotate point cloud relative to origin point
        '''
        points = np.asarray(points)

        roll, pitch, yaw = np.deg2rad(euler_angles)  # 将角度转换为弧度

        # 绕 x 轴旋转的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # 绕 y 轴旋转的旋转矩阵
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # 绕 z 轴旋转的旋转矩阵
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 计算总的旋转矩阵，旋转顺序为 R_z * R_y * R_x
        rotation_matrix = R_z @ R_y @ R_x

        # 应用旋转矩阵到点云
        rotated_points = points @ rotation_matrix.T  # 转置矩阵进行点乘

        return rotated_points

def rotate_point_cloud(pcd, euler_angles, center_point=np.array([0, 0, 0])):
    '''
    rotate point cloud around a given center point
    '''
    points = copy.deepcopy(pcd)

    points = np.asarray(points)
    center_point = np.asarray(center_point)

    # 将点云平移到原点
    points -= center_point

    # 旋转点云
    rotated_points = rotate_point_cloud_relative_to_origin_point(points, euler_angles)

    # 将点云平移回原始位置
    rotated_points += center_point

    return rotated_points
