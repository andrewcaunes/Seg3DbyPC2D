"""
Simple camera colorization for NuScenes point clouds.
Averages colors from all cameras that see each point.
"""
import os
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import argparse
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
import torch
import cv2
from seg_3D_by_2D.datasets.nuscenes.utils_nuscenes import get_closest_camera_tokens, get_T_lidar_to_camera

def get_camera_intrinsics(nusc, camera_data):
    """Get camera intrinsic parameters."""
    calibrated_sensor = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
    return np.array(calibrated_sensor['camera_intrinsic'])






def transform_points_batch(points, transform_matrix):
    """Transform points using torch for batch processing."""
    # Pre-allocate homogeneous points tensor with ones in last column
    batch_size = points.shape[0]
    points_homogeneous = torch.empty((batch_size, 4), dtype=points.dtype, device=points.device)
    points_homogeneous[:, :3] = points
    points_homogeneous[:, 3] = 1.0
    
    # Single matrix multiplication
    return torch.matmul(transform_matrix, points_homogeneous.t()).t()[:, :3]

def project_points_to_camera(points, 
                            intrinsic_matrix, 
                            extrinsic_matrix, 
                            image_shape,
                            camera_channel,
                            priorities,
                            min_depth=0,
                            max_depth=50,
                            depth_map=None,
                            depth_margin=0.5,
                            priorities_dict=None,
                            priority_depth_margin=0.5, # if given, use depth_margin for filtering points but use priority_depth_margin for updating priorities
                            distances=None
                            ):
    """Project 3D points to 2D camera coordinates with optional depth map occlusion testing."""
    # Transform points to camera coordinate system
    points_camera = transform_points_batch(points, extrinsic_matrix)
    
    # Get depths
    depths = points_camera[:, 2]
    
    # Project to image plane
    points_2d = torch.mm(intrinsic_matrix, points_camera.t()).t()
    depths_intrinsics = points_2d[:, 2].unsqueeze(1).clone()
    points_2d /= depths_intrinsics
    
    # Check which points are valid
    # logging.info('torch.unique(distances)=%s',torch.unique(distances))
    valid_mask = (depths > 0) & \
                 (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_shape[1]) & \
                 (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_shape[0]) & \
                 (depths >= min_depth) & (depths <= max_depth) 
    # logging.info('valid_mask.sum()=%s',valid_mask.sum())
    # valid_mask = valid_mask & (depths < distances)
    # logging.info('valid_mask.sum()=%s',valid_mask.sum())
    
    # Apply depth map occlusion testing if provided
    # logging.info('depth_map=%s',depth_map)
    if depth_map is not None and torch.any(valid_mask):
        valid_indices = torch.where(valid_mask)[0]
        valid_points_2d = points_2d[valid_mask].int()
        valid_depths = depths[valid_mask]
        
        # Get depth values from depth map at projected points
        depth_map_values = depth_map[valid_points_2d[:, 1], valid_points_2d[:, 0]]
        
        # Check if points are visible (not occluded)
        # Point is visible if its depth <= depth_map_value + margin
        visible_mask = valid_depths <= (depth_map_values + depth_margin)
        # logging.info('visible_mask.sum()=%s',visible_mask.sum())
        if priority_depth_margin is not None:
            visible_mask_lower_priority = visible_mask & (valid_depths >= (depth_map_values + priority_depth_margin))
            # logging.info('visible_mask_lower_priority.sum()=%s',visible_mask_lower_priority.sum())
        
        # Update valid_mask
        valid_mask[valid_indices] = visible_mask
        # valid_mask_low_priority = valid_mask.clone()
        # valid_mask_low_priority[valid_indices] = visible_mask_lower_priority
    
    if priorities_dict is not None: # filter out points that are already seen by a camera with higher priority
        current_priority = priorities_dict[camera_channel]
        valid_mask = valid_mask & (priorities <= current_priority) 
        priorities[valid_mask] = priorities_dict[camera_channel]
        if priority_depth_margin is not None:
            valid_mask_low_priority = valid_mask.clone()
            valid_mask_low_priority[valid_indices] = valid_mask_low_priority[valid_indices] & visible_mask_lower_priority
            priorities[valid_mask_low_priority] = current_priority // 2
        # logging.info('torch.unique(priorities)=%s',torch.unique(priorities))
    # distances[valid_mask] = depths[valid_mask]

    return points_2d, depths, valid_mask

def extract_colors_from_image(points_2d, image_path, valid_mask, device="cuda"):
    """Extract colors from camera image for projected points using torch."""
    # Load image and convert to torch tensor
    image = Image.open(image_path)
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).to(device=device, dtype=torch.float32) / 255.0  # (H, W, 3)
    
    colors = torch.zeros((points_2d.shape[0], 3), dtype=torch.float32, device=device)
    
    valid_points = points_2d[valid_mask].int()
    if len(valid_points) > 0:
        colors[valid_mask] = image_tensor[valid_points[:, 1], valid_points[:, 0], :3]
    
    return colors

def colorize_pointcloud_gpu(points, 
                           nusc, 
                           lidar_token, 
                           nuscenes_path, 
                           camera_channels,
                           priorities, # Tensor of ints corresponding to the priority of the last highest priority camera that saw each point
                           distances=None,
                           min_depth=0, 
                           max_depth=50,
                           depth_maps_folder=None,
                           depth_margin=0.5,
                           depth_encoding_max_depth=50,
                           depth_encoding_num_bins=5000,
                           depth_encoding_max_bits=2**16,
                           device="cuda",
                           priorities_dict=None,
                           ):
    """
    Colorize point cloud using torch for GPU acceleration with support for 6 cameras.
    
    Args:
        points: tensor of 3D points in shape (N, 3)
        nusc: NuScenes instance
        lidar_token: Token of the lidar data
        nuscenes_path: Path to NuScenes dataset
        camera_channels: List of camera channels to use
        min_depth: Minimum depth for valid points
        max_depth: Maximum depth for valid points
        depth_map_path: Optional path to depth map for occlusion testing
        depth_margin: Margin behind depth map to consider visible
        device: torch device
        
    Returns:
        colors: RGB colors for each point in shape (N, 3), normalized to [0, 1]
        counts: Number of cameras that saw each point in shape (N,)
    """
    with torch.no_grad():
        num_points = points.shape[0]
        
        # Get lidar data and camera tokens
        lidar_data = nusc.get('sample_data', lidar_token)
        camera_tokens_dict = get_closest_camera_tokens(nusc, lidar_token)
        
        # Initialize colors and counts arrays
        colors = torch.zeros((num_points, 3), dtype=torch.float32, device=device)
        counts = torch.zeros((num_points,), dtype=torch.float32, device=device)
        
        # Process each camera
        for camera_channel in camera_channels:
            camera_token = camera_tokens_dict[camera_channel]
            camera_data = nusc.get('sample_data', camera_token)
            
            # Get camera parameters
            intrinsic_matrix = get_camera_intrinsics(nusc, camera_data)
            extrinsic_matrix = get_T_lidar_to_camera(nusc, camera_data, lidar_data)
            
            if intrinsic_matrix is None or extrinsic_matrix is None:
                continue
                
            intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=torch.float32, device=device)
            extrinsic_matrix = torch.tensor(extrinsic_matrix, dtype=torch.float32, device=device)
            
            # Load image to get dimensions
            image_path = os.path.join(nuscenes_path, camera_data['filename'])
            if not os.path.exists(image_path):
                continue
            
            image = Image.open(image_path)
            image_shape = (image.height, image.width)
            
            # Load depth map if available
            depth_map = None
            if depth_maps_folder is not None:
                try:
                    depth_map_filename = os.path.basename(camera_data['filename']).replace(".jpg", ".png")
                    depth_map_path = os.path.join(depth_maps_folder, depth_map_filename)
                    if os.path.exists(depth_map_path):
                        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    depth_map = torch.tensor(depth_map, dtype=torch.float32, device=device, requires_grad=False)
                    depth_map = depth_map / (depth_encoding_max_bits // depth_encoding_num_bins) / depth_encoding_num_bins * depth_encoding_max_depth
                except:
                    logging.warning(f"Depth map not found: {depth_map_path}")
                    continue
            
            # Project points to camera
            points_2d, _, valid_mask = project_points_to_camera(
                points, 
                intrinsic_matrix, 
                extrinsic_matrix, 
                image_shape,
                camera_channel,
                priorities=priorities,
                min_depth=min_depth, 
                max_depth=max_depth,
                depth_map=depth_map, 
                depth_margin=depth_margin,
                priorities_dict=priorities_dict,
                distances=distances
            )
            
            if not torch.any(valid_mask):
                continue
            
            # if priorities_dict is not None: # filter out points that are already seen by a camera with higher priority
            #     current_priority = priorities_dict[camera_channel]
            #     valid_mask = valid_mask & (priorities <= current_priority) 
            #     priorities[valid_mask] = priorities_dict[camera_channel]

            # Extract colors
            camera_colors = extract_colors_from_image(points_2d, image_path, valid_mask, device)
            
            # Accumulate colors and counts
            colors[valid_mask] = camera_colors[valid_mask]
            counts[valid_mask] += 1.0
        
        # Average colors by counts (avoid division by zero)
        # valid_counts = counts > 0
        # if torch.any(valid_counts):
        #     colors[valid_counts] /= counts[valid_counts].unsqueeze(1)
    
    return colors, counts

# def project_points_to_camera_numpy(points, 
#             intrinsic_matrix, 
#             extrinsic_matrix, 
#             image_shape,
#             min_depth=0,
#             max_depth=50,):
#     """Project 3D points to 2D camera coordinates (numpy version for backward compatibility)."""
#     # Transform points to camera coordinate system
#     points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
#     points_camera = np.dot(extrinsic_matrix, points_homogeneous.T)
    
#     # Get depths
#     depths = points_camera[2, :]
    
#     # Project to image plane
#     points_image = np.dot(intrinsic_matrix, points_camera[:3, :])
#     points_image = points_image[:2, :] / points_image[2, :]
#     points_2d = points_image.T
    
#     # Check which points are valid
#     valid_mask = (depths > 0) & \
#                  (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_shape[1]) & \
#                  (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_shape[0]) & \
#                  (depths > min_depth) & (depths < max_depth)
    
#     return points_2d, depths, valid_mask


# def extract_colors_from_image_numpy(points_2d, image_path, valid_mask):
#     """Extract colors from camera image for projected points (numpy version for backward compatibility)."""
#     image = Image.open(image_path)
#     image_array = np.array(image)
    
#     colors = np.zeros((points_2d.shape[0], 3), dtype=np.float32)
    
#     valid_points = points_2d[valid_mask].astype(int)
#     if len(valid_points) > 0:
#         colors[valid_mask] = image_array[valid_points[:, 1], valid_points[:, 0]] / 255.0
    
#     return colors
