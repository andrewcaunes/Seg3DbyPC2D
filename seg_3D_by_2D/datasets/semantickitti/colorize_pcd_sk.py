"""
Simple camera colorization for KITTI point clouds.
Averages colors from all cameras that see each point.
"""
import os
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
from pyquaternion import Quaternion


def parse_calibration(filename):
    """Read KITTI calibration file with given filename.
    
    Returns
    -------
    dict
        Calibration matrices as 3x4 numpy arrays for P matrices and 4x4 for Tr.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        if key.startswith('P'):  # P0, P1, P2, P3 are 3x4 projection matrices
            pose = np.zeros((3, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
        else:  # Tr is 3x4 transformation matrix, convert to 4x4
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def get_camera_intrinsics(calibration, camera_id):
    """Get camera intrinsic parameters from KITTI calibration.
    
    Args:
        calibration: Calibration dictionary from parse_calibration
        camera_id: Camera ID (2 for left, 3 for right)
        
    Returns:
        intrinsic_matrix: 3x3 camera intrinsic matrix
    """
    # Ensure camera_id is an integer
    camera_id = int(camera_id)
    
    # KITTI uses P0, P1, P2, P3 for camera projection matrices
    P_key = f"P{camera_id}"
    if P_key not in calibration:
        logging.warning(f"Camera intrinsics {P_key} not found in calibration")
        return None
    
    P_rect = calibration[P_key]
    
    # Extract intrinsic matrix from projection matrix
    # P_rect is 3x4, we need 3x3 intrinsic matrix
    intrinsic_matrix = P_rect[:3, :3]
    
    return intrinsic_matrix

def get_camera_extrinsics(calibration, camera_id):
    """
    Returns a 4×4 matrix that transforms points from the LiDAR frame into the specified camera frame.
    """
    camera_id = int(camera_id)
    # Only support camera 2 directly
    if camera_id != 2:
        logging.warning(f"Extrinsic for camera {camera_id} is not directly supported. Returning camera 2 extrinsics.")

    # Get transformation from LiDAR to camera 2
    if "Tr" not in calibration:
        logging.warning("Tr (LiDAR to camera) not found in calibration data")
        return None

    return calibration["Tr"]

def transform_points_batch(points, transform_matrix):
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
                            min_depth=0.5, 
                            max_depth=50, 
                            depth_map=None, 
                            depth_margin=0.5):
    """Project 3D points to 2D camera coordinates with optional depth map occlusion testing.
    
    Args:
        points: 3D points in shape (N, 3)
        intrinsic_matrix: 3x3 camera intrinsic matrix
        extrinsic_matrix: 4x4 transformation matrix from LiDAR to camera
        image_shape: Tuple of (height, width)
        depth_map: Optional depth map for occlusion testing
        depth_margin: Margin behind depth map to consider visible (default: 0.5m)
        
    Returns:
        points_2d: 2D image coordinates in shape (N, 2)
        depths: Depth values in shape (N,)
        valid_mask: Boolean mask of valid projections
    """
    # Transform points to camera coordinate system
    # points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    # points_camera = np.dot(extrinsic_matrix, points_homogeneous.T)
    points_camera = transform_points_batch(points, extrinsic_matrix)
    # Get depths
    depths = points_camera[:, 2]
    
    # Project to image plane
    # points_image = np.dot(intrinsic_matrix, points_camera[:3, :])
    # points_image = points_image[:2, :] / points_image[2, :]
    # points_2d = points_image.T
    points_2d = torch.mm(intrinsic_matrix, points_camera.t()).t()
    depths_intrinsics = points_2d[:, 2].unsqueeze(1).clone()
    points_2d /= depths_intrinsics


    # Check which points are valid
    valid_mask = (depths > 0) & \
                 (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_shape[1]) & \
                 (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_shape[0]) & \
                 (depths >= min_depth) & (depths <= max_depth)
    
    # Apply depth map occlusion testing if provided
    if depth_map is not None and torch.any(valid_mask):
        valid_indices = torch.where(valid_mask)[0]
        valid_points_2d = points_2d[valid_mask].int()
        valid_depths = depths[valid_mask]
        
        # Get depth values from depth map at projected points
        
        # Clamp coordinates to depth map bounds to avoid out-of-bounds indexing
        h, w = depth_map.shape[0], depth_map.shape[1]
        valid_x = torch.clamp(valid_points_2d[:, 0], 0, w - 1)
        valid_y = torch.clamp(valid_points_2d[:, 1], 0, h - 1)
        depth_map_values = depth_map[valid_y, valid_x]
        
        # Convert depth map values back to actual depth (reverse the encoding)
        # Assuming the same encoding as in generate_depth_maps
        # depth_encoding_max_depth = 50
        # depth_encoding_num_bins = 5000
        # depth_encoding_max_bits = 2**16
        
        # Reverse the encoding
        # depth_map_values = depth_map_values.astype(float) / (depth_encoding_max_bits // depth_encoding_num_bins)
        # depth_map_values = depth_map_values / depth_encoding_num_bins * depth_encoding_max_depth
        
        # Check if points are visible (not occluded)
        # Point is visible if its depth <= depth_map_value + margin
        visible_mask = valid_depths <= (depth_map_values + depth_margin)
        
        # Update valid_mask
        valid_mask[valid_indices] = visible_mask
    
    return points_2d, depths, valid_mask


def extract_colors_from_image(points_2d, image_path, valid_mask, device="cuda"):
    """Extract colors from camera image for projected points using torch.
    
    Args:
        points_2d: 2D image coordinates in shape (N, 2) (torch tensor)
        image_path: Path to the camera image
        valid_mask: Boolean mask of valid projections (torch tensor or numpy array)
        device: torch device
        
    Returns:
        colors: RGB colors for each point in shape (N, 3), normalized to [0, 1] (torch tensor)
    """
    from PIL import Image
    import torch

    # Load image and convert to torch tensor
    image = Image.open(image_path)
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).to(device=device, dtype=torch.float32) / 255.0  # (H, W, 3)
    
    # colors = np.zeros((points_2d.shape[0], 3), dtype=np.float32)
    colors = torch.zeros((points_2d.shape[0], 3), dtype=torch.float32, device=device)
    
    valid_points = points_2d[valid_mask].int()
    if len(valid_points) > 0:
        # Get image dimensions
        h, w = image_tensor.shape[0], image_tensor.shape[1]
        
        # Clamp coordinates to image bounds to avoid out-of-bounds indexing
        valid_x = torch.clamp(valid_points[:, 0], 0, w - 1)
        valid_y = torch.clamp(valid_points[:, 1], 0, h - 1)
        
        colors[valid_mask] = image_tensor[valid_y, valid_x, :3]
    
    return colors




    # valid_indices = torch.where(valid_mask)[0]
    # if valid_indices.numel() > 0:
    #     valid_points = points_2d[valid_mask].long()
    #     # Clamp indices to image bounds to avoid out-of-bounds
    #     h, w = image_tensor.shape[0], image_tensor.shape[1]
    #     valid_x = torch.clamp(valid_points[:, 0], 0, w - 1)
    #     valid_y = torch.clamp(valid_points[:, 1], 0, h - 1)
    #     colors[valid_mask] = image_tensor[valid_y, valid_x, :3]


def colorize_pointcloud_gpu(points, 
                        calibration, 
                        scan_filename, 
                        kitti_path, 
                        depth_map_path=None, 
                        depth_margin=0.5,
                        max_depth=50,
                        min_depth=0.5,
                        depth_encoding_max_depth=50,
                        depth_encoding_num_bins = 5000, # depth values are binned to make the images lighter after compression (max 65535 bins)
                        depth_encoding_max_bits = 2**16,
                        device="cuda"):
    """
    Colorize point cloud using camera 2 (left color camera) with optional depth map occlusion testing.
    
    Args:
        points: tensor of 3D points in shape (N, 3)
        calibration: Calibration dictionary from parse_calibration
        scan_filename: Filename of the current scan (e.g., "000000.bin")
        kitti_path: Path to KITTI dataset root
        depth_maps_folder: Optional path to folder containing depth maps for occlusion testing
        depth_margin: Margin behind depth map to consider visible (default: 0.5m)
        
    Returns:
        colors: RGB colors for each point in shape (N, 3), normalized to [0, 1]
    """
    with torch.no_grad():
        num_points = points.shape[0]
        
        # Get scan index from filename
        scan_index = int(scan_filename.split('.')[0])
        
        # Get camera parameters for camera 2
        camera_id = 2
        intrinsic_matrix = get_camera_intrinsics(calibration, camera_id)
        intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=torch.float32, device=device)
        extrinsic_matrix = get_camera_extrinsics(calibration, camera_id)
        extrinsic_matrix = torch.tensor(extrinsic_matrix, dtype=torch.float32, device=device)
        
        if intrinsic_matrix is None or extrinsic_matrix is None:
            logging.warning(f"Camera {camera_id} parameters not available, skipping")
            raise

        # Load image to get dimensions
        image_filename = f"{scan_index:06d}.png"
        image_path = os.path.join(kitti_path, "image_2", image_filename)
        
        if not os.path.exists(image_path):
            logging.warning(f"Image not found: {image_path}")
            raise
        
        image = Image.open(image_path)
        image_shape = (image.height, image.width)
        
        # Load depth map if available
        depth_map = None
        if depth_map_path is not None:
            if os.path.exists(depth_map_path):
                depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                depth_map = torch.tensor(depth_map, dtype=torch.float32, device=device, requires_grad=False)
                depth_map = depth_map / (depth_encoding_max_bits // depth_encoding_num_bins) / depth_encoding_num_bins * depth_encoding_max_depth
        # Project points to camera
        try:
            points_2d, _, valid_mask = project_points_to_camera(
                points, 
                intrinsic_matrix, 
                extrinsic_matrix, 
                image_shape, 
                min_depth=min_depth,
                max_depth=max_depth,
                depth_map=depth_map, 
                depth_margin=depth_margin, 
            )
        except Exception as e:
            logging.error(f"Error projecting points to camera: {e}")
            return torch.zeros((num_points, 3), dtype=torch.float32, device=device)
        
        # logging.info('valid_mask=%s',valid_mask)
        # logging.info('valid_mask.device=%s',valid_mask.device)
        # logging.info('valid_mask.shape=%s',valid_mask.shape)
        # if not torch.any(valid_mask):
        #     logging.info(f"No valid points for camera {camera_id}")
        #     return torch.zeros((num_points, 3), dtype=torch.float32, device=device)
        # Extract colors
        colors = extract_colors_from_image(points_2d, image_path, valid_mask)
        
        # Count colored points
        # colored_points = torch.sum(torch.any(colors > 0, dim=1))
        # logging.info(f"Colored {colored_points}/{num_points} points ({colored_points/num_points*100:.1f}%)")
    
    return colors


def colorize_pointcloud(points, 
                        calibration, 
                        scan_filename, 
                        kitti_path, 
                        depth_maps_folder=None, 
                        depth_margin=0.5,
                        depth_max=50,
                        ):
    """
    Colorize point cloud using camera 2 (left color camera) with optional depth map occlusion testing.
    
    Args:
        points: 3D points in shape (N, 3)
        calibration: Calibration dictionary from parse_calibration
        scan_filename: Filename of the current scan (e.g., "000000.bin")
        kitti_path: Path to KITTI dataset root
        depth_maps_folder: Optional path to folder containing depth maps for occlusion testing
        depth_margin: Margin behind depth map to consider visible (default: 0.5m)
        
    Returns:
        colors: RGB colors for each point in shape (N, 3), normalized to [0, 1]
    """
    num_points = points.shape[0]
    
    # Get scan index from filename
    scan_index = int(scan_filename.split('.')[0])
    
    # Get camera parameters for camera 2
    camera_id = 2
    intrinsic_matrix = get_camera_intrinsics(calibration, camera_id)
    extrinsic_matrix = get_camera_extrinsics(calibration, camera_id)
    
    if intrinsic_matrix is None or extrinsic_matrix is None:
        logging.warning(f"Camera {camera_id} parameters not available, skipping")
        return np.zeros((num_points, 3), dtype=np.float32)
    
    # Load image to get dimensions
    image_filename = f"{scan_index:06d}.png"
    image_path = os.path.join(kitti_path, "image_2", image_filename)
    
    if not os.path.exists(image_path):
        logging.warning(f"Image not found: {image_path}")
        return np.zeros((num_points, 3), dtype=np.float32)
    
    image = Image.open(image_path)
    image_shape = (image.height, image.width)
    
    # Load depth map if available
    depth_map = None
    if depth_maps_folder is not None:
        depth_filename = f"depth_{scan_index:06d}.png"
        depth_path = os.path.join(depth_maps_folder, depth_filename)
        if os.path.exists(depth_path):
            import png
            with open(depth_path, 'rb') as f:
                depth_reader = png.Reader(file=f)
                depth_data = depth_reader.read()
                depth_map = np.array(list(depth_data[2]))
            # logging.info(f"Loaded depth map from {depth_path}")
            
            # Verify depth map dimensions match expected image size
            if depth_map.shape != (image_shape[0], image_shape[1]):
                logging.warning(f"Depth map size {depth_map.shape} doesn't match image size {image_shape}")
        else:
            logging.warning(f"Depth map not found: {depth_path}")
    
    # Project points to camera
    points_2d, depths, valid_mask = project_points_to_camera(
        points, intrinsic_matrix, extrinsic_matrix, image_shape, 
        depth_map=depth_map, depth_margin=depth_margin, depth_max=depth_max
    )
    
    if not np.any(valid_mask):
        logging.info(f"No valid points for camera {camera_id}")
        return np.zeros((num_points, 3), dtype=np.float32)
    
    # Extract colors
    colors = extract_colors_from_image(points_2d, image_path, valid_mask)
    
    # Count colored points
    colored_points = np.sum(np.any(colors > 0, axis=1))
    # logging.info(f"Colored {colored_points}/{num_points} points ({colored_points/num_points*100:.1f}%)")
    
    return colors
