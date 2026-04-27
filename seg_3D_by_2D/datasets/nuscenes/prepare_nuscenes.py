"""
Script made by Andrew Caunes.
Prepare datasets such as NuScenes for processing by seg_3D_by_2D.py.
Example use:
python script.py ...
"""
import multiprocessing
import os
import shutil
import argparse
import logging

from tqdm import tqdm
import pyrender
import png

logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import open3d as o3d
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud
from pyquaternion import Quaternion
from seg_3D_by_2D.utils.utils import find_roots, write_log, save_las_file, filter_intensity_knn_batch, get_scene_name_from_output
from seg_3D_by_2D.datasets.nuscenes.colorize_pcd_ns import colorize_pointcloud_gpu
import torch
from PIL import Image
from seg_3D_by_2D.datasets.nuscenes.utils_nuscenes import (
    get_scene_from_name, 
    get_T_lidar_to_ref_lidar, 
    generate_real_camera_poses, 
    generate_depth_maps_nuscenes)


def add_intensity_to_pointcloud(pcd, pointcloud, max_intensity=245, min_intensity=0):
    intensities = np.array(pointcloud.points)[3, :] / 255.0
    # Compute the histogram
    hist, bins = np.histogram(intensities, bins=256, range=(intensities.min(), intensities.max()), density=False)
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    # Normalize the CDF
    cdf_normalized = (cdf - cdf.min()) * (max_intensity-min_intensity) / (cdf.max() - cdf.min()) + min_intensity
    cdf_normalized = cdf_normalized.astype('uint8')    
    # Use linear interpolation of the CDF to find new pixel values
    intensities = np.interp(intensities.flatten(), bins[:-1], cdf_normalized)
    normalized_intensities = intensities.reshape(pointcloud.points.shape[1], 1) / 255.0
    # max_cap = 0.3
    # intensities = np.clip(intensities, 0, max_cap)
    # normalized_intensities = np.interp(intensities, (intensities.min(), intensities.max()), (0, 1))
    colors = np.zeros((pointcloud.points.shape[1], 3))
    colors[:, :] = normalized_intensities.reshape(-1, 1)
    # colors[:, :] = intensities.reshape(-1, 1)
    pcd.colors = o3d.utility.Vector3dVector(colors)


def crop_pointcloud(pointcloud, radius_close=None, box_far=None) -> np.ndarray:
    """
    Removes point too close within a certain radius from origin.
    :param radius: radius below which points are removed
    """
    x_filt_close = np.abs(pointcloud.points[0, :]) < radius_close
    y_filt_close = np.abs(pointcloud.points[1, :]) < radius_close
    z_filt_close = np.abs(pointcloud.points[2, :]) < radius_close
    close = np.logical_and(np.logical_and(x_filt_close, y_filt_close), z_filt_close)
    if box_far is None:
        far = np.zeros(pointcloud.points.shape[1], dtype=bool)
    else:
        x_min, x_max = box_far[0]
        y_min, y_max = box_far[1]
        z_min, z_max = box_far[2]
        x_filt_far = np.logical_or(pointcloud.points[0, :] < x_min, pointcloud.points[0, :] > x_max)
        y_filt_far = np.logical_or(pointcloud.points[1, :] < y_min, pointcloud.points[1, :] > y_max)
        z_filt_far = np.logical_or(pointcloud.points[2, :] < z_min, pointcloud.points[2, :] > z_max)
        far = np.logical_or(np.logical_or(x_filt_far, y_filt_far), z_filt_far)
    not_cropped = np.logical_not(np.logical_or(close, far))
    pointcloud.points = pointcloud.points[:, not_cropped]
    return not_cropped






def prepare_nuscenes(roots, 
                    nusc,
                    nuscenes_path, 
                    exp_name, 
                    plot=False, 
                    log_path=None, 
                    save_gt=True, 
                    save_sensor_positions=False, # save an array of shape (N,3) with the sensor positions for each point
                    shared=False,
                    point_segmasks_gt_filename="point_segmasks_gt.npy",
                    sensor_positions_filename="sensor_positions.npz",
                    re_prepare=False,
                    recompute_gt_only=False,
                    lidar_poses_filename="lidar_poses.npy",
                    radius_close=2,
                    box_far=[[-1000, 1000], [-1000, 1000], [-2.5, 1000]],
                    blur_intensity_k=1,
                    blur_with_random_intensity=False,
                    blur_random_values=[1, 1, 3, 5, 8],
                    max_intensity=245,
                    min_intensity=0,
                    colorize_with_cameras=False,
                    colorize_max_depth=50,
                    colorize_min_depth=0,
                    colorize_depth_margin=0.5,
                    colorize_depth_map_point_size=3,
                    freq_colorize=1,
                    camera_channels=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                    device_project="cuda",
                    subproc=True,
                    ):
    """Prepare nuscenes dataset for processing by seg_3D_by_2D.
    For each scene of nuscenes, create a folder in output_path then save 3 files to be processed : 
    - registered_pointcloud.las
    - timestamps_infos.npz
    - lidar_poses.npy
    timestamps_infos is a Nx3 array with timestamps(int), is_in_registered_pointcloud(bool), file_index(int).
    Note : the timestamps are NOT actual timestamps but only indices to identify each point. This is because nuscenes
    does not give timestamps for each points.
    Note : the file indices are the last integer in each filename in nuscenes
    file naming convention, for memory efficiency."""

    
    logging.info('Cropping pointclouds with radius_close=%s and box_far=%s', radius_close, box_far)

    # Checking required files
    required_files = {"registered_pointcloud.las", "timestamps_infos.npz", "lidar_poses.npy"}
    if save_gt:
        required_files.add(point_segmasks_gt_filename)
    if save_sensor_positions:
        required_files.add(sensor_positions_filename)
    skipping = []
    for root in roots:
        if isinstance(shared, str):
            save_folder = os.path.join(root, shared)
        elif shared:
            save_folder = os.path.join(root)
        else :
            save_folder = os.path.join(root, exp_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        else:
            missing_files = [required_file for required_file in required_files if not os.path.exists(os.path.join(save_folder, required_file))]
            if len(missing_files) == 0 and not re_prepare and not recompute_gt_only:
                skipping.append(root)
                write_log(log_path, root)
                continue
            else:
                logging.info("Missing files : %s", ", ".join(missing_files))



        if len(skipping) > 0:
            logging.info("Skipped from %s to %s", get_scene_name_from_output(skipping[0]), get_scene_name_from_output(skipping[-1]))
            logging.info("Preparing %s", get_scene_name_from_output(root))
        if recompute_gt_only:
            logging.info("Recomputing gt only for %s", get_scene_name_from_output(root))
        logging.info("Required files : %s", ", ".join(required_files))

        if nusc is None:
            # Load nuscenes
            logging.info("Loading nuscenes")
            nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_path, verbose=True)
        scene_name = os.path.basename(root) if not save_folder.endswith("/") else os.path.basename(root[:-1])
        scenes = get_scene_from_name(nusc, scene_name)
        if len(scenes) == 0:
            logging.info('No scene found for scene_name=%s',scene_name)
            continue
        elif len(scenes)==1:
            scene = scenes[0]
        else :
            logging.info('Multiple scenes found for scene_name=%s',scene_name)
            continue

        registered_pointcloud_path = os.path.join(save_folder, "registered_pointcloud.las")
        timestamps_infos_path = os.path.join(save_folder, "timestamps_infos.npz")
        lidar_poses_path = os.path.join(save_folder, lidar_poses_filename)
        if save_gt:
            point_segmasks_gt_path = os.path.join(save_folder, point_segmasks_gt_filename)
        if save_sensor_positions:
            sensor_positions_path = os.path.join(save_folder, sensor_positions_filename)

        # initialize ouptput
        lidar_poses = [] # poses will be added in the loop
        timestamps_infos = np.empty((0, 3), dtype=np.int32)
        timestamps_registered_pointcloud = np.array([], dtype=np.int32)
        max_timestamp = 0
        if save_gt: 
            point_segmasks_gt = np.empty((0, 3), dtype=np.int32)
        if save_sensor_positions:
            sensor_positions = np.empty((0, 3), dtype=np.float32)
        
        # Get first lidar data to initialize reference
        sample_token = scene["first_sample_token"]
        sample = nusc.get('sample', sample_token)
        first_lidar_token = sample['data']['LIDAR_TOP']
        lidar_data_ref = nusc.get('sample_data', first_lidar_token)

        # Get all lidar tokens for the sequence
        lidar_tokens = []
        current_lidar_data = lidar_data_ref
        while True:
            lidar_tokens.append(current_lidar_data['token'])
            next_token = current_lidar_data['next']
            if next_token == "":
                break
            current_lidar_data = nusc.get('sample_data', next_token)

        # Process all scans in a single loop
        for i, lidar_token in enumerate(tqdm(lidar_tokens, desc="Registering Lidar Sweeps", unit="sweep")):
            lidar_data = nusc.get('sample_data', lidar_token)
            
            # For first scan (i==0), use identity pose, otherwise compute relative pose
            if i == 0:
                lidar_pose = np.eye(4)
                lidar_data_ref = lidar_data  # Set reference for subsequent scans
            else:
                lidar_pose = get_T_lidar_to_ref_lidar(nusc, lidar_data, lidar_data_ref)
            
            lidar_poses.append(lidar_pose)

            # Get scan and gt filenames
            file_index = int(lidar_data["filename"].split("_")[-1].split(".")[0])
            pointcloud_filename = os.path.join(nusc.dataroot, lidar_data["filename"])
            # logging.info('pointcloud_filename=%s',pointcloud_filename)
            gt_filename = None
            if save_gt and lidar_data["is_key_frame"]:
                gt_data = nusc.get('lidarseg', lidar_data['token'])
                gt_filename = os.path.join(nusc.dataroot, gt_data["filename"])

            # Load pointcloud and gt pointcloud
            lidar_pointcloud = LidarPointCloud.from_file(pointcloud_filename)
            if gt_filename is not None:
                gt_pointcloud = LidarSegPointCloud(points_path=pointcloud_filename, labels_path=gt_filename)
            else:
                gt_pointcloud = None

            # Update timestamps
            if i == 0:
                max_timestamp = lidar_pointcloud.points.shape[1]
                timestamps_all = np.arange(max_timestamp)
            else:
                new_max_timestamp = max_timestamp + lidar_pointcloud.points.shape[1]
                timestamps_all = np.arange(max_timestamp, new_max_timestamp)
                max_timestamp = new_max_timestamp

            # Crop pointcloud
            if radius_close is not None or box_far is not None:
                not_cropped = crop_pointcloud(lidar_pointcloud, radius_close=radius_close, box_far=box_far)
            else:
                not_cropped = np.ones(lidar_pointcloud.points.shape[1], dtype=bool)
            num_points_not_cropped = np.sum(not_cropped)

            if not recompute_gt_only:
                # Transform pointcloud (only for non-first scans)
                if i > 0:
                    lidar_pointcloud.points[:3,:] = np.dot(lidar_pose, np.vstack((lidar_pointcloud.points[:3,:], np.ones(lidar_pointcloud.points.shape[1]))))[:3, :]

                # Convert to open3d format and add to pointcloud
                pcd_current = o3d.geometry.PointCloud()
                pcd_current.points = o3d.utility.Vector3dVector(lidar_pointcloud.points[:3, :].T)
                add_intensity_to_pointcloud(pcd_current, lidar_pointcloud, max_intensity=max_intensity, min_intensity=min_intensity)
                
                if i == 0:
                    pcd = pcd_current
                else:
                    pcd += pcd_current

            # Update timestamps_infos
            timestamps_not_cropped = timestamps_all[not_cropped]
            timestamps_registered_pointcloud = np.concatenate([timestamps_registered_pointcloud, timestamps_not_cropped])
            timestamps_infos = np.concatenate([timestamps_infos, 
                                            np.hstack((timestamps_all.reshape(-1, 1),
                                                        not_cropped.reshape(-1, 1),
                                                        np.ones((len(timestamps_all), 1), dtype=int) * file_index))
                                            ], axis=0)
            
            # Post-registration colorization will be done after all scans are registered
            if save_gt and lidar_data["is_key_frame"] and gt_pointcloud is not None and gt_pointcloud.labels is not None:
                point_segmasks_gt = np.concatenate([point_segmasks_gt, 
                                                    np.hstack((timestamps_all.reshape(-1, 1),
                                                                gt_pointcloud.labels.reshape(-1, 1),
                                                                gt_pointcloud.labels.reshape(-1, 1)))], axis=0) # 3rd column is for instances (not used)
            
            if save_sensor_positions:
                if i == 0:
                    sensor_positions = np.concatenate([sensor_positions, np.zeros((num_points_not_cropped, 3))], axis=0)
                else:
                    sensor_positions = np.concatenate([sensor_positions, np.tile(lidar_pose[:3, 3], (num_points_not_cropped, 1))], axis=0)
        
        # Post-registration colorization
        colors = None
        if not recompute_gt_only and colorize_with_cameras:
            logging.info("Starting post-registration colorization...")
            
            lidar_tokens_colorize = lidar_tokens[::freq_colorize]
            
            # Get the final registered points
            points_colorize = np.asarray(pcd.points)
            num_points = len(points_colorize)
            
            # Generate depth maps for occlusion 
            depth_maps_folder = os.path.join(save_folder, "depth_maps")
            
            camera_poses, camera_intrinsics, depth_filenames = generate_real_camera_poses(
                lidar_tokens_colorize, 
                camera_channels, 
                nusc, 
                nuscenes_path, 
                ref_lidar_data=lidar_data_ref
                )

            already_done = 0
            if os.path.exists(depth_maps_folder):
                already_done = len([f for f in os.listdir(depth_maps_folder) if f.endswith('.png')])
                logging.info('Already done %s / %s',already_done, len(lidar_tokens_colorize))
            if already_done < len(lidar_tokens_colorize):
                # generate_depth_maps_nuscenes(points_colorize, 
                #         np.asarray(pcd.colors), 
                #         camera_poses, 
                #         camera_intrinsics, 
                #         depth_filenames, 
                #         depth_maps_folder, 
                #         point_size=colorize_depth_map_point_size
                #         depth_encoding_max_depth=50, 
                #         depth_encoding_num_bins=5000, 
                #         depth_encoding_max_bits=2**16, 
                #     )
                args_render = (
                    points_colorize, 
                    np.asarray(pcd.colors), 
                    camera_poses, 
                    camera_intrinsics, 
                    depth_filenames, 
                    depth_maps_folder, 
                    colorize_depth_map_point_size
                )

                if subproc:
                    render_process = multiprocessing.Process(target=generate_depth_maps_nuscenes,
                                                            args=args_render)
                    render_process.start()
                    render_process.join()
                else:
                    generate_depth_maps_nuscenes(*args_render)

            # Convert to torch tensor
            points_colorize_tensor = torch.tensor(points_colorize, dtype=torch.float32, device=device_project, requires_grad=False)
            
            # Initialize colors and counts arrays
            colors = torch.zeros((num_points, 3), dtype=torch.float32, device=device_project, requires_grad=False)
            counts = torch.zeros((num_points,), dtype=torch.float32, device=device_project, requires_grad=False)
            priorities = torch.zeros((num_points,), dtype=torch.int32, device=device_project, requires_grad=False)
            distances = torch.ones((num_points,), dtype=torch.int32, device=device_project, requires_grad=False) * 50.0
            
            # Define priorities for cameras to handle multiple cameras seeing the same point. e.g. CAM_BACK seem of lesser quality than CAM_FRONT
            priorities_dict = {
                "CAM_FRONT": 6,
                "CAM_FRONT_LEFT": 5,
                "CAM_FRONT_RIGHT": 5,
                "CAM_BACK_LEFT": 5,
                "CAM_BACK_RIGHT": 5,
                "CAM_BACK": 5,
            }
            
            
            # Loop through all frames to colorize the registered pointcloud
            for i, lidar_token in enumerate(tqdm(lidar_tokens_colorize, desc="Colorizing with cameras")):
                # Transform registered points to current frame's coordinate system
                # First transform from registered frame to world frame, then to current frame
                ref_lidar_pose_tensor = torch.tensor(lidar_poses[0], dtype=torch.float32, device=device_project)
                points_world_tensor = torch.mm(ref_lidar_pose_tensor, torch.hstack((points_colorize_tensor, torch.ones((num_points, 1), dtype=torch.float32, device=device_project))).T).T[:, :3]
                
                # Get current frame's lidar pose
                current_lidar_data = nusc.get('sample_data', lidar_token)
                current_lidar_pose = get_T_lidar_to_ref_lidar(nusc, current_lidar_data, lidar_data_ref)
                current_lidar_pose_tensor = torch.tensor(current_lidar_pose, dtype=torch.float32, device=device_project)
                
                # Transform to current frame's coordinate system
                points_current_frame_tensor = torch.mm(torch.linalg.inv(current_lidar_pose_tensor), torch.hstack((points_world_tensor, torch.ones((num_points, 1), dtype=torch.float32, device=device_project))).T).T[:, :3]
                
                # Colorize points in current frame
                frame_colors, frame_counts = colorize_pointcloud_gpu(
                    points=points_current_frame_tensor,
                    nusc=nusc,
                    lidar_token=lidar_token,
                    nuscenes_path=nuscenes_path,
                    camera_channels=camera_channels,
                    min_depth=colorize_min_depth,
                    max_depth=colorize_max_depth,
                    depth_margin=colorize_depth_margin,
                    depth_maps_folder=depth_maps_folder,
                    device=device_project,
                    priorities_dict=priorities_dict,
                    priorities=priorities,
                    distances=distances
                )
                
                # Accumulate colors and counts
                valid_colors = frame_counts > 0
                if torch.any(valid_colors):
                    colors[valid_colors] = frame_colors[valid_colors]
                    counts[valid_colors] += frame_counts[valid_colors]
                    # priorities[valid_colors] = priorities[valid_colors]
            
                # if (i % 350) == 0 and i > 0:
                # if i == len(lidar_tokens_colorize)-1:
                #     # debug by plotting priorities and colors
                #     pcd_debug = o3d.geometry.PointCloud()
                #     pcd_debug.points = o3d.utility.Vector3dVector(points_current_frame_tensor.cpu().numpy())
                    
                #     # Map priorities (0-6) to distinct RGB colors
                #     priority_color_map = np.array([
                #         [0, 0, 1],      # 0: Blue
                #         [0, 1, 0],      # 1: Green
                #         [1, 0, 0],      # 2: Red
                #         [1, 0.5, 0],    # 3: Orange
                #         [1, 1, 0],      # 4: Yellow
                #         [0.5, 0, 0.5],  # 5: Purple
                #         [0, 1, 1],      # 6: Cyan
                #     ])
                #     priorities_np = priorities.cpu().numpy().astype(int)
                #     priorities_colors = priority_color_map[np.clip(priorities_np, 0, 6)]
                    
                    # Optionally blend with actual colors for visualization
                    # if np.max(colors.cpu().numpy()) > 0:
                    #     norm_colors = colors.cpu().numpy() / np.max(colors.cpu().numpy())
                    #     priorities_colors = priorities_colors * 0.8 + norm_colors * 0.2

                    # pcd_debug.colors = o3d.utility.Vector3dVector(priorities_colors)
                    # o3d.visualization.draw_geometries([pcd_debug])
                # if i == len(lidar_tokens_colorize)-1:
                #     # debug by plotting binned distance colors
                #     pcd_debug = o3d.geometry.PointCloud()
                #     pcd_debug.points = o3d.utility.Vector3dVector(points_current_frame_tensor.cpu().numpy())
                    
                #     # Define bin edges and colors
                #     bin_edges = np.arange(0, 33, 3)  # 0,3,6,...,30
                #     bin_colors = np.array([
                #         [1, 0, 0],      # 0-3m: Red
                #         [0, 0, 1],      # 3-6m: Blue
                #         [0, 1, 0],      # 6-9m: Green
                #         [1, 1, 0],      # 9-12m: Yellow
                #         [1, 0, 1],      # 12-15m: Magenta
                #         [0, 1, 1],      # 15-18m: Cyan
                #         [1, 0.5, 0],    # 18-21m: Orange
                #         [0.5, 0, 0.5],  # 21-24m: Purple
                #         [0.5, 0.5, 0.5],# 24-27m: Gray
                #         [0, 0, 0],      # 27-30m: Black
                #     ])
                #     distances_np = distances.cpu().numpy()
                #     # Bin the distances
                #     bin_indices = np.digitize(distances_np, bin_edges) - 1
                #     bin_indices = np.clip(bin_indices, 0, len(bin_colors)-1)
                #     debug_colors = bin_colors[bin_indices]
                    
                #     # Optionally blend with actual colors for visualization
                #     if np.max(colors.cpu().numpy()) > 0:
                #         norm_colors = colors.cpu().numpy() / np.max(colors.cpu().numpy())
                #         debug_colors = debug_colors * 0.8 + norm_colors * 0.2

                #     pcd_debug.colors = o3d.utility.Vector3dVector(debug_colors)
                #     o3d.visualization.draw_geometries([pcd_debug])
            # Average colors by counts (avoid division by zero)
            # valid_counts = counts > 0
            # if torch.any(valid_counts):
            #     colors[valid_counts] /= counts[valid_counts].unsqueeze(1)
            
            # Convert back to numpy
            colors = colors.cpu().numpy()
            counts = counts.cpu().numpy()
            
            logging.info(f"Post-registration colorization completed. Colors shape: {colors.shape}")
            colored_points = np.sum(counts > 0)
            logging.info(f"Colored points: {colored_points}/{num_points} ({colored_points/num_points*100:.1f}%)")
        
        if not recompute_gt_only:
            # save pointclouds
            points = np.asarray(pcd.points)
            intensities = np.asarray(pcd.colors)[:,0]

            logging.info("Min intensity = %s, Max intensity = %s", intensities.min(), intensities.max())

            if blur_with_random_intensity:
                k = np.random.choice(blur_random_values)
                logging.info('Randomly selected k=%s from %s',k, blur_random_values)
            else:
                k = blur_intensity_k
            if k > 1:
                # points, intensities = filter_intensity_knn_batch(points, intensities, k=blur_intensity_k)
                filter_intensity_knn_batch(points, intensities, k=k)

                ##### # Compute the histogram
                hist, bins = np.histogram(intensities, bins=256, range=(intensities.min(), intensities.max()), density=False)
                # Compute the cumulative distribution function (CDF)
                cdf = hist.cumsum()
                # Normalize the CDF
                # cdf_normalized = (cdf - cdf.min()) * max_intensity / (cdf.max() - cdf.min())
                cdf_normalized = (cdf - cdf.min()) * (max_intensity - min_intensity) / (cdf.max() - cdf.min()) + min_intensity
                logging.info('cdf_normalized.min()=\n%s',cdf_normalized.min())
                logging.info('cdf_normalized.max()=\n%s',cdf_normalized.max())
                cdf_normalized = cdf_normalized.astype('uint8')    
                # Use linear interpolation of the CDF to find new pixel values
                intensities = np.interp(intensities.flatten(), bins[:-1], cdf_normalized) / 255.0
            

            # save las file
            # logging.info('len(points)=\n%s',len(points))
            # logging.info('timestamps_infos[:,1].sum()=%s / %s total',timestamps_infos[:,1].sum(), len(timestamps_infos))
            # logging.info('colors.shape=%s',colors.shape)
            # logging.info('points.shape=%s',points.shape)
            # logging.info('intensities.shape=%s',intensities.shape)
            logging.info("Saving registered pointcloud to %s", "/".join(registered_pointcloud_path.split("/")[-2:]))
            intensities = intensities * 65535
            intensities = intensities.astype(int) # store the colors as intensities for memory

            if colorize_with_cameras and colors is not None:
                # fill uncolored points with intensity
                # Find points that have no color (all channels are 0)
                uncolored_points = np.all(colors == 0, axis=1)
                # Fill those points with intensity values for all channels
                colors[uncolored_points, :] = intensities[uncolored_points].reshape(-1, 1).astype(float) / 65535.0

            save_las_file(registered_pointcloud_path=registered_pointcloud_path,
                          points=points,
                          colors=colors,
                          timestamps=timestamps_registered_pointcloud,
                          intensities=intensities)

            # save lidar poses
            lidar_poses = np.array(lidar_poses)
            np.save(lidar_poses_path, lidar_poses)

            # save timestamps_infos
            np.savez_compressed(timestamps_infos_path, timestamps_infos=timestamps_infos)


        if save_gt:
            logging.info("Saving gt to %s", "/".join(point_segmasks_gt_path.split("/")[-2:]))
            np.save(point_segmasks_gt_path, point_segmasks_gt)
        if not recompute_gt_only and save_sensor_positions:
            np.savez_compressed(sensor_positions_path, sensor_positions)
            
        if plot :
            try:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
                # Try different visualization methods for different open3d versions
                try:
                    o3d.visualization.draw_geometries([coordinate_frame, pcd])
                except AttributeError:
                    # For newer versions of open3d
                    try:
                        o3d.visualization.draw([coordinate_frame, pcd])
                    except AttributeError:
                        # For even newer versions
                        o3d.visualization.draw_geometries_with_animation_callback([coordinate_frame, pcd], lambda vis: None)
            except Exception as e:
                logging.warning(f"Visualization failed: {e}")
                logging.info("Skipping visualization") 

        # if log_path is not None:
        #     with open(log_path, "a") as f:
        #         f.write("{}\n".format(save_folder))
        write_log(log_path, save_folder)
    return nusc

