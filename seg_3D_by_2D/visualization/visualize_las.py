# Visualize any las in open3D
# Usage: python visualize_las.py <las_file>

import copy
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import open3d as o3d
import laspy
import sys
import os
import numpy as np
import logging
import argparse
import json
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import os
# os.environ['LIBGL_DEBUG'] = 'verbose'
# os.environ['DRI_PRIME'] = '1'
from seg_3D_by_2D.core.classes_dicts import color_palette_float
from seg_3D_by_2D.utils.utils import read_las_file


def map_values_to_colors(values, vmin=0.0, vmax=100.0, colormap='coolwarm'):
    """
    Map values to colors using matplotlib colormap.
    
    Args:
        values: numpy array of values to map
        vmin: minimum value for color mapping (default: 0.0, maps to blue)
        vmax: maximum value for color mapping (default: 100.0, maps to red)
        colormap: matplotlib colormap name (default: 'coolwarm' for blue-to-red)
    
    Returns:
        numpy array of RGB colors in range [0, 1]
    """
    # Normalize values to [0, 1] range based on vmin and vmax
    normalized_values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    
    # Get colormap
    cmap = plt.cm.get_cmap(colormap)
    
    # Map normalized values to colors
    colors = cmap(normalized_values)
    
    # Return only RGB (remove alpha channel)
    return colors[:, :3]


def load_camera_parameters(json_path):
    """Load camera parameters from a JSON file."""
    logging.info(f"Loading camera parameters from {json_path}")
    try:
        with open(json_path, 'r') as f:
            camera_params_dict = json.load(f)
        
        # Create Open3D camera parameters object
        camera_params = o3d.camera.PinholeCameraParameters()
        
        # Set extrinsic matrix (4x4 transformation matrix)
        extrinsic = np.array(camera_params_dict["extrinsic"]).reshape(4, 4).T
        camera_params.extrinsic = extrinsic
        
        # Set intrinsic parameters
        intrinsic_dict = camera_params_dict["intrinsic"]
        width = intrinsic_dict["width"]
        height = intrinsic_dict["height"]
        
        # Parse intrinsic matrix (3x3 camera matrix)
        intrinsic_matrix = np.array(intrinsic_dict["intrinsic_matrix"]).reshape(3, 3)
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[2, 0]
        cy = intrinsic_matrix[2, 1]
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        camera_params.intrinsic = intrinsic
        
        logging.info("Camera parameters loaded successfully")
        logging.info(f"Camera intrinsic matrix:\n{intrinsic_matrix}")
        logging.info(f"Camera extrinsic matrix:\n{extrinsic}")
        logging.info(f"Camera width: {width}, height: {height}")
        logging.info(f"Camera focal lengths: fx={fx}, fy={fy}")
        return camera_params
    
    except Exception as e:
        logging.error(f"Failed to load camera parameters: {str(e)}")
        return None


def visualize_las(args):
    if args.pointcloud_path is not None:
        input_file_paths = args.pointcloud_path
        if isinstance(input_file_paths, str):
            input_file_paths = [input_file_paths]
    else:
        input_file_paths = [] 
    logging.info('input_file_paths=%s',input_file_paths)
    # Get the points
    plot_list = []
    offsets = np.array([0, 0, 0])
    for i, file_path in enumerate(input_file_paths):
        if file_path.endswith('.ply'): # For meshes
            logging.info('Loading mesh %s/%s, path=%s', i+1, len(input_file_paths), file_path)
            mesh = o3d.io.read_triangle_mesh(file_path)
            # vertices = np.asarray(mesh.vertices)
            # vertex_colors = np.asarray(mesh.vertex_colors)
            # vertex_normals = np.asarray(mesh.vertex_normals)
            # vertex_colors = vertex_normals
            # mesh.vertex_colors = copy.deepcopy(mesh.vertex_normals)
            # logging.info('vertices.shape=\n%s',vertices.shape)
            # logging.info('vertex_colors.shape=\n%s',vertex_colors.shape)
            # logging.info('vertex_colors.dtype=%s',vertex_colors.dtype)
            # logging.info('vertex_colors.max()=%s',vertex_colors.max())
            # logging.info('vertex_colors.min()=%s',vertex_colors.min())
            # logging.info('vertex_normals.shape=\n%s',vertex_normals.shape)
            # logging.info('vertex_normals.dtype=%s',vertex_normals.dtype)
            # logging.info('vertex_normals.max()=%s',vertex_normals.max())
            # logging.info('vertex_normals.min()=%s',vertex_normals.min())
            # compute normals for mesh
            if args.compute_normals:
                mesh.compute_vertex_normals()
            else:
                pcd = mesh
            pcd = mesh
            points = np.asarray(pcd.vertices)
            if args.mesh_intensities_path is not None:
                intensities = np.load(args.mesh_intensities_path)["intensities"]
                colors = np.tile(intensities[:, np.newaxis], (1, 3))
                logging.info('colors.shape=\n%s',colors.shape)
                pcd.vertex_colors = o3d.utility.Vector3dVector(colors)
            plot_list.append(pcd)
            
        elif file_path.endswith('.las'):
            logging.info('Loading pointcloud %s/%s, path=%s', i+1, len(input_file_paths), file_path)
            points, colors, timestamps, intensity = read_las_file(file_path, use_rgb_colors=not args.use_intensity_colors)
            logging.info('Min dimensions=%s',np.min(points, axis=0))
            logging.info('Max dimensions=%s',np.max(points, axis=0))
            logging.info('points.shape=\n%s',points.shape)
            logging.info('np.max(colors, axis=0)=\n%s',np.max(colors, axis=0))
            logging.info('np.min(colors, axis=0)=\n%s',np.min(colors, axis=0))
            
            # if args.other_pointcloud_paths is not None:
            #     plot_second = True
            #     other_points, other_colors, other_timestamps, other_intensity = read_las_file(args.other_pointcloud_paths)

            # Load masks
            if args.masks is not None:
                masks = np.load(args.masks)
                logging.info('masks.dtype=\n%s',masks.dtype)
                if masks.dtype == np.int64 or masks.dtype == np.int32:
                    if len(masks.shape) == 1:
                        logging.info("Integer mask of shape %s",masks.shape)
                        mask_indices = np.unique(masks)
                        logging.info('mask_indices=\n%s',mask_indices)
                        color_map = np.random.rand(len(mask_indices), 3)
                        for i, mask_index in enumerate(mask_indices):
                            if mask_index == 0:
                                continue
                            colors[masks==mask_index] = color_map[i]
                    elif len(masks.shape) == 2:
                            logging.info('masks[:5,:]=\n%s',masks[:5,:])
                            masks = masks[:,1]
                            logging.info("Integer mask of shape %s",masks.shape)
                            mask_indices = np.unique(masks)
                            logging.info('mask_indices=\n%s',mask_indices)
                            color_map = np.random.rand(len(mask_indices), 3)
                            for i, mask_index in enumerate(mask_indices):
                                if mask_index == 0:
                                    continue
                                colors[masks==mask_index] = color_map[i]
                else:
                    logging.info('masks.shape=\n%s',masks.shape)
                    # mask_one_dim = len(masks.shape) == 1
                    if len(masks.shape) == 1:
                        logging.info('min=%s, max=%s',np.min(masks),np.max(masks))
                        masks = masks / np.max(masks)
                        colors = np.vstack((masks, masks, masks)).T.astype(float)
                    else:
                        if masks.shape[1] == 1:
                            logging.info("float mask of shape %s",masks.shape)
                            # normalize between 0 and 1
                            logging.info("Normalizing masks between 0 and 1")
                            logging.info('min=%s, max=%s',np.min(masks),np.max(masks))
                            masks = masks / np.max(masks)
                            colors = np.vstack((masks, masks, masks)).T.astype(float)
                        
                        elif masks.shape[1] >= 2:
                            masks = masks[:,args.cls]
                            logging.info("float mask of shape %s",masks.shape)
                            # normalize between 0 and 1
                            logging.info("Normalizing masks between 0 and 1")
                            logging.info('min=%s, max=%s',np.min(masks),np.max(masks))
                            masks = masks / np.max(masks)
                            colors = np.vstack((masks, masks, masks)).T.astype(float)
                        
            if args.point_segmasks is not None:
                
                color_map = color_palette_float    
                point_segmasks_list = [args.point_segmasks] if isinstance(args.point_segmasks, str) else args.point_segmasks
                logging.info('point_segmasks_list=%s',point_segmasks_list)        
                point_segmasks = np.load(point_segmasks_list[0])
                
                # Deal with filtered registered pointclouds (some points may have been removed compared to GT segmask)
                if len(point_segmasks) != len(points):
                    logging.info('Lengh of point_segmasks (%s) does not match the number of points (%s)', len(point_segmasks), len(points))
                    logging.info('Attempting to get timestamps info.')
                    logging.info('point_segmasks.shape=\n%s',point_segmasks.shape)
                    timestamps_infos_path = os.path.join(os.path.dirname(point_segmasks_list[0]), args.timestamps_infos_filename)
                    try:
                        timestamps_infos = np.load(timestamps_infos_path)["timestamps_infos"]
                        # logging.info('timestamps_infos.files=\n%s',timestamps_infos.files)
                        logging.info('timestamps_infos.shape=\n%s',timestamps_infos.shape)
                    except FileNotFoundError:
                        logging.info('No timestamps_infos file found at %s',timestamps_infos_path)
                        logging.info('Exiting...')
                        sys.exit()
                    except Exception as e:
                        logging.info('An unexpected error occurred: %s', str(e))
                        logging.info('Exiting...')
                        sys.exit()

                    is_registered_mask = timestamps_infos[:,1].astype(bool)
                    logging.info('is_registered_mask.shape=\n%s',is_registered_mask.shape)
                    logging.info('is_registered_mask.sum()=\n%s',is_registered_mask.sum())
                    logging.info('point_segmasks[is_registered_mask]=\n%s',len(point_segmasks[is_registered_mask]))
                    point_segmasks = point_segmasks[is_registered_mask]
                    assert len(point_segmasks) == len(points), f"Length of point_segmasks ({len(point_segmasks)}) does not match the number of points ({len(points)})"
                
                logging.info('point_segmasks.shape=\n%s',point_segmasks.shape)
                if len(point_segmasks.shape)==1:
                    sem_segmask = point_segmasks
                    point_segmasks = np.zeros_like(sem_segmask)
                else:
                    sem_segmask = point_segmasks[:,1]
                    sem_classes = np.unique(sem_segmask)
                    point_segmasks = point_segmasks[:,2]
                logging.info('np.unique(sem_segmask)=\n%s',np.unique(sem_segmask))
                
                unique_classes = sorted(np.unique(sem_segmask))
                # dict_classes_to_inds = {sem_class: i for i, sem_class in enumerate(unique_classes)}
                
                logging.info('len(point_segmasks)=%s',len(point_segmasks))
                # Original coloring logic for semantic classes
                for sem_class in unique_classes:
                    # logging.info('color_map[dict_classes_to_inds[sem_class]]=%s',color_map[dict_classes_to_inds[sem_class]])
                    colors[sem_segmask == sem_class] = colors[sem_segmask == sem_class]*args.alpha + color_map[sem_class]*(1-args.alpha)
                    num_points = np.sum(sem_segmask == sem_class)
                    logging.info("Class %s has %s proportion of points, with %s points",sem_class, num_points / len(sem_segmask), num_points)
                
                # Check if we need to visualize errors between prediction and ground truth
                if args.show_errors and len(point_segmasks_list) >= 2:
                    logging.info('Showing errors between prediction and ground truth')
                    # Load the second segmask (ground truth)
                    gt_point_segmask = np.load(point_segmasks_list[1])
                    
                    # Extract semantic classes
                    if len(gt_point_segmask.shape) == 1:
                        gt_sem_segmask = gt_point_segmask
                    else:
                        gt_sem_segmask = gt_point_segmask[:,1]
                    
                    # Handle potential size differences
                    if len(gt_sem_segmask) != len(points):
                        logging.info('Lengh of gt_sem_segmask (%s) does not match the number of points (%s)', 
                                    len(gt_sem_segmask), len(points))
                        timestamps_infos_path = os.path.join(os.path.dirname(args.point_segmasks[1]), 
                                                            args.timestamps_infos_filename)
                        try:
                            timestamps_infos = np.load(timestamps_infos_path)["timestamps_infos"]
                            logging.info('timestamps_infos.shape=\n%s',timestamps_infos.shape)
                            is_registered_mask = timestamps_infos[:,1].astype(bool)
                            gt_sem_segmask = gt_sem_segmask[is_registered_mask]
                            assert len(gt_sem_segmask) == len(points)
                        except Exception as e:
                            logging.info('Error loading timestamps info: %s', str(e))
                            logging.info('Cannot compare prediction with ground truth, skipping error visualization')
                            gt_sem_segmask = None
                    
                    # Compare predictions with ground truth and highlight errors in red
                    if gt_sem_segmask is not None:
                        # Find where predictions don't match ground truth
                        error_mask = sem_segmask != gt_sem_segmask
                        logging.info('Number of errors: %s out of %s points (%.2f%%)', 
                                    np.sum(error_mask), 
                                    len(error_mask), 
                                    100 * np.sum(error_mask) / len(error_mask))
                        
                        # Color errors in red
                        colors[error_mask] = (np.array([1, 0, 0]) + colors[error_mask])/2  # Red for errors
                # else:
            
            elif args.cumulative_logits_path is not None:
                assert args.cls is not None, "--cls must be specified to plot logits"
                if args.cumulative_logits_path.endswith('.npy'):
                    logits = np.load(args.cumulative_logits_path)
                elif args.cumulative_logits_path.endswith('.npz'):
                    logits = np.load(args.cumulative_logits_path)["logits"]
                logging.info('logits.dtype=\n%s',logits.dtype)
                logging.info('logits.shape=\n%s',logits.shape)
                cls = args.cls
                logging.info('class = %s',cls)
                ## 
                logging.info('logits.max()=%s',logits.max())
                logging.info('logits.min()=%s',logits.min())
                mask_cls = logits[:,cls]
                
                # Use absolute values and map to color gradient
                logging.info("Using absolute logit values with color mapping")
                logging.info('min=%s, max=%s',np.min(mask_cls),np.max(mask_cls))
                logging.info('Color range: %s to %s (colormap: %s)', args.logits_vmin, args.logits_vmax, args.logits_colormap)
                
                if args.logits_ponderated:
                    assert args.cumulative_votes_path is not None, "Ponderated logits require --cumulative_votes_path"
                    votes = np.load(args.cumulative_votes_path)["votes"]
                    logging.info('votes.shape=%s',votes.shape)
                    logging.info('np.unique(np.sum(votes, axis=1))=%s',np.unique(np.sum(votes, axis=1)))
                    mask_cls[mask_cls > 0] = mask_cls[mask_cls > 0] / np.sum(votes[mask_cls > 0], axis=1)
                    logging.info('mask_cls.shape=%s',mask_cls.shape)
                    logging.info('mask_cls.max()=%s',mask_cls.max())
                    logging.info('mask_cls.min()=%s',mask_cls.min())
                    if args.logits_log:
                        mask_cls[mask_cls > 0] = mask_cls[mask_cls > 0] * np.log(np.sum(votes[mask_cls > 0], axis=1))
                    logging.info('Using adaptive color range since ponderated logits are used')
                    args.logits_vmax = np.max(mask_cls)
                    args.logits_vmin = np.min(mask_cls)

                # Map values to colors using the specified range
                colors = map_values_to_colors(
                    mask_cls, 
                    vmin=args.logits_vmin, 
                    vmax=args.logits_vmax,
                    colormap=args.logits_colormap
                )



                # debug
                # pred = np.argmax(logits[:, 2:], axis=1) + 2
                # for i in range(logits.shape[1]):
                #     color = color_palette_float[i]
                #     logging.info('class %s, color=%s',i,color)
                #     colors[pred == i] = color
                # filter_min_values = np.where(np.max(logits[:, 2:], axis=1) <= 0.0)[0]
                # logging.info('len(filter_min_values)=%s',len(filter_min_values))
                
                

                # colors[filter_min_values] = 0
                
                # Filter out points outside of 101.746120347296,
                # 65.558525292220807,
                # -56.532043940225634,
                # min_x = -105
                # min_y = 82
                # margin = 10

                # mask = np.logical_and(points[:,0] > min_x, points[:,0] < min_x + margin)
                # mask = np.logical_and(mask, np.logical_and(points[:,1] > min_y, points[:,1] < min_y + margin))     
                # points = points[mask]
                # colors = colors[mask]
                
                # logits = logits[mask]
                # pred = pred[mask]

                # logging.info('sum(mask)=%s',sum(mask))
                # logging.info('logits.shape=%s',logits.shape)
                # logging.info('pred.shape=%s',pred.shape)

                # for i in range(logits.shape[1]):
                #     logging.info('class %s',i)
                #     if logits[pred==i].shape[0] == 0:
                #         continue
                #     logging.info('np.min(logits[pred==i], axis=0).min()=%s',np.min(logits[pred==i], axis=0))
                #     logging.info('np.max(logits[pred==i], axis=0).max()=%s',np.max(logits[pred==i], axis=0))
                # mask_cls = np.where(mask_cls > 0, 1, 0)
                # colors = np.vstack((mask_cls, mask_cls, np.zeros_like(mask_cls))).T.astype(float)
                # Set points with max logits <= 0 to red
                # MAX_LOGITS=10.0
                # max_logits = np.max(logits, axis=1)
                # logging.info('sum(max_logits <= 0.0)=%s',sum(max_logits <= MAX_LOGITS))
                # colors[max_logits <= MAX_LOGITS] = np.array([1, 0, 0])
            elif args.cumulative_votes_path is not None:
                assert args.cls is not None, "--cls must be specified to plot votes"
                if args.cumulative_votes_path.endswith('.npy'):
                    votes = np.load(args.cumulative_votes_path)
                elif args.cumulative_votes_path.endswith('.npz'):
                    votes = np.load(args.cumulative_votes_path)["votes"]
                logging.info('votes.dtype=\n%s',votes.dtype)
                logging.info('votes.shape=\n%s',votes.shape)
                cls = args.cls
                if cls == -1:
                    logging.info('--cls -1, Summing votes over all classes')
                    mask_cls = np.sum(votes, axis=1)
                else:
                    logging.info('class = %s',cls)
                    mask_cls = votes[:,cls]

                logging.info("float mask of shape %s",votes.shape)
                # normalize between 0 and 1
                logging.info("Normalizing votes between 0 and 1")
                logging.info('min=%s, max=%s',np.min(mask_cls),np.max(mask_cls))
                mask_cls = mask_cls / np.max(mask_cls)
                logging.info('np.unique(mask_cls)=\n%s',np.unique(mask_cls))
                colors = np.vstack((mask_cls, mask_cls, np.zeros_like(mask_cls))).T.astype(float)
                # set to red the points with votes = 0
                # colors[mask_cls == 0] = np.array([1,0,0])
            
            elif args.pcd_normals and args.sensor_positions_path is not None:
                
                import torch
                from nksr.utils import estimate_normals
                import nksr.ext as ext
                import sys
                # import nksr.ext as ext
                logging.info("PYTHONPATH: %s", os.environ.get("PYTHONPATH"))
                logging.info("sys.path: %s", sys.path)
                logging.info("nksr.ext file: %s", ext.__file__)
                logging.info("nksr.ext contents: %s", sorted(dir(ext)))
                
                logging.info('Computing normals for pointcloud using NKSR.')
                # Convert points and sensor positions to torch tensors
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                points_tensor = torch.from_numpy(points).float().to(device)
                sensor_positions = np.load(args.sensor_positions_path)["arr_0"]
                sensor_positions_tensor = torch.from_numpy(sensor_positions).float().to(device)
                # Estimate normals using NKSR
                normals, indices = estimate_normals(
                    xyz=points_tensor,
                    sensor=sensor_positions_tensor,
                    knn=64,  # Number of neighbors for normal estimation
                    drop_threshold_degrees=85.0,  # Angle threshold for filtering normals
                    backend='nksr'  # Use NKSR's native normal estimation
                )
                # Keep only points with valid normals
                valid_points = points[indices.cpu().numpy()]
                normals_np = normals.cpu().numpy()
                # Update points and map normals from [-1,1] to [0,1] for visualization as colors
                points = valid_points
                colors = (normals_np * 0.5 + 0.5)  # Map from [-1,1] to [0,1] range
                # Update pcd object if it's being kept
                # if keep_pcd:
                #     pcd = o3d.geometry.PointCloud()
                #     pcd.points = o3d.utility.Vector3dVector(points)
                #     pcd.colors = o3d.utility.Vector3dVector(colors)
                logging.info('Normal computation complete with NKSR.')
                logging.info('normals.shape=%s', normals_np.shape)


            # Create the point cloud
            logging.info('Creating point cloud...')
            pcd = o3d.geometry.PointCloud()
            # Translate to origin for display
            
            # offsets = [0, 0, 0]
            if i == 0 and not args.dont_translate_to_centroid:
                # logging.info("Centering points to mean for display without saturation.")
                logging.info("Setting first point to origin for display without saturation.")
                offsets = points[0] #np.mean(points, axis=0)
            points = points - offsets
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            if args.downsample:
                logging.info('Downsampling point cloud...')
                downpcd = pcd.voxel_down_sample(voxel_size=0.05)
                pcd = downpcd

            plot_list.append(pcd)

    # Visualize the point cloud
    if not args.no_frame:
        logging.info("Displaying %sm reference frame", args.frame_size )
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.frame_size, origin=[0, 0, 0])
        plot_list.append(coordinate_frame)
    
    
    set_offsets_with_traj = False
    if len(input_file_paths) == 0 and (args.trajectory_filenames is not None or args.trajectory_paths is not None) :
        set_offsets_with_traj = True
    if args.trajectory_filenames is not None:
        logging.info('Adding trajectories by filenames...')
        for i, trajectory_filename in enumerate(args.trajectory_filenames):
            color = color_palette_float[i+1]
            trajectory_path = os.path.join(os.path.dirname(args.pointcloud_path), trajectory_filename)
            if not os.path.exists(trajectory_path):
                trajectory_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.pointcloud_path))), trajectory_filename)
            assert os.path.exists(trajectory_path), "Trajectory file not found at %s" % trajectory_path
            logging.info('Loading trajectory from %s',trajectory_path)
            trajectory_poses = np.load(trajectory_path)
            if i == 0 and set_offsets_with_traj:
                offsets = trajectory_poses[0,:3,3]
            pcd_trajectory, dir_lineset = get_trajectory_plots(trajectory_poses, offset=offsets, color=color)
            plot_list.append(pcd_trajectory)
            plot_list.append(dir_lineset)        

    if args.trajectory_paths is not None:
        logging.info('Adding trajectories by paths...')
        for i, trajectory_path in enumerate(args.trajectory_paths):
            color = color_palette_float[i+1]
            assert os.path.exists(trajectory_path), "Trajectory file not found at %s" % trajectory_path
            logging.info('Loading trajectory from %s',trajectory_path)
            trajectory_poses = np.load(trajectory_path)
            if i == 0 and set_offsets_with_traj:
                offsets = trajectory_poses[0,:3,3].copy()
            positions = trajectory_poses[:, :3, 3]
            logging.info('positions.max()=%s',positions.max(axis=0))
            logging.info('positions.min()=%s',positions.min(axis=0))
            logging.info('trajectory_poses.shape=\n%s',trajectory_poses.shape)
            logging.info('offsets=%s',offsets)
            pcd_trajectory, dir_lineset = get_trajectory_plots(trajectory_poses, offset=offsets.copy(), color=color)
            plot_list.append(pcd_trajectory)
            # if os.path.basename(trajectory_path).startswith("kalman"):
            plot_list.append(dir_lineset)
                
            logging.info('offsets=%s',offsets)
    
    logging.info('plot_list=%s',plot_list)
    
    # Create visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometry to the scene
    for geom in plot_list:
        vis.add_geometry(geom)
    
    # Set camera parameters if provided
    if args.camera_json:
        camera_params = load_camera_parameters(args.camera_json)
        if camera_params:
            control = vis.get_view_control()
            control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
            logging.info("Applied camera parameters from JSON file")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()
    
    # # If no camera_json was provided, fall back to the default visualization
    # if not args.camera_json:
    #     o3d.visualization.draw_geometries(plot_list, mesh_show_back_face=True)

# Define a function to create direction lines
def create_direction_lines(poses, scale=0.5):
    lines = []
    colors = []
    for pose in poses:
        # Extract the translation part
        t = pose[:3, 3]
        # Extract the rotation part
        R = pose[:3, :3]
        # Define the directions (x, y, z)
        directions = np.eye(3) * scale
        # Compute the end points of the direction lines
        for i in range(3):
            end_point = t + R @ directions[i]
            lines.append([t, end_point])
            color = [0, 0, 0]
            color[i] = 1  # Set the color to red, green, or blue
            colors.append(color)
    lines = np.array(lines).reshape(-1, 2, 3)
    colors = np.array(colors)    
    return lines, colors

def get_trajectory_plots(trajectory_poses, offset=None, color=None):
        """Returns points and lines for a trajectory plot
        Args:
            trajectory_poses ((N,4,4) np.array): Poses of the trajectory
            offset ((3,) np.array): Offset to apply to the trajectory"""
        assert trajectory_poses.shape[1] == 4 and trajectory_poses.shape[2] == 4, "Trajectory must be poses of shape (N, 4, 4)"
        pcd_trajectory = o3d.geometry.PointCloud()
        if offset is not None:
            points_trajectory = trajectory_poses[:,:3,3] - offset # Translate to origin
        else:
            points_trajectory = trajectory_poses[:,:3,3]
        pcd_trajectory.points = o3d.utility.Vector3dVector(points_trajectory)
        if color is not None:
            pcd_trajectory.paint_uniform_color(color)
        trajectory_poses[:,:3,3] = points_trajectory
        dir_lines, colors = create_direction_lines(trajectory_poses, scale=0.5)
        dir_lineset = o3d.geometry.LineSet()
        dir_lineset.points = o3d.utility.Vector3dVector(dir_lines.reshape(-1, 3))
        dir_lineset.lines = o3d.utility.Vector2iVector(np.arange(dir_lines.shape[0] * 2).reshape(-1, 2))
        dir_lineset.colors = o3d.utility.Vector3dVector(np.repeat(colors, 2, axis=0))
        
        return pcd_trajectory, dir_lineset


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--pointcloud_path', help='Pointcloud file path. Supported format are : .las, .ply', nargs='+', type=str)
    parser.add_argument('--downsample', action='store_true', help='Downsample the point cloud')
    parser.add_argument('--masks', default=None, help='numpy array with masks of the pointcloud to plot in random colors')
    parser.add_argument('--cls', default=0, type=int, help='class to plot')
    parser.add_argument('--alpha', default=0.6, type=float, help='alpha for the point cloud colors')
    parser.add_argument('--point_segmasks', default=None, help='numpy array (Nx3) with point instance segmask. columns are : time, sem_class, instance_id', nargs='+')
    parser.add_argument('--show_errors', action="store_true", help='If True, given 2 point_segmasks files given in --point_segmasks (first is pred, second is gt), will show the errors in red')
    parser.add_argument('--timestamps_infos_filename', default='timestamps_infos.npz', help='numpy array (Nx3) with point instance segmask. columns are : time, sem_class, instance_id')
    parser.add_argument('--compute_normals', action="store_true", help='Compute normals for the mesh')
    parser.add_argument('--dont_translate_to_centroid', action="store_true", help='')
    # parser.add_argument('--trajectory_filenames', default=None, help='poses array (Nx4x4) for trajectory plotting')
    parser.add_argument('--trajectory_filenames', default=None, help='filenames to other trajectories to plot', nargs='+', type=str)
    parser.add_argument('--trajectory_paths', default=None, help='paths to other trajectories to plot', nargs='+', type=str)
    parser.add_argument('--other_pointcloud_paths', default=None, help='Plot a second pointcloud in the same frame')
    parser.add_argument('--cumulative_logits_path', default=None, help='Plot cumulative logits for this pointcloud')
    parser.add_argument('--logits_vmin', default=0.0, type=float, help='Minimum value for logits color mapping (blue)')
    parser.add_argument('--logits_vmax', default=100.0, type=float, help='Maximum value for logits color mapping (red)')
    parser.add_argument('--logits_ponderated', default=False, action='store_true', help='Use ponderated logits color mapping (red)')
    parser.add_argument('--logits_log', default=False, action='store_true', help='Use log of logits color mapping (red)')
    parser.add_argument('--logits_colormap', default='coolwarm', type=str, help='Matplotlib colormap for logits (e.g., coolwarm, viridis, plasma)')
    parser.add_argument('--cumulative_votes_path', default=None, help='Plot cumulative votes for this pointcloud')
    parser.add_argument('--mesh_intensities_path', default=None, help='Path to a npz file containing mesh intensities for coloring the mesh')
    parser.add_argument('--frame_size', default=3.0, type=float, help='Size of the reference frame')
    parser.add_argument('--no_frame', action="store_true", help='Do not display the reference frame')
    parser.add_argument('--camera_json', type=str, help='Path to a JSON file containing camera parameters to set the view')
    parser.add_argument('--sensor_positions_path', type=str, help='Path to a JSON file containing camera parameters to set the view', default=None)
    parser.add_argument('--pcd_normals', action="store_true", help='Compute normals for the pointcloud')
    parser.add_argument('--use_intensity_colors', action="store_true", help='Use intensity colors for the pointcloud')

    args = parser.parse_args()
    # input_file_paths = args.pointcloud_path
    # logging.info('args.point_segmask=%s',args.point_segmask)
    logging.info('args.point_segmasks=%s',args.point_segmasks)
    visualize_las(args)