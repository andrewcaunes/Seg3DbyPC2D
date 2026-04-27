"""
Script made by Andrew Caunes.
Prepare datasets such as NuScenes for processing by seg_3D_by_2D.py.
Example use:
python script.py ...
"""
import multiprocessing
import os
from re import L
import shutil
import argparse
import logging

import torch
from tqdm import tqdm
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
import laspy
import pyrender
from scipy.spatial.transform import Rotation as R
from seg_3D_by_2D.utils.utils import save_las_file, filter_intensity_knn_batch, get_unique_file_index_sk
from seg_3D_by_2D.datasets.semantickitti.colorize_pcd_sk import colorize_pointcloud, colorize_pointcloud_gpu, parse_calibration

def add_intensity_to_pointcloud(pcd,
                                intensities, 
                                min_intensity=0, 
                                max_intensity=245, 
                                min_clip=0, 
                                max_clip=255,
                                target_median=79,
                                target_ground_color=30,
                                ground_color=None):
    """Add intensities to point cloud.
    Uniformly normalize intensities between min_intensity and max_intensity after clipping."""
    
    hist, bins = np.histogram(intensities, 
                              bins=256, 
                              range=(intensities.min(), 
                              intensities.max()), 
                              density=False)
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    # Normalize the CDF
    cdf_normalized = (cdf - cdf.min()) * max_intensity / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')    
    # Use linear interpolation of the CDF to find new pixel values
    intensities = np.interp(intensities.flatten(), bins[:-1], cdf_normalized)
    # normalized_intensities = intensities.reshape(-1, 1) / 255.0
    normalized_intensities = intensities.reshape(-1, 1)
    colors = np.zeros((len(intensities), 3))
    colors[:, :] = normalized_intensities.reshape(-1, 1)
    # colors[:, :] = intensities.reshape(-1, 1)
    pcd.colors = o3d.utility.Vector3dVector(colors)


def crop_pointcloud(scan, radius_close=None, box_far=None):
    """
    Removes point too close within a certain radius from origin.
    :param radius: radius below which points are removed
    """
    x_filt_close = np.abs(scan[:, 0]) < radius_close
    y_filt_close = np.abs(scan[:, 1]) < radius_close
    z_filt_close = np.abs(scan[:, 2]) < radius_close
    close = np.logical_and(np.logical_and(x_filt_close, y_filt_close), z_filt_close)
    if box_far is None:
        far = np.zeros(scan.shape[0], dtype=bool)
    else:
        x_min, x_max = box_far[0]
        y_min, y_max = box_far[1]
        z_min, z_max = box_far[2]
        x_filt_far = np.logical_or(scan[:, 0] < x_min, scan[:, 0] > x_max)
        y_filt_far = np.logical_or(scan[:, 1] < y_min, scan[:, 1] > y_max)
        z_filt_far = np.logical_or(scan[:, 2] < z_min, scan[:, 2] > z_max)
        far = np.logical_or(np.logical_or(x_filt_far, y_filt_far), z_filt_far)
    not_cropped = np.logical_not(np.logical_or(close, far))
    
    return scan[not_cropped, :], not_cropped

def get_ground_color(scan, size=10):
    """Get the color of the ground in the point cloud by taking the median color of the points in the neighborhood 
    of the origin of the a scan"""
    # logging.info('np.abs(scan[:, 0]).min()=\n%s',np.abs(scan[:, 0]).min())
    # logging.info('np.abs(scan[:, 0]).max()=\n%s',np.abs(scan[:, 0]).max())
    # logging.info('np.abs(scan[:, 1]).min()=\n%s',np.abs(scan[:, 1]).min())
    # logging.info('np.abs(scan[:, 1]).max()=\n%s',np.abs(scan[:, 1]).max())
    x_filt_close = np.abs(scan[:, 0]) < 1
    y_filt_close = np.abs(scan[:, 1]) < 10
    z_filt_close = np.logical_and(scan[:, 2] > -2.2, scan[:, 2] < -1.5)
    neighborhood = np.logical_and(np.logical_and(x_filt_close, y_filt_close), z_filt_close)
    # scan_neigborhood = scan[neighborhood, :]
    # plot scan for debug
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(scan_neigborhood[:, :3])
    # # pcd.colors = o3d.utility.Vector3dVector((scan_neigborhood[:, 3]/255.0).reshape(1,-1))
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, coordinate_frame])

    # logging.info('neighborhood.shape=\n%s',neighborhood.shape)
    # logging.info('neighborhood.sum()=\n%s',neighborhood.sum())
    return np.median(scan[neighborhood, 3], axis=0) 


# def save_las_file(registered_pointcloud_path, points, colors, timestamps, intensities, point_format=1):
#     out_file = laspy.create(file_version="1.2", point_format=point_format)
#     out_file.X = (points[:, 0] * 1e4).astype(np.int32)
#     out_file.Y = (points[:, 1] * 1e4).astype(np.int32)
#     out_file.Z = (points[:, 2] * 1e4).astype(np.int32)
#     if colors is not None:
#         out_file.red = colors[:, 0] * 65535
#         out_file.green = colors[:, 1] * 65535
#         out_file.blue = colors[:, 2] * 65535
#     out_file.gps_time = timestamps
#     out_file.intensity = intensities
#     # Save the LAS file
#     out_file.write(registered_pointcloud_path)

def open_label(filename, points_shape):
    """ Open raw scan and fill in attributes
    From semantickitti
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in "label"):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == points_shape[0]:
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16    # instance id in upper half
    else:
        raise ValueError("Scan and Label don't contain same number of points")

    # Verify that the combined semantic and instance labels match the original label data
    assert((sem_label + (inst_label << 16) == label).all())

    # if project:
    #     do_label_projection()
    
    return sem_label #, inst_label

def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def parse_poses(filename, calibration): 
    """ read poses file with per-scan poses from seg_3D_by_2D.given filename 
    
        Returns 
        ------- 
        list 
            list of poses as 4x4 numpy arrays. 
    """ 
    file = open(filename) 
    
    poses = [] 
    
    Tr = calibration["Tr"] 
    Tr_inv = np.linalg.inv(Tr) 
    
    for line in file: 
        values = [float(v) for v in line.strip().split()] 
    
        pose = np.zeros((4, 4)) 
        pose[0, 0:4] = values[0:4] 
        pose[1, 0:4] = values[4:8] 
        pose[2, 0:4] = values[8:12] 
        pose[3, 3] = 1.0 
    
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr))) 
    
    return poses

def get_camera_coefficients_from_calibration(calibration, image_size):
    """Get camera coefficients from KITTI calibration for rendering"""
    # Use P2 (left color camera) intrinsics
    P2 = calibration["P2"]  # 3x4 projection matrix
    
    # Extract intrinsic matrix from projection matrix
    intrinsic_matrix = P2[:3, :3]
    
    # Scale intrinsics to match the desired image size
    # KITTI images are typically 1241x376
    original_width = 1241
    original_height = 376
    
    # Scale factors
    scale_x = image_size[0] / original_width
    scale_y = image_size[1] / original_height
    
    # Scale the intrinsic matrix
    fx = intrinsic_matrix[0, 0] * scale_x
    fy = intrinsic_matrix[1, 1] * scale_y
    cx = intrinsic_matrix[0, 2] * scale_x
    cy = intrinsic_matrix[1, 2] * scale_y
    
    return fx, fy, cx, cy

def generate_depth_maps(points,    
                        colors,    
                        camera_poses,  
                        output_folder,   
                        calibration,
                        image_size=(1241, 376),
                        point_size=3,
                        depth_encoding_max_depth=50,
                        depth_encoding_num_bins = 5000, # depth values are binned to make the images lighter after compression (max 65535 bins)
                        depth_encoding_max_bits = 2**16,
                        ):
    """Generate depth maps from pointcloud at given camera poses using pyrender"""
    logging.info("Generating depth maps for %d camera poses", len(camera_poses))
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    import png
    depth_writer = png.Writer(width=image_size[0], height=image_size[1], bitdepth=16, greyscale=True)
    # Setup renderer
    r = pyrender.OffscreenRenderer(image_size[0], image_size[1], point_size=point_size)
    
    # Setup scene
    scene = pyrender.Scene(bg_color=[0, 0, 0])
    
    # Add pointcloud to scene
    mesh = pyrender.Mesh.from_points(points=points, colors=colors)
    scene.add(mesh)
    
    # Setup camera with proper intrinsics
    fx, fy, cx, cy = get_camera_coefficients_from_calibration(calibration, image_size)

    
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    nc = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(nc)
    
    # Setup rendering flags
    flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.FLAT
    
    # Rotation to align with camera coordinate system
    rot_x = R.from_euler('x', 180, degrees=True).as_matrix()
    
    depth_maps = []
    
    for i in tqdm(range(len(camera_poses)), desc="Generating depth maps"):
        depth_filename = f"depth_{i:06d}.png"
        if os.path.exists(os.path.join(output_folder, depth_filename)):
            continue
        camera_pose = camera_poses[i].copy()
        
        # Apply rotation to align with camera coordinate system
        camera_pose[:3, :3] = np.dot(camera_pose[:3, :3], rot_x)
        scene.set_pose(nc, pose=camera_pose)
        
        # Render
        _, depth_image = r.render(scene, flags=flags)
        
        # Save depth map
        depth_path = os.path.join(output_folder, depth_filename)
        depth_image[depth_image<=0.0] = depth_encoding_max_depth
        depth_image[depth_image>depth_encoding_max_depth] = depth_encoding_max_depth
        depth_image = (depth_image / depth_encoding_max_depth * depth_encoding_num_bins).astype(np.uint16)
        depth_image *= (depth_encoding_max_bits // depth_encoding_num_bins)

        with open(depth_path, 'wb') as f:            
            depth_writer.write(f, depth_image.tolist())
        
        depth_maps.append(depth_image)
    
    # Cleanup
    r.delete()
    scene.clear()
    
    logging.info("Generated %d depth maps in %s", len(depth_maps), output_folder)
    return depth_maps

def get_scans_labels_poses_indices(
        scans_filenames,
        scans_folder,
        labels_folder,
        lidar_poses,
        sequence,
        start_frame,
        end_frame,
        frame_inds=None,
    ):
    scans_labels_poses_indices = []
 
    if frame_inds is not None:
        # logging.info('frame_inds=%s',frame_inds)
        scans_filenames = [scans_filenames[i] for i in frame_inds]
        lidar_poses = [lidar_poses[i] for i in frame_inds]
        assert len(scans_filenames) == len(lidar_poses), "Number of scans and poses do not match : %s vs %s" % (len(scans_filenames), len(lidar_poses))
    else:
        scans_filenames = scans_filenames[start_frame:end_frame+1]
        lidar_poses = lidar_poses[start_frame:end_frame+1]
    assert len(scans_filenames) == len(lidar_poses), "Number of scans and poses do not match : %s vs %s" % (len(scans_filenames), len(lidar_poses))
    

    for i in range(len(scans_filenames)):
        scan_filename = scans_filenames[i]
        scan_path = os.path.join(scans_folder, scan_filename)
        label_path = os.path.join(labels_folder, scan_filename.replace(".bin", ".label"))
        assert os.path.exists(label_path), "Label file not found: %s" % label_path
        
        # Make unique file_index for each file in each sequence
        file_index = get_unique_file_index_sk(sequence, scan_filename)
        scans_labels_poses_indices.append((scan_path, label_path, lidar_poses[i], file_index))

    # Ensure file indices are unique, which may be use to go from registered data to original scans
    num_unique_file_indices = len(np.unique([file_index for _, _, _, file_index in scans_labels_poses_indices]))
    assert num_unique_file_indices == len(scans_labels_poses_indices), f"File indices are not unique : {num_unique_file_indices} vs {len(scans_labels_poses_indices)}"        
    return scans_labels_poses_indices


def prepare_semantickitti(roots_to_dataset, 
                    exp_name, 
                    output_folder,
                    # scene_length=500,
                    save_sensor_positions=True,
                    plot=False, 
                    radius_close=0, 
                    box_far=[[-1000, 1000], [-1000, 1000], [-2.2, 1000]], 
                    log_path=None, 
                    save_gt=True, 
                    shared=False,
                    sensor_positions_filename="sensor_positions.npz",
                    re_prepare=False,
                    recompute_gt_only=False,
                    min_intensity=0,
                    max_intensity=245,
                    min_clip=0,
                    max_clip=255,
                    blur_intensity_k=1, # 1 means no blurring
                    blur_with_random_intensity=False,
                    blur_random_values=[1, 1, 3, 5, 8],
                    gt_from_pred=False, # Use to prepare gt from predictions (should be stored in semantickitti path as /predictions folder instead of /labels) for debug
                    extend_scenes_with_scans = 0, # Number of scan to add before and after the start/end frames to make the scene denser at the edge.
                    freq_extended = 2, # Number of scan to add before and after the start/end frames to make the scene denser at the edge.
                    colorize_with_cameras=False,
                    colorize_depth_margin=0.5,
                    colorize_max_depth=50,
                    colorize_min_depth=0,
                    colorize_frame_margin=80, # numberir of frames to colorize before and after the current start/end frames to colorize edge points
                    colorize_depth_map_point_size=3,
                    device_project="cuda",
                    subproc=True,
                    ):
    """Prepare KITTI dataset for processing by seg_3D_by_2D.
    For each scene of KITTI, create a folder in output_path then save 3 files to be processed : 
    - registered_pointcloud.las
    - timestamps_infos.npz
    - lidar_poses.npy
    timestamps_infos is a Nx3 array with timestamps(int), is_in_registered_pointcloud(bool), file_index(int).
    Note : the timestamps are NOT actual timestamps but only indices to identify each point.
    Note : the file indices are the last integer in each filename in KITTI
    file naming convention, for memory efficiency.
    
    If colorize_with_cameras is True, the colorization is done after registration by projecting
    the registered pointcloud to each camera frame and averaging colors from all frames.
    
    Extend with scans explained:
    - Add scans at the start and end samples by freq_extended
    - Remedies sparse points at the edge of the scene. Recommended to use
    e.g. a 60 scan scene with 20+20 scans for extension instead of full 100 scan scene.
    - These can be used for PC2D datasets as well, should not be a problem,
    could add some relevant data in transitions
    - Will be taken into account in compare_annos (so the eval is biased)
    - Will be handled in point_segmasks_to_scan, meaning extension won't 
    be part of the predictions and the eval_by_scan will not be biased

    """

    
    # radius_close=None
    # box_far=None
    logging.info("Cropping pointcloud with \n  - radius_close=%s \n  - box_far=%s", radius_close, box_far)


    required_files = {"registered_pointcloud.las", "timestamps_infos.npz", "lidar_poses.npy"}
    if save_gt:
        required_files.add("point_segmasks_gt.npy")
    if save_sensor_positions:
        required_files.add(sensor_positions_filename)

    for root, dataset_path in roots_to_dataset.items():
        scene_name = os.path.basename(root) if not root.endswith("/") else os.path.basename(root[:-1])
        sequence, start_frame, end_frame = scene_name.split('_')[0] , int(scene_name.split('_')[1]), int(scene_name.split('_')[2])
        sequence_path = dataset_path
        # destination_root_name = scene_name + '_' + str(start_frame) + '_' + str(end_frame)
        if isinstance(shared, str):
            save_folder = os.path.join(output_folder, scene_name, shared)
        elif shared:
            save_folder = os.path.join(output_folder, scene_name)
        else :
            save_folder = os.path.join(output_folder, scene_name, exp_name)
        logging.info("Preparing root %s in %s", root, save_folder)
        logging.info("Required files : %s", ", ".join(required_files))
        logging.info('os.path.join(save_folder, required_file) for required_file in required_files]=%s',[os.path.join(save_folder, required_file) for required_file in required_files])
        logging.info('[os.path.exists(os.path.join(save_folder, required_file)) for required_file in required_files]=%s',[os.path.exists(os.path.join(save_folder, required_file)) for required_file in required_files])
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        elif np.all([os.path.exists(os.path.join(save_folder, required_file)) for required_file in required_files]) \
            and not re_prepare and not recompute_gt_only:
            if log_path is not None:
                with open(log_path, "a") as f:
                    f.write("{}\n".format(root))
            continue
        elif recompute_gt_only:
            logging.info("Recomputing gt only")
            if os.path.exists(os.path.join(save_folder, "point_segmasks_gt.npy")):
                logging.info("Removing existing point_segmasks_gt.npy to recompute it")
                os.remove(os.path.join(save_folder, "point_segmasks_gt.npy"))
        # logging.info("Required files : %s", ", ".join(required_files))
        
        
        registered_pointcloud_path = os.path.join(save_folder, "registered_pointcloud.las")
        timestamps_infos_path = os.path.join(save_folder, "timestamps_infos.npz")
        lidar_poses_path = os.path.join(save_folder, "lidar_poses.npy")
        if save_gt:
            point_segmasks_gt_path = os.path.join(save_folder, "point_segmasks_gt.npy")
        if save_sensor_positions:
            sensor_positions_path = os.path.join(save_folder, sensor_positions_filename)

        # initialize ouptput
        timestamps_infos = np.empty((0, 3), dtype=np.int32)
        timestamps_registered_pointcloud = np.array([], dtype=np.int32)
        max_timestamp = 0
        # if save_gt: 
        point_segmasks_gt = np.empty((0, 3), dtype=np.int32)
        if save_sensor_positions:
            sensor_positions = np.empty((0, 3), dtype=np.float32)
        
        # Initialize camera colors array for colorization
        # if colorize_with_cameras:
            # logging.info("Starting camera colorization for KITTI dataset - will be done after registration")


        # # get lidar poses
        # poses = np.loadtxt(os.path.join(root, "poses.txt"))[:length]
        calibration_file = os.path.join(sequence_path, "calib.txt")
        calibration = parse_calibration(calibration_file)
        lidar_poses = parse_poses(os.path.join(sequence_path, "poses.txt"), calibration)
        num_lidar_poses = len(lidar_poses)
        
        if extend_scenes_with_scans:
            start_frame_extended = max(start_frame - extend_scenes_with_scans, 0)
            end_frame_extended = min(end_frame + extend_scenes_with_scans, num_lidar_poses-1)
            frame_inds = np.arange(start_frame_extended, start_frame, freq_extended)
            frame_inds = np.concatenate([frame_inds, np.arange(start_frame, end_frame+1)])
            frame_inds = np.concatenate([frame_inds, np.arange(end_frame+freq_extended, end_frame_extended+1, freq_extended)])
            logging.info("Scans will be concatened from frame %s to %s, including an extension of %s on each side with frequency %s", start_frame_extended, end_frame_extended, extend_scenes_with_scans, freq_extended)
            # logging.info('frame_inds=%s',frame_inds)
            # logging.info('frame_inds.dtype=%s',frame_inds.dtype)
        else:
            start_frame_extended = start_frame
            end_frame_extended = end_frame
            frame_inds = np.arange(start_frame, end_frame+1)

        if colorize_with_cameras:
            start_frame_cameras = max(start_frame - colorize_frame_margin, 0)
            end_frame_cameras = min(end_frame + colorize_frame_margin, num_lidar_poses-1)
            logging.info("Colorizing will use cameras from frame %s to %s", start_frame_cameras, end_frame_cameras)

            frame_inds_colorize = np.arange(start_frame_cameras, end_frame_cameras+1)



        # get all scans and labels
        scans_folder = os.path.join(sequence_path, "velodyne")
        labels_folder = os.path.join(sequence_path, "labels")
        if gt_from_pred:
            labels_folder = os.path.join(sequence_path, "predictions")

        scans_filenames = [f for f in os.listdir(scans_folder) if f.endswith(".bin")]
        scans_filenames = sorted(scans_filenames, key=lambda x: int(x.split(".")[0]))

        

        scans_labels_poses_indices = get_scans_labels_poses_indices(
            scans_filenames=scans_filenames,
            scans_folder=scans_folder,
            labels_folder=labels_folder,
            lidar_poses=lidar_poses,
            sequence=sequence,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_inds=frame_inds,
        )

        if colorize_with_cameras:
            scans_labels_poses_indices_colorize = get_scans_labels_poses_indices(
                scans_filenames=scans_filenames,
                scans_folder=scans_folder,
                labels_folder=labels_folder,
                lidar_poses=lidar_poses,
                sequence=sequence,
                start_frame=start_frame_cameras,
                end_frame=end_frame_cameras,
                frame_inds=frame_inds_colorize,
            )

        lidar_poses = [lidar_poses[i] for i in frame_inds]
        
        scan_path, label_path, lidar_pose, file_index = scans_labels_poses_indices[0]
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        len_scan = scan.shape[0]

        label = open_label(label_path, scan.shape)
        # if classes_converter is not None:
        #     label = classes_converter.cs1_to_cs2(label)

        ref_lidar_pose = lidar_pose.copy()

        # crop
        if radius_close is not None or box_far is not None:
            scan, not_cropped = crop_pointcloud(scan, radius_close=radius_close, box_far=box_far)
        else :
            not_cropped = np.ones(scan.shape[0], dtype=bool)
        num_points_not_cropped = np.sum(not_cropped)
        # label = label[not_cropped]
        
        if not recompute_gt_only:
            ground_color = get_ground_color(scan)
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
            add_intensity_to_pointcloud(pcd, scan[:, 3], 
                                        min_intensity=min_intensity, 
                                        max_intensity=max_intensity, 
                                        min_clip=min_clip, 
                                        max_clip=max_clip,
                                        ground_color=ground_color)
        else:
            # Initialize empty pcd for GT-only mode to avoid undefined variable errors
            pcd = None

        
        # update timestamps_infos
        max_timestamp = len_scan
        timestamps_all = np.arange(max_timestamp)
        timestamps_not_cropped = timestamps_all[not_cropped]
        timestamps_registered_pointcloud = np.concatenate([timestamps_registered_pointcloud, timestamps_not_cropped])
        timestamps_infos = np.concatenate([timestamps_infos, 
                                           np.hstack((timestamps_all.reshape(-1, 1),
                                                    not_cropped.reshape(-1, 1),
                                                    np.ones((len(timestamps_all), 1), dtype=int) * file_index))
                                        ], axis=0)
        point_segmasks_gt = np.concatenate([point_segmasks_gt, 
                                                np.hstack((timestamps_all.reshape(-1, 1),
                                                            label.reshape(-1, 1),
                                                            label.reshape(-1, 1)))], axis=0) # 3rd column is for instances (not used)
        if save_sensor_positions:
            sensor_positions = np.concatenate([sensor_positions, np.zeros((num_points_not_cropped, 3))], axis=0)
         

        for i, (scan_path, label_path, lidar_pose, file_index) in tqdm(enumerate(scans_labels_poses_indices[1:]), 
                                                                       total=len(scans_labels_poses_indices[1:])):     

            scan = np.fromfile(scan_path, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            len_scan = scan.shape[0]


            label = open_label(label_path, scan.shape)
            # if classes_converter is not None:
            #     label = classes_converter.cs1_to_cs2(label)
            # crop
            if radius_close is not None or box_far is not None:
                scan, not_cropped = crop_pointcloud(scan, radius_close=radius_close, box_far=box_far)
            else :
                not_cropped = np.ones(scan.shape[0], dtype=bool)
            
            if not recompute_gt_only:
                ground_color = get_ground_color(scan)
            num_points_not_cropped = np.sum(not_cropped)

            # label = label[not_cropped]

            # update timestamps_infos
            new_max_timestamp = max_timestamp + len_scan
            timestamps_all = np.arange(max_timestamp, new_max_timestamp)
            max_timestamp = new_max_timestamp

            timestamps_not_cropped = timestamps_all[not_cropped]
            timestamps_registered_pointcloud = np.concatenate([timestamps_registered_pointcloud, timestamps_not_cropped])
            timestamps_infos = np.concatenate([timestamps_infos, 
                                            np.hstack((timestamps_all.reshape(-1, 1),
                                                        not_cropped.reshape(-1, 1),
                                                        np.ones((len(timestamps_all), 1), dtype=int) * file_index))
                                        ], axis=0)
            
            point_segmasks_gt = np.concatenate([point_segmasks_gt, 
                                                np.hstack((timestamps_all.reshape(-1, 1),
                                                            label.reshape(-1, 1),
                                                            label.reshape(-1, 1)))], axis=0) # 3rd column is for instances (not used)
            if save_sensor_positions:
                sensor_positions = np.concatenate([sensor_positions, np.tile(lidar_pose[:3, 3], (num_points_not_cropped, 1))], axis=0)

            if not recompute_gt_only:
                # transform pointcloud
                scan[:,:3] = np.dot(lidar_pose, 
                                    np.hstack((scan[:,:3], np.ones((len(scan), 1)))).T).T[:,:3]
                scan[:,:3] = np.dot(np.linalg.inv(ref_lidar_pose), 
                                    np.hstack((scan[:,:3], np.ones((len(scan), 1)))).T).T[:,:3]

                # Convert the transformed point cloud to open3d format and concatenate
                pcd_current = o3d.geometry.PointCloud()
                pcd_current.points = o3d.utility.Vector3dVector(scan[:, :3])
                add_intensity_to_pointcloud(pcd_current, scan[:, 3], 
                                        min_intensity=min_intensity, 
                                        max_intensity=max_intensity, 
                                        min_clip=min_clip, 
                                        max_clip=max_clip,
                                        ground_color=ground_color)
                pcd += pcd_current

        
        if not recompute_gt_only:
            if plot:
                try:
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
                    o3d.visualization.draw_geometries([coordinate_frame, pcd])
                except Exception as e:
                    logging.warning(f"Visualization failed: {e}")
                    logging.info("Skipping visualization") 

            # save las file
            points = np.asarray(pcd.points)
            # logging.info('len(points)=\n%s',len(points))
            # logging.info('timestamps_infos[:,1].sum()=%s / %s total',timestamps_infos[:,1].sum(), len(timestamps_infos))
            intensities = np.asarray(pcd.colors)[:,0]
        else:
            # Initialize dummy variables for GT-only mode to avoid undefined variable errors
            points = np.array([])
            intensities = np.array([])
        
        # Post-registration colorization
        colors = None
        if not recompute_gt_only and colorize_with_cameras:
            # logging.info("Starting post-registration colorization...")
            
            # Get the final registered points
            points_colorize = np.asarray(pcd.points)
            num_points = len(points_colorize)
            
            # Generate depth maps for occlusion testing
            # logging.info("Generating depth maps for occlusion testing...")
            depth_maps_folder = os.path.join(save_folder, "depth_maps")
            
            # Create camera poses from lidar poses using KITTI calibration
            camera_poses = []
            Tr_velo_to_cam = calibration["Tr"]  # Transformation from velodyne to camera (3x4)
            
            # Convert Tr to 4x4 matrix
            # Tr_velo_to_cam_4x4[:4, :4] = Tr_velo_to_cam
            
            for _, _, lidar_pose, _ in scans_labels_poses_indices_colorize:
                # Transform from lidar to camera coordinate system
                camera_pose = np.linalg.inv(ref_lidar_pose) @ lidar_pose @ np.linalg.inv(Tr_velo_to_cam)
                camera_poses.append(camera_pose)

            args_render = (
                points_colorize,
                np.asarray(pcd.colors),
                camera_poses,
                depth_maps_folder,
                calibration,
                (1241, 376),
                colorize_depth_map_point_size,
            )

            if subproc:
                render_process = multiprocessing.Process(target=generate_depth_maps,
                                                        args=args_render)
                render_process.start()
                render_process.join()
            else:
                generate_depth_maps(*args_render)
            # generate_depth_maps(
            #     points=points_colorize,
            #     colors=np.asarray(pcd.colors),
            #     camera_poses=camera_poses,
            #     output_folder=depth_maps_folder,
            #     image_size=(1241, 376),
            #     calibration=calibration,
            #     point_size=colorize_depth_map_point_size,
            # )
            # np.save(os.path.join(depth_maps_folder, "camera_poses_colorize.npy"), camera_poses)
            
            points_colorize_tensor = torch.tensor(points, dtype=torch.float32, device=device_project, requires_grad=False)
            ref_lidar_pose_tensor = torch.tensor(ref_lidar_pose, dtype=torch.float32, device=device_project)
            # Initialize colors array
            colors = torch.zeros(size=(num_points, 3),
                                            dtype=torch.float32, 
                                            device=device_project, 
                                            requires_grad=False)
            
            
            # Loop through all frames to colorize the registered pointcloud
            for i, (scan_path, label_path, lidar_pose, file_index) in tqdm(enumerate(scans_labels_poses_indices_colorize), total=len(scans_labels_poses_indices_colorize)):
                scan_filename = os.path.basename(scan_path)
                scan_index = int(scan_filename.split('.')[0])
                
                # Transform registered points to current frame's coordinate system
                # First transform from registered frame to world frame, then to current frame
                points_world_tensor = torch.mm(ref_lidar_pose_tensor, torch.hstack((points_colorize_tensor, torch.ones((num_points, 1), dtype=torch.float32, device=device_project))).T).T[:, :3]
                lidar_pose_tensor = torch.tensor(lidar_pose, dtype=torch.float32, device=device_project)
                points_current_frame_tensor = torch.mm(torch.linalg.inv(lidar_pose_tensor), torch.hstack((points_world_tensor, torch.ones((num_points, 1), dtype=torch.float32, device=device_project))).T).T[:, :3]
                # logging.info('points_current_frame.shape=%s',points_current_frame.shape)
                # Colorize points in current frame
                depth_filename = f"depth_{i:06d}.png"
                # frame_colors = colorize_pointcloud(
                frame_colors = colorize_pointcloud_gpu(
                    points=points_current_frame_tensor,
                    calibration=calibration,
                    scan_filename=scan_filename,
                    kitti_path=sequence_path,
                    depth_map_path=os.path.join(depth_maps_folder, depth_filename) if depth_maps_folder is not None else None,
                    depth_margin=colorize_depth_margin,
                    max_depth=colorize_max_depth,
                    min_depth=colorize_min_depth,
                )
                
                valid_colors = torch.any(frame_colors > 0, dim=1)
                colors[valid_colors] = frame_colors[valid_colors]
                
                # total_colored_points = torch.sum(torch.any(colors > 0, dim=1))
                # logging.info(f"Frame {scan_index}: total colored so far: {total_colored_points}/{num_points} ({total_colored_points/num_points*100:.1f}%) points")
            
            # logging.info(f"Post-registration colorization completed. Colors shape: {colors.shape}")
            colored_points = torch.sum(torch.any(colors > 0, dim=1))
            logging.info(f"Colored points: {colored_points}/{num_points} ({colored_points/num_points*100:.1f}%)")
            
        colors = colors.cpu().numpy() if colors is not None else None
        if recompute_gt_only:
            colors = None
        
        if not recompute_gt_only:
            # normalize over registered points
            # intensities = np.clip(intensities, 0.05, 0.95)
            # intensities = intensities - intensities.min()
            # intensities = intensities / intensities.max()

            # ##### # Compute the histogram
            logging.info("Normalizing intensities with min_intensity=%s, max_intensity=%s", min_intensity, max_intensity)
            hist, bins = np.histogram(intensities, bins=256, range=(intensities.min(), intensities.max()), density=False)
            # Compute the cumulative distribution function (CDF)
            cdf = hist.cumsum()
            # Normalize the CDF
            # cdf_normalized = (cdf - cdf.min()) * max_intensity / (cdf.max() - cdf.min())
            cdf_normalized = (cdf - cdf.min()) * (max_intensity - min_intensity) / (cdf.max() - cdf.min()) + min_intensity
            cdf_normalized = cdf_normalized.astype('uint8')    
            # Use linear interpolation of the CDF to find new pixel values
            intensities = np.interp(intensities.flatten(), bins[:-1], cdf_normalized) / 255.0

            if blur_with_random_intensity:
                k = np.random.choice(blur_random_values)
            else:
                k = blur_intensity_k
            if k > 1:
                # points, intensities = filter_intensity_knn_batch(points, intensities, k=blur_intensity_k)
                filter_intensity_knn_batch(points, intensities, k=k)

                logging.info("Re-Normalizing after blur with intensities with min_intensity=%s, max_intensity=%s", min_intensity, max_intensity)
                ##### # Compute the histogram
                hist, bins = np.histogram(intensities, bins=256, range=(intensities.min(), intensities.max()), density=False)
                # Compute the cumulative distribution function (CDF)
                cdf = hist.cumsum()
                # Normalize the CDF
                # cdf_normalized = (cdf - cdf.min()) * max_intensity / (cdf.max() - cdf.min())
                cdf_normalized = (cdf - cdf.min()) * (max_intensity - min_intensity) / (cdf.max() - cdf.min()) + min_intensity
                cdf_normalized = cdf_normalized.astype('uint8')    
                # Use linear interpolation of the CDF to find new pixel values
                intensities = np.interp(intensities.flatten(), bins[:-1], cdf_normalized) / 255.0
            intensities = intensities * 65535
            intensities = intensities.astype(int) # store the colors as intensities for memory

            if colorize_with_cameras and colors is not None:
                # fill uncolored points with intensity
                # Find points that have no color (all channels are 0)
                uncolored_points = np.all(colors == 0, axis=1)
                # Fill those points with intensity values for all channels
                colors[uncolored_points, :] = intensities[uncolored_points].reshape(-1, 1).astype(float) / 65535.0

            # logging.info('points.shape=\n%s',points.shape)
            # logging.info('timestamps.shape=\n%s',timestamps_registered_pointcloud.shape)
            # logging.info('intensities.shape=\n%s',intensities.shape)
            logging.info("Saving registered pointcloud to %s", "/".join(registered_pointcloud_path.split("/")[-2:]))
            save_las_file(registered_pointcloud_path=registered_pointcloud_path,
                          points=points,
                          colors=colors,
                          timestamps=timestamps_registered_pointcloud,
                          intensities=intensities)

            # save lidar poses
            # Transform poses to be relative to reference pose
            ref_lidar_pose_inv = np.linalg.inv(ref_lidar_pose)
            transformed_lidar_poses = np.array([np.dot(ref_lidar_pose_inv, pose) for pose in lidar_poses])
            np.save(lidar_poses_path, transformed_lidar_poses)

            # save timestamps_infos
            np.savez(timestamps_infos_path, timestamps_infos=timestamps_infos)

        if save_gt:
            # logging.info('point_segmasks_gt.shape=\n%s',point_segmasks_gt.shape)
            logging.info("Saving point segmasks gt to %s", "/".join(point_segmasks_gt_path.split("/")[-2:]))
            np.save(point_segmasks_gt_path, point_segmasks_gt)
        if not recompute_gt_only and save_sensor_positions:
            logging.info("Saving sensor positions to %s", "/".join(sensor_positions_path.split("/")[-2:]))
            np.savez_compressed(sensor_positions_path, sensor_positions)
            

        if log_path is not None:
            with open(log_path, "a") as f:
                f.write("{}\n".format(save_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--output_folder', help='', required=True)
    parser.add_argument('--exp_name', help='', default=None)
    parser.add_argument('--semantickitti_path', help='', default="~/datasets/semantickitti")
    parser.add_argument('--num_scenes', help='', default=1, type=int)
    parser.add_argument('--start_frame', help='', default=0, type=int)
    parser.add_argument('--end_frame', help='', default=100, type=int)
    parser.add_argument('--plot', help='', action='store_true')
    parser.add_argument('--generate_all', help='', action='store_true')
    parser.add_argument('--colorize_with_cameras', help='', action='store_true')
    parser.add_argument('--camera_channels', nargs='+', type=int, default=[2, 3], help='Camera IDs to use')
    args = parser.parse_args()

    # Example usage - you would need to implement the main logic here
    logging.info("This script is meant to be imported and used as a module")
    logging.info("Use prepare_semantickitti() function directly")