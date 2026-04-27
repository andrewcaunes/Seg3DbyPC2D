"""
Script made by Andrew Caunes.
Prepare datasets such as NuScenes for processing by seg_3D_by_2D.py.
Example use:
python script.py ...
"""
import os
import shutil
import argparse
import logging

from tqdm import tqdm
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
import laspy
from seg_3D_by_2D.utils.utils import save_las_file, filter_intensity_knn_batch, get_unique_file_index_sk, get_scene_name_from_output, write_log


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


def crop_pointcloud(scan, radius_close=None, box_far=None) -> np.ndarray:
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
    From synlidar
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


def parse_poses(filename): 
    """ read poses file with per-scan poses from seg_3D_by_2D.given filename 
    
        Returns 
        ------- 
        list 
            list of poses as 4x4 numpy arrays. 
    """ 
    file = open(filename) 
    
    poses = [] 
    
    # Tr = calibration["Tr"] 
    # Tr_inv = np.linalg.inv(Tr) 
    
    for line in file: 
        values = [float(v) for v in line.strip().split()] 
    
        pose = np.zeros((4, 4)) 
        pose[0, 0:4] = values[0:4] 
        pose[1, 0:4] = values[4:8] 
        pose[2, 0:4] = values[8:12] 
        pose[3, 3] = 1.0 
        poses.append(pose)
        # poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr))) # no need for calib with local generated poses for synlidar
    
    return poses

def prepare_synlidar(roots_to_dataset, 
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
                    min_intensity=0,
                    max_intensity=245,
                    min_clip=0,
                    max_clip=255,
                    blur_intensity_k=1, # 1 means no blurring
                    blur_with_random_intensity=False,
                    blur_random_values=[1, 1, 3, 5, 8],
                    gt_from_pred=False, # Use to prepare gt from predictions (should be stored in synlidar path as /predictions folder instead of /labels) for debug
                    extend_scenes_with_scans=0, # Number of scans to add before and after start/end to densify edges
                    freq_extended=2, # Frequency for added scans in the extension window
                    recompute_gt_only=False,
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

    
    # radius_close=None
    # box_far=None
    logging.info("Cropping pointcloud with \n  - radius_close=%s \n  - box_far=%s", radius_close, box_far)


    required_files = {"registered_pointcloud.las", "timestamps_infos.npz", "lidar_poses.npy"}
    if save_gt:
        required_files.add("point_segmasks_gt.npy")
    if save_sensor_positions:
        required_files.add(sensor_positions_filename)

    skipping = []
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
        # logging.info('save_folder=\n%s',save_folder)
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
        # logging.info("Required files : %s", ", ".join(required_files))
        
        if len(skipping) > 0:
            logging.info("Skipped from %s to %s", get_scene_name_from_output(skipping[0]), get_scene_name_from_output(skipping[-1]))
            logging.info("Preparing %s", get_scene_name_from_output(root))
        if recompute_gt_only:
            logging.info("Recomputing gt only for %s", get_scene_name_from_output(root))
        logging.info("Required files : %s", ", ".join(required_files))
        
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


        # # get lidar poses
        # poses = np.loadtxt(os.path.join(root, "poses.txt"))[:length]
        # calibration_file = os.path.join(sequence_path, "calib.txt")
        # calibration = parse_calibration(calibration_file)
        logging.info('sequence_path=\n%s',sequence_path)
        logging.info('start_frame=\n%s',start_frame)
        logging.info('end_frame=\n%s',end_frame)
        lidar_poses = parse_poses(os.path.join(sequence_path, "poses.txt"))
        # Determine extended frame indices similar to SemanticKITTI
        num_lidar_poses = len(lidar_poses)
        if extend_scenes_with_scans:
            start_frame_extended = max(start_frame - extend_scenes_with_scans, 0)
            end_frame_extended = min(end_frame + extend_scenes_with_scans, num_lidar_poses - 1)
            frame_inds = np.arange(start_frame_extended, start_frame, freq_extended)
            frame_inds = np.concatenate([frame_inds, np.arange(start_frame, end_frame + 1)])
            frame_inds = np.concatenate([frame_inds, np.arange(end_frame + freq_extended, end_frame_extended + 1, freq_extended)])
            logging.info("Scans will be concatenated from frame %s to %s, including an extension of %s on each side with frequency %s", start_frame_extended, end_frame_extended, extend_scenes_with_scans, freq_extended)
        else:
            frame_inds = np.arange(start_frame, end_frame + 1)
        # Keep poses only for selected frames
        lidar_poses = [lidar_poses[i] for i in frame_inds]
        # for pose in poses:
        #     lidar_poses.append(np.vstack((np.reshape(pose, (3, 4)), [0, 0, 0, 1])))
        # lidar_poses = np.array(lidar_poses)

        # get all scans and labels
        scans_folder = os.path.join(sequence_path, "velodyne")
        labels_folder = os.path.join(sequence_path, "labels")
        if gt_from_pred:
            labels_folder = os.path.join(sequence_path, "predictions") #TODO REMOVE
        scans_filenames = [f for f in os.listdir(scans_folder) if f.endswith(".bin")]
        scans_filenames = sorted(scans_filenames, key=lambda x: int(x.split(".")[0]))
        # Select filenames using the computed frame indices
        scans_filenames = [scans_filenames[i] for i in frame_inds]
        scans_labels_poses_indices = []
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
        
        scan_path, label_path, lidar_pose, file_index = scans_labels_poses_indices[0]
        ref_lidar_pose = lidar_pose.copy()
        ref_lidar_pose_inv = np.linalg.inv(ref_lidar_pose)
        transformed_lidar_poses = np.array([np.dot(ref_lidar_pose_inv, pose) for pose in lidar_poses])
        np.save(lidar_poses_path, transformed_lidar_poses)

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
            not_cropped = np.ones(scan.points.shape[1], dtype=bool)
        num_points_not_cropped = np.sum(not_cropped)
        # label = label[not_cropped]
        ground_color = get_ground_color(scan)
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
        add_intensity_to_pointcloud(pcd, scan[:, 3], 
                                    max_intensity=max_intensity,
                                    min_intensity=min_intensity, 
                                    min_clip=min_clip, 
                                    max_clip=max_clip,
                                    ground_color=ground_color)

        
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
                not_cropped = np.ones(scan.points.shape[1], dtype=bool)
            
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

        if plot :
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([coordinate_frame, pcd]) 

        # save las file
        points = np.asarray(pcd.points)
        # logging.info('len(points)=\n%s',len(points))
        # logging.info('timestamps_infos[:,1].sum()=%s / %s total',timestamps_infos[:,1].sum(), len(timestamps_infos))
        intensities = np.asarray(pcd.colors)[:,0]
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

        # logging.info('points.shape=\n%s',points.shape)
        # logging.info('timestamps.shape=\n%s',timestamps_registered_pointcloud.shape)
        # logging.info('intensities.shape=\n%s',intensities.shape)
        logging.info("Saving registered pointcloud to %s", "/".join(registered_pointcloud_path.split("/")[-2:]))
        save_las_file(registered_pointcloud_path=registered_pointcloud_path,
                      points=points,
                      colors=None,
                      timestamps=timestamps_registered_pointcloud,
                      intensities=intensities)


        # save timestamps_infos
        np.savez(timestamps_infos_path, timestamps_infos=timestamps_infos)

        if save_gt:
            # logging.info('point_segmasks_gt.shape=\n%s',point_segmasks_gt.shape)
            logging.info("Saving point segmasks gt to %s", "/".join(point_segmasks_gt_path.split("/")[-2:]))
            np.save(point_segmasks_gt_path, point_segmasks_gt)
        if save_sensor_positions:
            logging.info("Saving sensor positions to %s", "/".join(sensor_positions_path.split("/")[-2:]))
            np.savez_compressed(sensor_positions_path, sensor_positions)
            

        if log_path is not None:
            with open(log_path, "a") as f:
                f.write("{}\n".format(save_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--output_folder', help='', required=True)
    parser.add_argument('--exp_name', help='', default=None)
    parser.add_argument('--synlidar_path', help='', default="~/datasets/synlidar")
    parser.add_argument('--num_scenes', help='', default=1, type=int)
    parser.add_argument('--start_frame', help='', default=0, type=int)
    parser.add_argument('--end_frame', help='', default=100, type=int)
    parser.add_argument('--plot', help='', action='store_true')
    parser.add_argument('--generate_all', help='', action='store_true')
    args = parser.parse_args()

    main(args)
