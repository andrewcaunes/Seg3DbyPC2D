"""
Script made by Andrew Caunes.
Example use:
python script.py ...
"""
import os
import shutil
import argparse
import logging
import json
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import open3d as o3d
import nuscenes
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud
import laspy
# from seg_3D_by_2D.core.classes_dicts import lidarseg_to_global_dict, lidarseg_ext_to_global_dict, 
#     semantickitti_to_global_dict, color_palette
from seg_3D_by_2D.utils.utils import read_las_file
from seg_3D_by_2D.core.classes_dicts import (color_palette_float, classes_systems_converter, 
    nuscenes_uda_classes_system, uda_classes_system, 
    semantickitti_uda_classes_system, global_ext_classes_system,
    dglss_classes_system, nuscenes_global_classes_system,
    semantickitti_global_classes_system)

# classes_map_lidarseg={i:0 for i in range(32)}
# classes_map_lidarseg[24] = 1 # road
# classes_map_lidarseg[25] = 2 # sidewalk
# classes_map_lidarseg[26] = 5 # terrain
# classes_map_lidarseg[27] = 6 # other
# classes_map_lidarseg[28] = 3 # building
# classes_map_lidarseg[29] = 4 # vegetation
# classes_map_lidarseg[30] = 7 # static other

# classes_map_lidarseg={i:0 for i in range(32)}
# classes_map_lidarseg[24] = 1 # road
# classes_map_lidarseg[25] = 2 # sidewalk
# classes_map_lidarseg[26] = 5 # terrain
# classes_map_lidarseg[27] = 6 # other
# classes_map_lidarseg[28] = 3 # building
# classes_map_lidarseg[29] = 4 # vegetation
# classes_map_lidarseg[30] = 7 # static other


# color_map = color_palette.astype(np.float32) / 255.0
color_map = color_palette_float

def load_camera_parameters(json_path):
    """Load camera parameters from a JSON file."""
    logging.info("Loading camera parameters from %s", json_path)
    try:
        with open(json_path, 'r') as f:
            camera_params_dict = json.load(f)

        camera_params = o3d.camera.PinholeCameraParameters()

        extrinsic = np.array(camera_params_dict["extrinsic"]).reshape(4, 4).T
        camera_params.extrinsic = extrinsic

        intrinsic_dict = camera_params_dict["intrinsic"]
        width = intrinsic_dict["width"]
        height = intrinsic_dict["height"]

        intrinsic_matrix = np.array(intrinsic_dict["intrinsic_matrix"]).reshape(3, 3)
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[2, 0]
        cy = intrinsic_matrix[2, 1]

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        camera_params.intrinsic = intrinsic

        logging.info("Camera parameters loaded successfully")
        logging.info("Camera intrinsic matrix:\n%s", intrinsic_matrix)
        logging.info("Camera extrinsic matrix:\n%s", extrinsic)
        logging.info("Camera width: %s, height: %s", width, height)
        logging.info("Camera focal lengths: fx=%s, fy=%s", fx, fy)
        return camera_params
    except Exception as e:
        logging.error("Failed to load camera parameters: %s", str(e))
        return None

def plot_gt(args):
    logging.info("args = %s", args)
    # plot_gt(args.root=args.args.root,
    #             args.dataset_name=args.args.dataset_name, 
    #             # dataset_path=args.dataset_path, 
    #             # nuscenes_mode=args.nuscenes_mode, 
    #             args.pointcloud_filename=args.args.pointcloud_filename,
    #             args.gt_filename=args.args.gt_filename,
                
    #         )

# def (args.root,
#                 args.dataset_name,
#                 # dataset_path,
#                 # nuscenes_mode,
#                 classes_map=None,
#                 args.pointcloud_filename="registered_pointcloud.las",
#                 args.gt_filename="point_segmasks_gt.npy",
#             ):
    
    if args.display_cs == 'uda':
        display_classes_system = uda_classes_system
    elif args.display_cs == 'dglss':
        display_classes_system = dglss_classes_system
    elif args.display_cs == 'global_ext':
        display_classes_system = global_ext_classes_system
    else:
        display_classes_system = None
    

    if args.display_cs != "global_ext" and args.display_cs is not None:
        gt_classes_system = globals()[args.dataset_name+"_"+args.display_cs+"_classes_system"]
    elif args.display_cs == "global_ext":
        gt_classes_system = globals()[args.dataset_name+"_global_classes_system"]  
    else:
        gt_classes_system = None
                
    if display_classes_system is not None:
        cs_converter = classes_systems_converter(gt_classes_system, display_classes_system) 
    else:
        cs_converter = None


    # # # nusc = nuscenes.NuScenes(version='v1.0-'+nuscenes_mode, dataroot=dataset_path, verbose=True)
    registered_pointcloud_path = os.path.join(args.root, args.pointcloud_filename)
    # las = laspy.read(las_file)
    # points = np.vstack((las.x, las.y, las.z)).T.astype(float)
    # timestamps_pcd = np.array(las.gps_time).astype(int)
    # # intensity = las.intensity
    # colors = np.array(las.intensity).astype(float) / 65535.0
    # colors = np.vstack((colors, colors, colors)).T
    points, colors, _, timestamps_pcd = read_las_file(registered_pointcloud_path, use_rgb_colors=not args.use_intensity_colors) 


    gt_path = os.path.join(args.root, args.gt_filename)
    gt = np.load(gt_path)
    timestamps_gt = gt[:,0]

    # keep only points that are in both lidar and gt
    timestamps = np.intersect1d(timestamps_pcd, timestamps_gt)
    logging.info('len(timestamps_pcd)=\n%s',len(timestamps_pcd))
    logging.info('len(timestamps_gt)=\n%s',len(timestamps_gt))
    logging.info('len(timestamps)=\n%s',len(timestamps))
    mask_pcd = np.isin(timestamps_pcd, timestamps)
    mask_gt = np.isin(timestamps_gt, timestamps)
    points = points[mask_pcd]
    colors = colors[mask_pcd]
    logging.info('gt[:3]=%s',gt[:3])
    segmask = gt[mask_gt,1]
    logging.info('segmask.shape=%s',segmask.shape)
    # map classes to classes of interest
    # segmask = np.array([classes_map_gt[i] for i in segmask])
    
    unique = np.unique(segmask, return_counts=True)
    for i in range(len(unique[0])):
        logging.info('unique_class=%s, count=%s',unique[0][i],unique[1][i])
    unique = unique[0]
    dict_classes_to_inds = {sem_class: i for i, sem_class in enumerate(unique)}
    if cs_converter is not None:
        logging.info('cs_converter.cs1_to_cs2_dict=\n%s',cs_converter.cs1_to_cs2_dict)
        segmask = cs_converter.cs1_to_cs2(segmask)
    logging.info('np.unique(segmask, return_counts=True)=%s',np.unique(segmask, return_counts=True))
    if display_classes_system is not None:
        logging.info('display_classes_system.classes=%s',display_classes_system.classes)
    logging.info('np.unique(segmask)=\n%s',np.unique(segmask))
    
    unique_classes = sorted(np.unique(segmask))
    dict_classes_to_inds = {sem_class: i for i, sem_class in enumerate(unique_classes)}
    logging.info('dict_classes_to_inds=%s',dict_classes_to_inds)
    for sem_class in unique_classes:
        logging.info('color_map[dict_classes_to_inds[sem_class]]=%s',color_map[dict_classes_to_inds[sem_class]])
        colors[segmask == sem_class] = colors[segmask == sem_class]*args.alpha + color_map[dict_classes_to_inds[sem_class]]*(1-args.alpha)
        num_points = np.sum(segmask == sem_class)
        logging.info("Class %s has %s proportion of points, with %s points",sem_class, num_points / len(segmask), num_points)
                

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)

    if args.camera_json:
        camera_params = load_camera_parameters(args.camera_json)
        if camera_params:
            control = vis.get_view_control()
            control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
            logging.info("Applied camera parameters from JSON file")

    vis.run()
    vis.destroy_window()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--dataset_name', help='nuscenes or semantickitti', default="nuscenes")
    # parser.add_argument('--dataset_path', help='', default="/media/andrew/andrewSSD/datasets/nuscenes")
    # parser.add_argument('--nuscenes_mode', help='', default="trainval")
    parser.add_argument('--root', help='Root containing point_segmasks_gt.npy and registered_pointcloud.npy', required=True, type=str)
    parser.add_argument('--pointcloud_filename', help='', default="registered_pointcloud.las")
    parser.add_argument('--display_cs', help='', type=str, default=None, choices=["uda", 'dglss', "global_ext"])
    parser.add_argument('--gt_filename', help='', default="point_segmasks_gt.npy")
    parser.add_argument('--alpha', help='', default=0.5, type=float)
    parser.add_argument('--camera_json', type=str, help='Path to a JSON file containing camera parameters to set the view')
    parser.add_argument('--use_intensity_colors', action='store_true', help='Use intensity colors instead of RGB colors')
    args = parser.parse_args()

    plot_gt(args)
