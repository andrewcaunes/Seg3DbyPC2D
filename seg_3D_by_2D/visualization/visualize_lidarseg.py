"""
Script made by Andrew Caunes.
Example use:
python3 -m seg_3D_by_2D.visualization.visualize_lidarseg 
    --lidarseg_folder /media/andrew/T9/jean-zay/projects/ia4markings/results/test_point_segmasks_to_scans_3/trainval/ 
    --scene_name scene-0012 
    --nuscenes_path /media/andrew/T9/datasets/nuscenes/ 
    --out_path /media/andrew/T9/jean-zay/projects/ia4markings/results/test_point_segmasks_to_scans_3/pred_visualization
"""
import os
import shutil
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np

from nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud

# from seg_3D_by_2D.core.classes_dicts import global_to_lidarseg_dict, uda_to_lidarseg_dict


def main(args):
    logging.info("args = %s", args)
    visualize_lidarseg(scene_name=args.scene_name,
                        lidarseg_folder=args.lidarseg_folder,
                        nuscenes_path=args.nuscenes_path,
                        out_path=args.out_path,
                        num_imgs=args.num_imgs,
                        lidarseg_suffix=args.lidarseg_suffix,
                        output_suffix=args.output_suffix)

def get_scene_from_name(nusc, scene_name):
    """Get the scene from the scene name."""
    logging.info('scene_name=\n%s',scene_name)
    scenes = [s for s in nusc.scene if s["name"] == scene_name]
    logging.info('scene=\n%s',scenes)
    # assert len(scene) == 1, "Error: Invalid scene %s" % scene_name
    if len(scenes) != 1:
        logging.info('scenes %s not length 1, for %s', scenes, scene_name)
        return None
    return scenes[0]

def visualize_lidarseg(scene_name,
                       lidarseg_folder,
                       nuscenes_path,
                       out_path=None,
                       num_imgs=10,
                       lidarseg_suffix="_lidarseg.bin",
                       output_suffix="_gt.png",
                       ):
    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        
    lidarseg_folder = os.path.join(lidarseg_folder, args.nuscenes_mode)
    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_path, verbose=True)
    scene = get_scene_from_name(nusc, scene_name)
    sample_token = scene["first_sample_token"]
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    i=0
    while lidar_token != "" and i<num_imgs:
        logging.info('lidar_token=\n%s',lidar_token)

        # render predicted lidarseg
        lidarseg_path = os.path.join(lidarseg_folder, str(lidar_token) + lidarseg_suffix)
        out_pred_img_path = os.path.join(out_path, str(lidar_token) + output_suffix) if out_path is not None else None
        nusc.render_pointcloud_in_image(sample['token'],
                                        pointsensor_channel='LIDAR_TOP',
                                        camera_channel='CAM_FRONT',
                                        render_intensity=False,
                                        show_lidarseg=True,
                                        show_lidarseg_legend=True,
                                        lidarseg_preds_bin_path=lidarseg_path,
                                        out_path=out_pred_img_path)
        # render ground truth lidarseg
        out_gt_img_path = os.path.join(out_path, str(lidar_token) + "_gt.png") if out_path is not None else None
        nusc.render_pointcloud_in_image(sample['token'],
                                        pointsensor_channel='LIDAR_TOP',
                                        camera_channel='CAM_FRONT',
                                        render_intensity=False,
                                        show_lidarseg=True,
                                        show_lidarseg_legend=True,
                                        out_path=out_gt_img_path)
        

        sample = nusc.get('sample', sample['next'])
        lidar_token = sample['data']['LIDAR_TOP']
        i+=1


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('--global_root', help='', default="results/nuscenes/")
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--scene_name', help='', required=True)
    parser.add_argument('--lidarseg_folder', help='', default="/media/andrew/T9/datasets/nuscenes/lidarseg")
    # parser.add_argument('--exp_name', help='', default="large_ref")
    parser.add_argument('--nuscenes_path', help='', default="/media/andrew/T9/datasets/nuscenes")
    parser.add_argument('--nuscenes_mode', help='', default="v1.0-trainval")
    parser.add_argument('--out_path', help='', default=None)
    parser.add_argument('--num_imgs', help='', default=10, type=int)
    parser.add_argument('--lidarseg_suffix', help='', default="_lidarseg.bin")
    parser.add_argument('--output_suffix', help='', default="_gt.png")
    # parser.add_argument('--debug', help='', action='store_true')
    # parser.add_argument('--num_proc', help='', default=1, type=int)
    # parser.add_argument('--start_from', help='', default=None, type=int)
    # parser.add_argument('--end_at', help='', default=None, type=int)
    # parser.add_argument('--not_shared', help='', action='store_true')
    # parser.add_argument('--seg2d_results_foldername', help='', default="seg2d_results")
    # parser.add_argument('--point_segmasks_filename', help='', default="point_segmasks_seg2d.npy")

    args = parser.parse_args()

    main(args)