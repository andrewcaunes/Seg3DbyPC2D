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
from nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
# from pyquaternion import Quaternion
# import laspy

def main(args):
    logging.info("args = %s", args)
    nusc = NuScenes(version='v1.0-'+args.nuscenes_mode, dataroot=args.nuscenes_path, verbose=True)
    # Determine which cameras to use.
    # If --cams is provided, use that. Otherwise fall back to --cam if given.
    # If neither is provided, use all standard NuScenes cameras.
    if getattr(args, "cams", None):
        cams = args.cams
    elif getattr(args, "cam", None):
        cams = [args.cam]
    else:
        cams = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

    show_images(scenes=args.nuscenes_scenes, 
                     nusc=nusc,
                     output_folder=args.output_folder,
                     cams=cams)

def get_scene_from_name(nusc, scene_name):
    """Get the scene from the scene name."""
    scenes = [s for s in nusc.scene if s["name"] == scene_name]
    # logging.info('scene_name=%s, len=%s',scene_name, len(scenes))
    # logging.info('scene=\n%s',scenes)
    # assert len(scene) == 1, "Error: Invalid scene %s" % scene_name
    return scenes

def show_images(scenes, nusc, output_folder, cams):
    # Save all nuscenes images of a scene to a folder
    # Create folder
    input_output_folder = output_folder
    for scene in scenes:
        scene_name = "scene-{:04d}".format(scene)
        logging.info('scene=\n%s',scene_name)
        # Base directory for all scenes
        base_output = input_output_folder if input_output_folder is not None else "./nuscenes_images"
        # One subdirectory per scene
        output_folder = os.path.join(base_output, scene_name)
        os.makedirs(output_folder, exist_ok=True)
        # Get scene
        scene_data = get_scene_from_name(nusc, scene_name)
        # logging.info('scene_data=\n%s',scene_data)
        if isinstance(scene_data, list):
            scene_data = scene_data[0]
        # scene_data = nusc.scene[args.scene]
        # for k, v in scene_data.items():
            # logging.info(f"{k} = {v}")
        # Get sample tokens
        sample_token = scene_data['first_sample_token']
        # Loop over samples
        while sample_token != '':
            # Get sample
            sample = nusc.get('sample', sample_token)
            # Loop over requested cameras
            for cam in cams:
                cam_token = sample['data'][cam]
                cam_data = nusc.get('sample_data', cam_token)
                filename = cam_data['filename']
                img_path = f"{nusc.dataroot}/{filename}"
                # Copy image
                out_img_path = os.path.join(output_folder, os.path.basename(filename))
                shutil.copy(img_path, out_img_path)
            # Get next sample

            # lidar_token = sample['data']['LIDAR_TOP']
            # lidar_data = nusc.get('sample_data', lidar_token)
            # filename = lidar_data['filename']
            # file_index = int(lidar_data["filename"].split("_")[-1].split(".")[0])
            # my_pred_path = "/media/andrew/andrewSSD/datasets/nuscenes/pred_lidarseg/trainval/"+str(file_index)+"_pred_lidarseg.bin"
            # true_lidarseg_path = nusc.get('lidarseg', lidar_token)['filename']
            # true_lidarseg_path = os.path.join(nusc.dataroot, true_lidarseg_path)
            # # nusc.render_pointcloud_in_image(sample['token'],
            # #                         pointsensor_channel='LIDAR_TOP',
            # #                         camera_channel='CAM_FRONT',
            # #                         render_intensity=False,
            # #                         show_lidarseg=True,
            # #                         # filter_lidarseg_labels=[22, 23],
            # #                         show_lidarseg_legend=True,
            # #                         lidarseg_preds_bin_path=true_lidarseg_path)
            # nusc.render_pointcloud_in_image(sample['token'],
            #                         pointsensor_channel='LIDAR_TOP',
            #                         camera_channel='CAM_FRONT',
            #                         render_intensity=False,
            #                         show_lidarseg=True,
            #                         show_lidarseg_legend=True,
            #                         lidarseg_preds_bin_path=my_pred_path)

            sample_token = sample['next']



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--output_folder', help='', default="nuscenes_images")
    parser.add_argument('--nuscenes_scenes', help='List of scenes of nuscenes to use (int indices)', nargs='+', type=int, default=None)
    parser.add_argument('--nuscenes_path', help='', default="/media/andrew/T9/datasets/nuscenes")
    parser.add_argument('--nuscenes_mode', help='', default="trainval")
    parser.add_argument('--cam', help='Single camera channel (backward compatible)', default=None, type=str)
    parser.add_argument('--cams', help='List of camera channels to use', nargs='+', default=None)

    args = parser.parse_args()

    main(args)