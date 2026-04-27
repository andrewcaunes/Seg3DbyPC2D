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
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import laspy
from seg_3D_by_2D.utils.utils import find_roots_dataset
from seg_3D_by_2D.utils.utils import nuscenes_available_scenes

def main(args):
    logging.info("args = %s", args)
    nusc = NuScenes(version='v1.0-'+args.nuscenes_mode, dataroot=args.nuscenes_path, verbose=True)
    analyse_times(scenes=args.nuscenes_scenes, 
                  num_scenes=args.num_scenes,
                     nusc=nusc)

def get_scene_from_name(nusc, scene_name):
    """Get the scene from the scene name."""
    scenes = [s for s in nusc.scene if s["name"] == scene_name]
    # logging.info('scene_name=%s, len=%s',scene_name, len(scenes))
    # logging.info('scene=\n%s',scenes)
    # assert len(scene) == 1, "Error: Invalid scene %s" % scene_name
    return scenes

def analyse_times(scenes,
                  num_scenes, 
                  nusc ):
    if scenes is None:
        scenes = [int(nuscenes_available_scenes[i].split('-')[1]) for i in range(num_scenes)]
    logging.info('scenes=\n%s',scenes)
    timestamps = []
    for scene in scenes:
        scene_name = "scene-{:04d}".format(scene)
        logging.info('scene=\n%s',scene_name)
        # if input_output_folder is None:
        #     output_folder = "./nuscenes_images"
        #     output_folder = os.path.join(output_folder, scene_name)
        # os.makedirs(output_folder, exist_ok=True)
        # Get scene
        scene_data = get_scene_from_name(nusc, scene_name)[0]
        # scene_data = nusc.scene[args.scene]
        # for k, v in scene_data.items():
            # logging.info(f"{k} = {v}")
        # Get sample tokens
        first_sample_token = scene_data['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)
        last_sample_token = scene_data['last_sample_token']
        last_sample = nusc.get('sample', last_sample_token)
        timestamps.append((scene_name, first_sample['timestamp'], last_sample['timestamp']))
        # last_timestamps.append((scene_name, sample['timestamp']))
        # first_timestamps.append((scene_name, sample['timestamp']))
        # logging.info('sample["timestamp"]=%s',sample["timestamp"])
        # cam_front_token = sample['data'][args.cam]
    # logging.info('timestamps=\n%s',timestamps)
    sorted_timestamps = sorted(timestamps, key=lambda x: x[1])

    for i in range(len(sorted_timestamps)):
        scene, first_timestamp, _ = sorted_timestamps[i]
        if i == 0:
            previous_last_timestamp = first_timestamp
        else:
            _, _, previous_last_timestamp = sorted_timestamps[i-1]
        diff_with_previous = (first_timestamp-previous_last_timestamp)/1e6
        logging.info('%s, %s = %s, diff_with_previous=%.3f',i, scene, first_timestamp/1e6, diff_with_previous)    
        # cam_front_data = nusc.get('sample_data', cam_front_token)
        # filename = cam_front_data['filename']
        # timestamps.append(cam_front_data['timestamp'])
        # img_path = f"{nusc.dataroot}/{filename}"
        # ti
        # # Loop over samples
        # logging.info('args.cam=\n%s',args.cam)
        # while sample_token != '':
        #     # Get sample
        #     sample = nusc.get('sample', sample_token)
        #     cam_front_token = sample['data'][args.cam]
        #     cam_front_data = nusc.get('sample_data', cam_front_token)
        #     filename = cam_front_data['filename']
        #     img_path = f"{nusc.dataroot}/{filename}"
        #     # Copy image
        #     out_img_path = os.path.join(output_folder, os.path.basename(filename))
        #     # create dir
        #     shutil.copy(img_path, out_img_path)
        #     # Get next sample

        #     # lidar_token = sample['data']['LIDAR_TOP']
        #     # lidar_data = nusc.get('sample_data', lidar_token)
        #     # filename = lidar_data['filename']
        #     # file_index = int(lidar_data["filename"].split("_")[-1].split(".")[0])
        #     # my_pred_path = "/media/andrew/andrewSSD/datasets/nuscenes/pred_lidarseg/trainval/"+str(file_index)+"_pred_lidarseg.bin"
        #     # true_lidarseg_path = nusc.get('lidarseg', lidar_token)['filename']
        #     # true_lidarseg_path = os.path.join(nusc.dataroot, true_lidarseg_path)
        #     # # nusc.render_pointcloud_in_image(sample['token'],
        #     # #                         pointsensor_channel='LIDAR_TOP',
        #     # #                         camera_channel='CAM_FRONT',
        #     # #                         render_intensity=False,
        #     # #                         show_lidarseg=True,
        #     # #                         # filter_lidarseg_labels=[22, 23],
        #     # #                         show_lidarseg_legend=True,
        #     # #                         lidarseg_preds_bin_path=true_lidarseg_path)
        #     # nusc.render_pointcloud_in_image(sample['token'],
        #     #                         pointsensor_channel='LIDAR_TOP',
        #     #                         camera_channel='CAM_FRONT',
        #     #                         render_intensity=False,
        #     #                         show_lidarseg=True,
        #     #                         show_lidarseg_legend=True,
        #     #                         lidarseg_preds_bin_path=my_pred_path)

        #     sample_token = sample['next']



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--output_folder', help='', default=None)
    parser.add_argument('--nuscenes_scenes', help='List of scenes of nuscenes to use (int indices)', nargs='+', type=int, default=None)
    parser.add_argument('--num_scenes', help='List of scenes of nuscenes to use (int indices)', type=int, default=100)
    parser.add_argument('--nuscenes_path', help='', default="/media/andrew/andrewSSD/datasets/nuscenes")
    parser.add_argument('--nuscenes_mode', help='', default="trainval")
    parser.add_argument('--cam', help='', default="CAM_FRONT", type=str)

    args = parser.parse_args()

    main(args)