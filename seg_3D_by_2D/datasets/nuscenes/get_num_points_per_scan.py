"""
Script made by Andrew Caunes.
Example use:
python script.py ...
"""
import os
import shutil
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
from nuscenes.nuscenes import NuScenes
from seg_3D_by_2D.utils.utils import get_unique_file_index_sk
import json
from tqdm import tqdm


def get_num_points(pointcloud_filename,
                   dataset_name="nuscenes"):
    # Get the file size in bytes
    file_size = os.path.getsize(pointcloud_filename)
    
    if dataset_name == "nuscenes":
        K = 5 
    elif dataset_name=="semantickitti" or dataset_name=="synlidar":
        K = 4
    
    # Calculate the number of points
    num_points = file_size // (np.dtype(np.float32).itemsize * K)
    return num_points


def get_num_points_ns(args):
    num_points_per_scan = {}
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataset_path, verbose=True)
    for j, scene in enumerate(nusc.scene):
        logging.info('scene=%s',scene)
        sample_token = scene["first_sample_token"]
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        i=0
        # Iterate over all scans (sweeps).
        while lidar_token != "":
            if i % 100 == 0 :
                logging.info('%s iter %s', scene["name"], i)
                # if file_index is not None:
                #     num_points_per_scan[file_index]
            i += 1
            # Get lidar data
            lidar_data = nusc.get('sample_data', lidar_token)
            
            # Extract file index
            file_index = int(lidar_data["filename"].split("_")[-1].split(".")[0])
            if file_index in num_points_per_scan or str(file_index) in num_points_per_scan:
                lidar_token = lidar_data["next"]
                continue
            
            # Construct pointcloud filename
            pointcloud_filename = os.path.join(nusc.dataroot, lidar_data["filename"])
            
            num_points_per_scan[file_index] = get_num_points(pointcloud_filename)
            
            # Move to the next lidar token
            lidar_token = lidar_data["next"]
            
            # End timing for the entire operation
        if j % 10 == 0:
            with open(args.output_path, 'w') as f:
                json.dump(num_points_per_scan, f)
                logging.info('Saving at scene-%s',j)
    
    with open(args.output_path, 'w') as f:
        json.dump(num_points_per_scan, f)

def get_num_points_sk(args):
    num_points_per_scan = {}
    
    list_file_indices = []
    
    sequences = range(args.start_sequence, args.end_sequence + 1)
    for sequence in sequences:
        sequence_str = str(sequence).zfill(2)
        sequence_path = os.path.join(args.dataset_path, 'sequences', sequence_str)
        scans_folder = os.path.join(sequence_path, "velodyne")
        
        if not os.path.exists(scans_folder):
            logging.warning(f"Sequence {sequence_str} not found at {scans_folder}")
            continue
            
        scans_filenames = [f for f in os.listdir(scans_folder) if f.endswith(".bin")]
        scans_filenames = sorted(scans_filenames, key=lambda x: int(x.split(".")[0]))
        
        for i, scan_filename in tqdm(enumerate(scans_filenames), desc=f"Processing sequence {sequence_str}", total=len(scans_filenames)):
            # Unique file_index for each file in each sequence (defined in prepare_semantickitti.py)
            file_index = get_unique_file_index_sk(sequence_str, scan_filename)
            
            list_file_indices.append(file_index)
            if str(file_index) in num_points_per_scan:
                raise ValueError(f"file_index={file_index} already in num_points_per_scan")
                
            scan_path = os.path.join(scans_folder, scan_filename)
            num_points_per_scan[str(file_index)] = get_num_points(scan_path, dataset_name="semantickitti")
    assert len(num_points_per_scan) == len(list_file_indices), f"len(num_points_per_scan)={len(num_points_per_scan)} != len(list_file_indices)={len(list_file_indices)}"
    
    with open(args.output_path, 'w') as f:
        logging.info(f"Saving num_points_per_scan to {args.output_path}")
        json.dump(num_points_per_scan, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_path', help='', default=None)
    parser.add_argument('--output_path', help='', required=True)
    
    parser.add_argument('--dataset_path', help='', default="/media/andrew/T9/datasets/nuscenes")
    parser.add_argument('--nuscenes_mode', help='', default="trainval")
    
    # Add SemanticKITTI specific arguments
    parser.add_argument('--start_sequence', help='Starting sequence number', type=int, default=0)
    parser.add_argument('--end_sequence', help='Ending sequence number', type=int, default=10)
    parser.add_argument('--dataset_name', help='Dataset type: nuscenes or semantickitti', choices=['nuscenes', 'semantickitti'], required=True)
    
    args = parser.parse_args()
    
    if args.dataset_name == 'nuscenes':
        get_num_points_ns(args)
    else:
        logging.info("Processing SemanticKITTI dataset")
        get_num_points_sk(args)