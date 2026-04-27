"""
Script made by Andrew Caunes @ Logiroad.
Example use:
python3 -m visualize_las las_file.las
"""

import open3d as o3d
import laspy
import os
import numpy as np
import os
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

def main(args):
    visualize_las(pointclouds_folder=args.pointclouds_folder,
                  pointclouds_filenames=args.pointclouds_filenames,
                  downsample=args.downsample,
                  dont_translate_to_origin=args.dont_translate_to_origin,
                  frame_size=args.frame_size,
                  no_frame=args.no_frame)

def read_las_file(las_file, return_header=False):
    """Read a las file and return the points, colors, timestamps and intensity."""
    assert os.path.exists(las_file), f"File {las_file} not found."
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).T.astype(float)
    colors_from_intensity = np.array(las.intensity).astype(float) / 65535.0
    colors = np.vstack((colors_from_intensity, colors_from_intensity, colors_from_intensity)).T
    intensity = np.array(las.intensity)
    timestamps = np.array(las.gps_time)
    if not return_header:
        return points, colors, intensity, timestamps
    else:
        return points, colors, intensity, timestamps, las.header

def visualize_las(pointclouds_folder,
                  pointclouds_filenames,
                  downsample=False,
                  dont_translate_to_origin=False,
                  frame_size=3.0,
                  no_frame=False):
    
    input_file_paths = [os.path.join(pointclouds_folder, filename) for filename in pointclouds_filenames]
    for i, file_path in enumerate(input_file_paths):
        assert os.path.exists(file_path), f"File {file_path} not found."
        assert file_path.endswith('.las'), f"File {file_path} must be a .las file."

    plot_list = []
    offsets = np.array([0, 0, 0])
    for i, file_path in enumerate(input_file_paths):            
        logging.info('Loading pointcloud %s/%s, path=%s', i+1, len(input_file_paths), file_path)
        points, colors, timestamps, intensity = read_las_file(file_path)
        
        # Output some statistics
        logging.info('Point Cloud Statistics:')
        logging.info('  Min dimensions: %s', np.min(points, axis=0))
        logging.info('  Max dimensions: %s', np.max(points, axis=0))
        logging.info('  Shape: %s', points.shape)
        logging.info('  Max color values: %s', np.max(colors, axis=0))
        logging.info('  Min color values: %s', np.min(colors, axis=0))
                
       # Translate to origin for display        
        if i == 0 and not args.dont_translate_to_origin:
            logging.info("Setting points to origin for display without saturation.")
            offsets = points[0] #np.mean(points, axis=0)
        points = points - offsets
            
        # Create the point cloud
        logging.info('Creating point cloud...')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Downsample the point cloud
        if downsample:
            logging.info('Downsampling point cloud...')
            downpcd = pcd.voxel_down_sample(voxel_size=0.05)
            pcd = downpcd
        
        # Add the point cloud to the plot list
        plot_list.append(pcd)

    # Add reference frame
    if not no_frame:
        logging.info("Displaying %sm reference frame", frame_size )
        if not dont_translate_to_origin:
            origin = [0, 0, 0]
        else:
            origin = points[0].copy()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=origin)
        plot_list.append(coordinate_frame)
    
    try:
        o3d.visualization.draw_geometries(plot_list)
    except Exception as e:
        logging.error('Error during draw_geometries: %s', e)
    logging.info('Finished drawing geometries.')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--pointclouds_folder', help='Pointcloud folder in which files are located', type=str, required=True)
    parser.add_argument('--pointclouds_filenames', help='Pointcloud filenames to visualize', nargs='+', type=str, required=True)
    parser.add_argument('--downsample', action='store_true', help='Downsample the point cloud using voxel downsampling')
    parser.add_argument('--dont_translate_to_origin', action="store_true", help='Dont translate the point cloud to origin for display (Used for resolution problems.)')
    parser.add_argument('--frame_size', default=3.0, type=float, help='Size of the displayed reference frame (at the origin)')
    parser.add_argument('--no_frame', action="store_true", help='Do not display the reference frame')

    args = parser.parse_args()
    main(args)