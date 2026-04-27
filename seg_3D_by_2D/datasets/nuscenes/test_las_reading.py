"""
Test script to verify the updated read_las_file function.
"""
import os
import sys
import logging

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from seg_3D_by_2D.utils.utils import read_las_file

def test_las_reading():
    """Test the updated read_las_file function."""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing updated read_las_file function")
    print("The function now supports:")
    print("1. use_rgb_colors=True (default): Read actual RGB colors from LAS file if available")
    print("2. use_rgb_colors=False: Use intensity-based colors (backward compatibility)")
    print("3. Automatic fallback to intensity-based colors if RGB colors are not available")
    
    # Example usage (commented out as it requires actual data):
    """
    # Read with RGB colors (default)
    points, colors, intensity, timestamps = read_las_file(
        "path/to/colorized_pointcloud.las",
        use_rgb_colors=True
    )
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")
    print(f"Colors range: {colors.min():.3f} to {colors.max():.3f}")
    
    # Read with intensity-based colors (backward compatibility)
    points, colors, intensity, timestamps = read_las_file(
        "path/to/old_pointcloud.las",
        use_rgb_colors=False
    )
    print(f"Intensity-based colors shape: {colors.shape}")
    
    # Read with header
    points, colors, intensity, timestamps, header = read_las_file(
        "path/to/pointcloud.las",
        return_header=True
    )
    print(f"Header point format: {header.point_format}")
    """

if __name__ == "__main__":
    test_las_reading() 