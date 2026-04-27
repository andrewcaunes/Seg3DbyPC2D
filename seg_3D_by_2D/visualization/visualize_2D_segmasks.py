"""
Script made by Andrew Caunes.
Visualize in random colors segmasks that are originally in integer format (too dark).
Example use:
python visualize_2D_segmasks.py --folder /path/to/folder/with/2D/segmasks
"""
import os
import shutil
import argparse
import logging
from matplotlib import pyplot as plt
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import cv2
from PIL import Image

def main(args):
    logging.info("args = %s", args)
    
    classes=('background', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky')#, 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 #'motorcycle', 'bicycle'),
    logging.info('len(classes)=\n%s',len(classes))
    palette=[[20, 20, 20], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180]]
    visualize(args.folder, classes, palette)

def visualize(folder, classes=None, palette=None):
    try:
        sorted_images = sorted(os.listdir(folder), key=lambda x: int(x.split('.')[0]))
    except ValueError:
        sorted_images = sorted(os.listdir(folder))
    image_paths = [os.path.join(folder, f) for f in sorted_images if f.endswith('.png')]
    for image_path in image_paths:
        logging.info('image_path=\n%s',image_path)       
         # Load the image
        image = Image.open(image_path)

        # Convert image to numpy array
        image_np = np.array(image)

        # Check if the number of classes matches the palette length
        # logging.info('len(classes)=\n%s',len(classes))
        # logging.info('len(palette)=\n%s',len(palette))
        if len(classes) != len(palette):
            raise ValueError("The length of classes and palette must be the same")

        # Create an empty RGB image
        colored_image = np.zeros((*image_np.shape, 3), dtype=np.uint8)

        # Map each class to its corresponding color
        for i, color in enumerate(palette):
            colored_image[image_np == i] = color

        # Plot the image in fullsize
        plt.figure(figsize=(12, 12))
        plt.imshow(colored_image)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--folder', help="", required=True)
    args = parser.parse_args()

    main(args)