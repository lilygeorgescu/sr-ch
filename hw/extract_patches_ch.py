import os
import numpy as np
import cv2 as cv
import pdb
import nrrd

from utils import *

folder_name = '/home/igeorgescu/datasets/super_res/ch/train'
files = os.listdir(folder_name)

for file_name in files:
    base_folder_image = os.path.join(OUTPUT_PATH, file_name)
    folder_in = os.path.join(base_folder_image, input_folder_name)
    folder_gt = os.path.join(base_folder_image, gt_folder_name)

    if not os.path.isdir(base_folder_image):
        os.mkdir(base_folder_image)

    if not os.path.isdir(folder_in):
        os.mkdir(folder_in)

    if not os.path.isdir(folder_gt):
        os.mkdir(folder_gt)

    idx_image = 0
    print('%s' % file_name)
    images_names = os.listdir(os.path.join(folder_name, file_name, 'original'))

    for image_name in images_names:
        image = np.load(os.path.join(folder_name, file_name, 'original', image_name))
        image = process_image(image, is_ch=True)
        idx_image = extract_patch_save_images(image, dim_patch, stride, resize_factor, folder_in, folder_gt, idx_image)
