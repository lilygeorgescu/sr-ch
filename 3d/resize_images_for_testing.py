import os
import numpy as np
import cv2 as cv
from scipy.signal import convolve
import pydicom
import nrrd
import pdb
from utils import create_folder, resize_3d_image_standard

folder_name = 'D:\\Research\\super-resolution\\datasets\\test'
files_names = os.listdir(folder_name)
resize_factor = 2
input_folder_name = 'input_3d_x%d' % resize_factor


for file_name in files_names:
    print(file_name)
    images_names = os.listdir(os.path.join(folder_name, file_name, 'original'))

    folder_in = os.path.join(folder_name, file_name, input_folder_name)
    if not os.path.isdir(folder_in):
        os.mkdir(folder_in)
    images = []

    for image_name in images_names:
        if image_name.find('.npy') == -1:
            continue
        image = np.load(os.path.join(folder_name, file_name, 'original', image_name))
        images.append(image)

    # now D x H x W
    images = np.array(images)
    d, h, w = images.shape
    input_images = resize_3d_image_standard(images, int(d // resize_factor), int(h // resize_factor), int(w // resize_factor))

    create_folder(os.path.join(folder_name, file_name, 'original_3d'))
    np.save(os.path.join(folder_name, file_name, 'original_3d', '%dx' % resize_factor, '3d_image.npy'), images)

    create_folder(os.path.join(folder_name, file_name, 'input_3d'))
    np.save(os.path.join(folder_name, file_name, 'input_3d', '%dx' % resize_factor, '3d_image.npy'), input_images)