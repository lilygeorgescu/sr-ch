import os
import numpy as np
import cv2 as cv
from scipy.signal import convolve
import pydicom
import nrrd
import pdb
from utils import create_folder

folder_name = 'D:\\Research\\super-resolution\\datasets\\test'
files_names = os.listdir(folder_name)
resize_factor = 2
input_folder_name = 'input_d_x%d' % resize_factor


for file_name in files_names:
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

    images = np.array(images)
    # from D x H x W -> H x W x D
    images = np.transpose(images, [1, 2, 0])
    # now H x W x D
    for i in range(images.shape[0]):
        image = images[i, :, :]
        create_folder(os.path.join(folder_name, file_name, 'original_d'))
        np.save(os.path.join(folder_name, file_name, 'original_d', '%05d.npy' % i), image)
        in_image = cv.resize(image, None, fx=1 / resize_factor, fy=1)
        np.save(os.path.join(folder_in, '%05d.npy' % i), in_image)
