import os
import numpy as np
import cv2 as cv
from scipy.signal import convolve
import pydicom
import nrrd
import pdb

folder_name = 'D:\\Research\\super-resolution\\datasets\\test'
files_names = os.listdir(folder_name)
resize_factor = 4
input_folder_name = 'input_x%d' % resize_factor


for file_name in files_names:
    images_names = os.listdir(os.path.join(folder_name, file_name, 'original'))

    folder_in = os.path.join(folder_name, file_name, input_folder_name)
    if not os.path.isdir(folder_in):
        os.mkdir(folder_in)

    if images_names[0].find('.npy') != -1:
        for image_name in images_names:
            if os.path.isdir(os.path.join(folder_name, image_name)):
                continue

            image_full_path = os.path.join(folder_name, file_name, 'original', image_name)

            image = np.load(image_full_path)
            in_image = cv.resize(image, None,  fx=1 / resize_factor, fy=1 / resize_factor)
            np.save(os.path.join(folder_in, image_name), in_image)

    elif images_names[0].find('.dcm') != -1:
        for image_name in images_names:
            if os.path.isdir(os.path.join(folder_name, image_name)):
                continue

            image_full_path = os.path.join(folder_name, file_name, 'original', image_name)

            image = pydicom.dcmread(image_full_path).pixel_array
            np.save(image_full_path[:-4] + ".npy", image)
            in_image = cv.resize(image, None,  fx=1 / resize_factor, fy=1 / resize_factor)
            image_name = image_name[:-4]
            np.save(os.path.join(folder_in, image_name + ".npy"), in_image)

    elif images_names[0].find('.nrrd') != -1:
        readdata, header = nrrd.read(os.path.join(folder_name, file_name, 'original', images_names[0]))
        for i in range(readdata.shape[2]):
            image = readdata[:, :, i]
            np.save(os.path.join(folder_name, file_name, 'original', '%05d.npy' % i), image)
            in_image = cv.resize(image, None, fx=1 / resize_factor, fy=1 / resize_factor)

            np.save(os.path.join(folder_in, '%05d.npy' % i), in_image)