import os
import numpy as np
import cv2 as cv
from scipy.signal import convolve

folder_name = 'D:\\disertatie\\materiale-radu\\raw-images\\test'
files_names = os.listdir(folder_name)
resize_factor = 2
input_folder_name = 'input_x%d' % resize_factor


for file_name in files_names:
    images_names = os.listdir(os.path.join(folder_name, file_name, 'original'))

    folder_in = os.path.join(folder_name, file_name, input_folder_name)
    if not os.path.isdir(folder_in):
        os.mkdir(folder_in)

    for image_name in images_names:
        if os.path.isdir(os.path.join(folder_name, image_name)):
            continue

        image_full_path = os.path.join(folder_name, file_name, 'original', image_name)

        image = np.load(image_full_path)
        in_image = cv.resize(image, None,  fx=1 / resize_factor, fy=1 / resize_factor)
        np.save(os.path.join(folder_in, image_name), in_image)
