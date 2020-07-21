import cv2 as cv
import numpy as np
from scipy.signal import convolve
import os

OUTPUT_PATH = '/media/usb/igeorgescu/super-resolution/ct_train'
MIN_VALUE = -1100
MAX_VALUE = 2500
dim_patch = 14
stride = 13
resize_factor = 2
input_folder_name = 'input_%d_%d' % (dim_patch, resize_factor)
gt_folder_name = 'gt_%d_%d' % (dim_patch, resize_factor)


def get_kernel(dim, sigma):
    x = cv.getGaussianKernel(dim, sigma)
    y = cv.getGaussianKernel(dim, sigma)
    res = x.dot(y.T)
    return res


def extract_patch_save_images(image, dim_patch, stride, resize_factor, folder_in, folder_gt, idx_image):
    h, w = image.shape
    images_in = []
    images_gt = []
    idx_image = idx_image + 1
    for i in range(0, h - dim_patch, stride):
        for j in range(0, w - dim_patch, stride):
            gt_patch = image[i:i + dim_patch, j:j + dim_patch]
            sigma = np.random.rand()
            kernel = get_kernel(3, sigma)

            if np.random.rand() < 0.5:
                in_patch = convolve(gt_patch, kernel, mode='same')
                in_patch = cv.resize(in_patch, (0, 0), fx=1 / resize_factor, fy=1 / resize_factor)
            else:
                in_patch = cv.resize(gt_patch, (0, 0), fx=1 / resize_factor, fy=1 / resize_factor)

            if np.sum(in_patch) == 0:
                continue

            images_in.append(in_patch)
            images_gt.append(gt_patch)

    np.save(os.path.join(folder_gt, '%05d.npy' % idx_image), np.array(images_gt))
    np.save(os.path.join(folder_in, '%05d.npy' % idx_image), np.array(images_in))

    return idx_image


def process_image(image, is_ch=False):
    image = np.float32(image)
    image = np.clip(image, MIN_VALUE, MAX_VALUE)
    if not is_ch:
        image -= MIN_VALUE
    image /= (MAX_VALUE - MIN_VALUE)
    return image
