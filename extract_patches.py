import os
import numpy as np
import cv2 as cv
import pdb
from scipy.signal import convolve


def get_kernel(dim, sigma):
    x = cv.getGaussianKernel(dim, sigma)
    y = cv.getGaussianKernel(dim, sigma)
    res = x.dot(y.T)
    return res


def extract_patch_save_images(image, dim_patch, stride, resize_factor, folder_in, folder_gt, idx_image):
    h, w = image.shape

    for i in range(0, h - dim_patch, stride):
        for j in range(0, w - dim_patch, stride):
            idx_image = idx_image + 1
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

            np.save(os.path.join(folder_gt, '%05d.npy' % idx_image), gt_patch)
            np.save(os.path.join(folder_in, '%05d.npy' % idx_image), in_patch)

    return idx_image


folder_name = '/home/igeorgescu/datasets/super_res/ch/train'
files = os.listdir(folder_name)
dim_patch = 14
stride = 13
resize_factor = 2
input_folder_name = 'input_%d_%d' % (dim_patch, resize_factor)
gt_folder_name = 'gt_%d_%d' % (dim_patch, resize_factor)

for file_name in files:
    src_folder = os.path.join(folder_name, file_name, 'original')
    images_names = os.listdir(src_folder)
    folder_in = os.path.join(folder_name, file_name, input_folder_name)
    folder_gt = os.path.join(folder_name, file_name, gt_folder_name)

    if not os.path.isdir(folder_in):
        os.mkdir(folder_in)

    if not os.path.isdir(folder_gt):
        os.mkdir(folder_gt)

    idx_image = 0
    print('%s' % file_name)
    for image_name in images_names:
        if os.path.isdir(os.path.join(src_folder, image_name)):
            continue

        image_full_path = os.path.join(src_folder, image_name)
        image = np.load(image_full_path)
        idx_image = extract_patch_save_images(image, dim_patch, stride, resize_factor, folder_in, folder_gt, idx_image)

