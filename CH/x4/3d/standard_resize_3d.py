import utils
import cv2 as cv
import numpy as np
import pdb


def resize(downscaled_image, original_image, interpolation_method):
    standard_resize = utils.resize_3d_image_standard(downscaled_image, new_depth=original_image.shape[0],
                                                     new_height=original_image.shape[1],
                                                     new_width=original_image.shape[2],
                                                     interpolation_method=interpolation_method)
    
    ssim_standard, psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image)
    return ssim_standard, psnr_standard 
     

def compute_performance_indeces(test_images_gt, test_images, interpolation_method): 
    num_images = 0 
    ssim_standard_sum = 0
    psnr_standard_sum = 0
 
    for index in range(len(test_images)): 
        ssim_standard, psnr_standard = resize(test_images[index], test_images_gt[index], interpolation_method) 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images_gt[index].shape[0] 
                
    return psnr_standard_sum/num_images, ssim_standard_sum/num_images 


def read_images(test_path):
    test_images_gt = utils.read_3d_images_from_directory_test(test_path, add_to_path='original_3d//%dx' % scale_factor)
    test_images = utils.read_3d_images_from_directory_test(test_path, add_to_path='input_3d//%dx' % scale_factor, is_test=True)

    return test_images_gt, test_images    

test_path = 'D:\\Research\\super-resolution\\datasets\\test'
scale_factor = 2
test_images_gt, test_images = read_images(test_path)

# interpolation_methods= {'INTER_LANCZOS4': cv.INTER_LANCZOS4}
                         
    
interpolation_methods= {'INTER_LANCZOS4': cv.INTER_LANCZOS4,
                        'INTER_CUBIC': cv.INTER_CUBIC,
                        'INTER_LINEAR': cv.INTER_LINEAR,
                        'INTER_NEAREST': cv.INTER_NEAREST}

for interpolation_method in interpolation_methods.keys():

    psnr, ssim = compute_performance_indeces(test_images_gt, test_images, interpolation_methods[interpolation_method])
    print('interpolation method %s has ssim %f psnr %f' % (interpolation_method, ssim, psnr))

# interpolation method INTER_LANCZOS4 has ssim 0.636653 psnr 42.062030
# interpolation method INTER_CUBIC has ssim 0.643121 psnr 41.696603
# interpolation method INTER_LINEAR has ssim 0.598780 psnr 39.259366
# interpolation method INTER_NEAREST has ssim 0.642384 psnr 37.348054