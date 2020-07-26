import utils
import params
import cv2 as cv
import pdb
import numpy as np


def resize(downscaled_image, original_image, interpolation_method): 
    
    standard_resize = utils.resize_height_width_3d_image_standard(downscaled_image, int(downscaled_image.shape[1]), int(original_image.shape[2]), interpolation_method=interpolation_method)
    
    standard_resize = np.int16(standard_resize)
    
    ssim_standard, psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image)

    return ssim_standard, psnr_standard 


def read_images(test_path):

    add_to_path_gt = 'original_d'
    add_to_path_in = 'input_d_x%d' % scale
        
    test_images, lists_idx = utils.read_all_directory_images_from_directory_test_depth(test_path, add_to_path=add_to_path_in)  
    test_images_gt = utils.read_all_directory_images_from_directory_test_depth(test_path, add_to_path=add_to_path_gt, list_idx=lists_idx)
    
    return test_images_gt, test_images


def compute_performance_indeces(test_images_gt, test_images, interpolation_method): 
    num_images = 0
    ssim_standard_sum = 0
    psnr_standard_sum = 0
 
    for index in range(len(test_images)): 
        ssim_standard, psnr_standard = resize(test_images[index], test_images_gt[index], interpolation_method) 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images_gt[index].shape[0]
         
            
    return psnr_standard_sum/num_images, ssim_standard_sum/num_images 


interpolation_methods= {'INTER_LINEAR': cv.INTER_LINEAR,
                        'INTER_CUBIC': cv.INTER_CUBIC,
                        'INTER_LANCZOS4': cv.INTER_LANCZOS4,
                        'INTER_NEAREST': cv.INTER_NEAREST}
 
test_path = 'D:\\Research\\super-resolution\\datasets\\test'
scale = 2
test_images_gt, test_images = read_images(test_path)

for i in range(len(test_images)):
    test_images[i] = utils.process_image_gt(test_images[i])
    test_images_gt[i] = np.int16(utils.process_image_gt(test_images_gt[i]))

for interpolation_method in interpolation_methods.keys():
    psnr, ssim = compute_performance_indeces(test_images_gt, test_images, interpolation_methods[interpolation_method])
    print('interpolation method %s has ssim %f psnr %f' % (interpolation_method, ssim, psnr))


# interpolation method INTER_LINEAR has ssim 0.761849 psnr 39.794171
# interpolation method INTER_CUBIC has ssim 0.779637 psnr 40.386321
# interpolation method INTER_LANCZOS4 has ssim 0.776177 psnr 40.417771
# interpolation method INTER_NEAREST has ssim 0.748987 psnr 38.219267

# only 200 images
# interpolation method INTER_LINEAR has ssim 0.765367 psnr 42.654237
# interpolation method INTER_CUBIC has ssim 0.783579 psnr 43.346798
# interpolation method INTER_LANCZOS4 has ssim 0.780883 psnr 43.370962
# interpolation method INTER_NEAREST has ssim 0.755139 psnr 40.960190

# epoch 2 cnn D:\Research\super-resolution\datasets\test --- psnr = 42.18894417566856 ssim = 0.5545928152963565
# epoch 3 cnn D:\Research\super-resolution\datasets\test --- psnr = 43.02141657158047 ssim = 0.7716466596210669
# epoch 8 cnn D:\Research\super-resolution\datasets\test --- psnr = 44.25283611521931 ssim = 0.7487121732866311
# epoch 13 cnn D:\Research\super-resolution\datasets\test --- psnr = 45.09157188725168 ssim = 0.7528350801127718
# epoch 14 cnn D:\Research\super-resolution\datasets\test --- psnr = 44.608375728687484 ssim = 0.5681245345565314
# epoch 18 cnn D:\Research\super-resolution\datasets\test --- psnr = 45.69158402657238 ssim = 0.792154583611225
