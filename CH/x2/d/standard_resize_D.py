import utils
import params
import cv2 as cv
import pdb
import numpy as np

def resize(downscaled_image, original_image, interpolation_method): 
    
    standard_resize = utils.resize_height_width_3d_image_standard(downscaled_image, int(downscaled_image.shape[1]), int(original_image.shape[2]), interpolation_method=interpolation_method)
      
     
    ssim_standard, psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image)

    return ssim_standard, psnr_standard 
        
def read_images(test_path):

    if transposed_2_1:
        add_to_path_gt = 'transposed_2_1'
        add_to_path_in = 'input_d_2_1_x%d' % scale 
    else:
        add_to_path_gt = 'transposed'
        add_to_path_in = 'input_d_x%d' % scale
        
    test_images, lists_idx = utils.read_all_directory_images_from_directory_test_depth(test_path, add_to_path=add_to_path_in)  
    test_images_gt = utils.read_all_directory_images_from_directory_test_depth(test_path, add_to_path=add_to_path_gt, list_idx=lists_idx)
    
    return test_images_gt, test_images
     
def compute_performance_indeces(test_images_gt, test_images, interpolation_method): 
    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images)): 
        ssim_standard, psnr_standard = resize(test_images[index], test_images_gt[index], interpolation_method) 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images_gt[index].shape[0]
         
            
    return psnr_standard_sum/num_images, ssim_standard_sum/num_images 
    
interpolation_methods= {'INTER_LINEAR': cv.INTER_LINEAR,
                        'INTER_CUBIC': cv.INTER_CUBIC,
                        'INTER_LANCZOS4': cv.INTER_LANCZOS4,
                        'INTER_NEAREST': cv.INTER_NEAREST}
 
test_path = 'C:\\Research\\SR\\medical images\\namic\\images-testing\\t1w'
scale = 2
transposed_2_1 = False
test_images_gt_2_1, test_images_2_1 = read_images(test_path)

transposed_2_1 = True
test_images_gt, test_images = read_images(test_path)

test_images_gt += test_images_gt_2_1
test_images += test_images_2_1

for interpolation_method in interpolation_methods.keys():
    psnr, ssim = compute_performance_indeces(test_images_gt, test_images, interpolation_methods[interpolation_method])
    print('interpolation method %s has ssim %f psnr %f' % (interpolation_method, ssim, psnr))



