import utils
import params
import cv2 as cv
import pdb


def resize(downscaled_image, original_image, interpolation_method): 
    
    standard_resize = utils.resize_height_width_3d_image_standard(downscaled_image,
                                                                  int(original_image.shape[1]),
                                                                  int(original_image.shape[2]),
                                                                  interpolation_method=interpolation_method)
    ssim_standard, psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image)
    return ssim_standard, psnr_standard 


def get_key_by_value(val):
    for key, value in interpolation_methods.items():
        if value == val:
            return key

            
def read_images(test_path):

    test_images_gt, folder_names_gt = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='original')
    test_images, _ = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='input_x%d' % scale)
    
    return test_images_gt, test_images, folder_names_gt


def compute_performance_indeces(test_images_gt, test_images, interpolation_method):
    num_images = 0
    ssim_standard_sum = 0
    psnr_standard_sum = 0
 
    for index in range(len(test_images)): 
        ssim_standard, psnr_standard = resize(test_images[index], test_images_gt[index], interpolation_method) 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images[index].shape[0]
      
    return psnr_standard_sum/num_images, ssim_standard_sum/num_images 
    
# interpolation_methods= {'INTER_LINEAR': cv.INTER_LINEAR,
                        # 'INTER_CUBIC': cv.INTER_CUBIC,
                        # 'INTER_LANCZOS4': cv.INTER_LANCZOS4,
                        # 'INTER_NEAREST': cv.INTER_NEAREST}


interpolation_methods = {'INTER_LANCZOS4': cv.INTER_LANCZOS4}

test_path = 'D:\\Research\\super-resolution\\datasets\\test'
scale = 2
test_images_gt, test_images, folder_names_gt = read_images(test_path)
for i in range(len(test_images)):
    is_ch = False
    if folder_names_gt[i] == '07' or folder_names_gt[i] == '11':
        is_ch = True
    print(is_ch)
    test_images[i] = utils.process_image_gt(test_images[i], is_ch)
    test_images_gt[i] = utils.process_image_gt(test_images_gt[i], is_ch)

for interpolation_method in interpolation_methods.keys():
    psnr, ssim = compute_performance_indeces(test_images_gt, test_images, interpolation_methods[interpolation_method])
    print('interpolation method %s has ssim %f psnr %f' % (interpolation_method, ssim, psnr))

# interpolation method INTER_LANCZOS4 has ssim 0.731972 psnr 45.543306 - with normalization
# interpolation method INTER_LANCZOS4 has ssim 0.730074 psnr 45.536157
# interpolation method INTER_NEAREST has ssim 0.745930 psnr 40.032056
# interpolation method INTER_LINEAR has ssim 0.696005 psnr 42.063468
# interpolation method INTER_CUBIC has ssim 0.735909 psnr 44.940752

# epoch 2 cnn D:\Research\super-resolution\datasets\test --- psnr = 47.24636465615944 ssim = 0.7537146614974554
# epoch 3 cnn D:\Research\super-resolution\datasets\test --- psnr = 48.15208423993577 ssim = 0.7314415282172549
# epoch 4 cnn D:\Research\super-resolution\datasets\test --- psnr = 48.663724668959446 ssim = 0.7483125980771304
# epoch 7 cnn D:\Research\super-resolution\datasets\test --- psnr = 48.6509703921561 ssim = 0.7494254511269222
# epoch 8 cnn D:\Research\super-resolution\datasets\test --- psnr = 48.88591108143797 ssim = 0.7423979462190043
# epoch 9 cnn D:\Research\super-resolution\datasets\test --- psnr = 48.297099567460464 ssim = 0.750783353229994
# epoch 10 cnn D:\Research\super-resolution\datasets\test --- psnr = 48.264408283425865 ssim = 0.7519349263799185
# epoch 11 cnn D:\Research\super-resolution\datasets\test --- psnr = 48.53501441570524 ssim = 0.756140298186663
# epoch 12 cnn D:\Research\super-resolution\datasets\test --- psnr = 48.06688022954905 ssim = 0.5597180080203259