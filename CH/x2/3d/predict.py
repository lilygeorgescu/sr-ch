import cv2 as cv
import numpy as np
import tensorflow as tf
 
import utils
import params
import pdb

config = tf.ConfigProto(
        device_count={'GPU': 1}
    ) 


def resize_h_w(downscaled_image, original_image=None):    
    tf.reset_default_graph()

    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    
    if is_v2:
        _, output = nets_hs.SRCNN_late_upscaling_H_W(input) 
    else:
        output = nets_hs.SRCNN_late_upscaling_H_W(input) 

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint_h_w)
        saver.restore(sess, checkpoint_h_w)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output, np.float32)    
        
        if original_image is not None:
            ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(cnn_output, original_image)
            return ssim_cnn, psnr_cnn
        else:
            return cnn_output


def resize_depth(downscaled_image, original_image=None):

    tf.reset_default_graph()
    
    downscaled_image = np.transpose(downscaled_image, [1, 2, 0, 3])
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  

    _, output = nets_d.SRCNN_late_upscaling_D(input)

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint_d)
        saver.restore(sess, checkpoint_d)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output, np.float32)   
        print('shape', cnn_output.shape)
        cnn_output = np.int16(cnn_output * params.MAX_INTERVAL)
        
        cnn_output = np.transpose(cnn_output, [2, 0, 1, 3])  
            
        if original_image is not None: 
            ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(cnn_output, original_image)
            return ssim_cnn, psnr_cnn
        else:
            return cnn_output


def compute_performance_indices(test_images_gt, test_images):

    num_images = 0 
    ssim_cnn_sum = 0;
    psnr_cnn_sum = 0

    for index in range(len(test_images)):
        ssim_cnn, psnr_cnn = resize_depth(resize_h_w(test_images[index]), test_images_gt[index])
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn  
        num_images += test_images_gt[index].shape[0]
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))


def read_images(test_path):
    test_images_gt = utils.read_3d_images_from_directory_test(test_path, add_to_path='original_3d//%dx' % scale)
    test_images = utils.read_3d_images_from_directory_test(test_path, add_to_path='input_3d//%dx' % scale, is_test=True)

    return test_images_gt, test_images


test_path = 'D:\\Research\\super-resolution\\datasets\\test'
scale = 2
is_v2 = True


from d import networks as nets_d
from hw import networks as nets_hs
checkpoint_d = './d/data_ckpt/model.ckpt3'
checkpoint_h_w = './hw/data_ckpt/model.ckpt11'

test_images_gt, test_images = read_images(test_path)  

for i in range(len(test_images)): 
    test_images[i] = utils.process_image(test_images[i])
    test_images_gt[i] = np.int16(utils.process_image_gt(test_images_gt[i]))

compute_performance_indices(test_images_gt, test_images)



 









