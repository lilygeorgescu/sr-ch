import tensorflow as tf
import numpy as np
import cv2 as cv 
import params
import utils
import pdb
import re
import os 

params.show_params()

config = tf.ConfigProto(
        device_count = {'GPU': 1}
    ) 
    
def upscale(downscaled_image, checkpoint):

    scale_factor = params.scale   
     
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    _, output = params.network_architecture(input) 
     

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint)
        saver.restore(sess, checkpoint)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output)   
        cnn_output = np.round(cnn_output) 
        cnn_output[cnn_output > 255] = 255 
        
        return cnn_output
        
def predict(downscaled_image, original_image, checkpoint):
    scale_factor = params.scale   
     
            
    standard_resize = utils.resize_height_width_3d_image_standard(downscaled_image, int(original_image.shape[1]), int(original_image.shape[2]), interpolation_method = params.interpolation_method)
    num_iters = 1 #int(np.log2(scale))
    
    for iter in range(num_iters):
         downscaled_image = upscale(downscaled_image, checkpoint)
         tf.reset_default_graph()
    print(downscaled_image.shape, original_image.shape)
    name = image_names.pop(0)
    utils.write_3d_images(name, downscaled_image, 'cnn')     
    utils.write_3d_images(name, standard_resize, 'lanczos')     
    ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(downscaled_image, original_image)
    ssim_standard, psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image)

    return ssim_cnn, psnr_cnn, ssim_standard, psnr_standard 
        
def read_images(test_path):

    test_images_gt = utils.read_all_directory_images_from_directory_test(test_path)
    test_images = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='input_%d' % scale) 
    return test_images_gt, test_images
    
def compute_performance_indeces(test_path, test_images_gt, test_images, checkpoint, write_to_summary=True):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images)):
        # pdb.set_trace()
        ssim_cnn, psnr_cnn, ssim_standard, psnr_standard = predict(test_images[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images[index].shape[0]
     
    print('standard {} --- psnr = {} ssim = {}'.format(test_path, psnr_standard_sum/num_images, ssim_standard_sum/num_images)) 
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))
    
    if test_path.find('test') != -1 and write_to_summary == True:
    
        tf.summary.scalar('psnr_standard', psnr_standard_sum/num_images) 
        tf.summary.scalar('psnr_cnn', psnr_cnn_sum/num_images)  
        tf.summary.scalar('ssim_standard', ssim_standard_sum/num_images)  
        tf.summary.scalar('ssim_cnn', ssim_cnn_sum/num_images)  
        merged = tf.summary.merge_all() 
        writer = tf.summary.FileWriter('test.log')  
        epoch = re.findall(r'\d+', checkpoint)
        epoch = int(epoch[0]) 
        with tf.Session(config=config) as sess:
            merged_ = sess.run(merged)
            writer.add_summary(merged_, epoch)
        
test_path = './data/test' 
eval_path = './data/train'
scale = 4

image_names = ['00001_0007', '00001_0009', '00001_0010', '00001_0011']
test_images_gt, test_images = read_images(test_path)  
# checkpoint = tf.train.latest_checkpoint(params.folder_data)  
checkpoint = os.path.join(params.folder_data, 'model.ckpt%d' % 43)
compute_performance_indeces(test_path, test_images_gt, test_images, checkpoint, write_to_summary=False) 
exit()

for i in range(40, 45):
    checkpoint = os.path.join(params.folder_data, 'model.ckpt%d' % i)
    
    compute_performance_indeces(test_path, test_images_gt, test_images, checkpoint)
    # compute_performance_indeces(eval_path, eval_images_gt, eval_images, checkpoint)  
 