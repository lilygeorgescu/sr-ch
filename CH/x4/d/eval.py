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
        device_count={'GPU': 1}
    )


def run_network(downscaled_image, checkpoint):
    scale_factor = params.scale  
    
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    _, output = params.network_architecture(input, is_training=False) 
     

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint)
        saver.restore(sess, checkpoint)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])

        cnn_output = np.array(cnn_output, np.float32)
        cnn_output = np.round(cnn_output * params.MAX_INTERVAL)
        return cnn_output    


def predict(downscaled_image, original_image, checkpoint): 

    cnn_output = run_network(downscaled_image, checkpoint)
    print(cnn_output.shape, original_image.shape)
    cnn_output = cnn_output[:, :, :original_image.shape[2], :]
    original_image = original_image[:, :, :cnn_output.shape[2], :]
    ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(cnn_output, original_image)
    print(ssim_cnn, psnr_cnn)
    return ssim_cnn, psnr_cnn


def read_images(test_path):
    add_to_path_gt = 'original_d'
    add_to_path_in = 'input_d_x%d' % scale

    test_images, lists_idx = utils.read_all_directory_images_from_directory_test_depth(test_path,
                                                                                       add_to_path=add_to_path_in)
    test_images_gt = utils.read_all_directory_images_from_directory_test_depth(test_path, add_to_path=add_to_path_gt,
                                                                               list_idx=lists_idx)

    return test_images_gt, test_images


def compute_performance_indeces(test_path, test_images_gt, test_images, checkpoint, add_to_summary=True):

    num_images = 0 
    ssim_cnn_sum = 0
    psnr_cnn_sum = 0
 
    for index in range(len(test_images_gt)): 
            
        ssim_cnn, psnr_cnn = predict(test_images[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn
        num_images += test_images_gt[index].shape[0]

    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))
    
    if test_path.find('test') != -1 and add_to_summary:
        tf.summary.scalar('psnr_cnn', psnr_cnn_sum/num_images)
        tf.summary.scalar('ssim_cnn', ssim_cnn_sum/num_images)  
        merged = tf.summary.merge_all()
         
        writer = tf.summary.FileWriter('test.log') 
            
        epoch = re.findall(r'\d+', checkpoint)
        epoch = int(epoch[0])
        
        with tf.Session(config=config) as sess:
            merged_ = sess.run(merged)
            writer.add_summary(merged_, epoch)


scale = 2
test_path = 'D:\\Research\\super-resolution\\datasets\\test'
test_images_gt, test_images = read_images(test_path)

for i in range(len(test_images)):
    test_images[i] = utils.process_image(test_images[i])
    test_images_gt[i] = utils.process_image_gt(test_images_gt[i])



# checkpoint = os.path.join(params.folder_data, 'model.ckpt%d' % 6) 

checkpoint = tf.train.latest_checkpoint(params.folder_data)
compute_performance_indeces(test_path, test_images_gt, test_images, checkpoint, add_to_summary=False)
# exit()
# compute_performance_indeces(eval_path, eval_images, eval_images_gt, checkpoint)  


# for i in range(9, 10):
#     checkpoint = os.path.join(params.folder_data, 'model.ckpt%d' % i)
#     compute_performance_indeces(test_path, test_images_gt, test_images, checkpoint)
 