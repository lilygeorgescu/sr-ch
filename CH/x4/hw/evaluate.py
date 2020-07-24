from utils import *
import params
import tensorflow as tf
import pdb
import cv2 as cv 

def trim_image(image):
    image[image > 255] = 255
    return image
    

def upscale(downscaled_image, checkpoint):
 
    # network for original image
    config = tf.ConfigProto(
            device_count = {'GPU': 1}
        ) 
    
    or_graph = tf.Graph()
    sess_or = tf.Session(graph=or_graph, config=config)
    with or_graph.as_default():
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        _, output_or = params.network_architecture(input_or) 
        saver = tf.train.Saver() 
        saver.restore(sess_or, checkpoint)
        
    tr_graph = tf.Graph()
    sess_tr = tf.Session(graph=tr_graph, config=config)
    with tr_graph.as_default():
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        _, output_tr = params.network_architecture(input_tr, reuse=False)      
        saver = tf.train.Saver() 
        saver.restore(sess_tr, checkpoint)
    
    num_images = downscaled_image.shape[0]
    cnn_output = []
    for image in downscaled_image:
        out_images = [] 
        
        # original 0 
        res = trim_image(sess_or.run(output_or, {input_or: [image]})[0]) 
        out_images.append(res)
        
         
        # flip 0 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(image)]})[0])
        out_images.append(reverse_flip_image(res))
        
        # original 90
        rot90_image = rotate_image_90(image)
        res = trim_image(sess_tr.run(output_tr, {input_tr: [rot90_image]})[0])
        out_images.append(reverse_rotate_image_90(res)) 
        
        # flip 90 
        res = trim_image(sess_tr.run(output_tr, {input_tr: [flip_image(rot90_image)]})[0])
        out_images.append(reverse_rotate_image_90(reverse_flip_image(res)))   

        # original 180
        rot180_image = rotate_image_180(image)
        res = trim_image(sess_or.run(output_or, {input_or: [rot180_image]})[0])
        out_images.append(reverse_rotate_image_180(res)) 
        
        # flip 180 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(rot180_image)]})[0])
        out_images.append(reverse_rotate_image_180(reverse_flip_image(res)))          
        
        # original 270
        rot270_image = rotate_image_270(image)
        res = trim_image(sess_tr.run(output_tr, {input_tr: [rot270_image]})[0])
        out_images.append(reverse_rotate_image_270(res)) 
        
        # flip 270 
        res = trim_image(sess_tr.run(output_tr, {input_tr: [flip_image(rot270_image)]})[0])
        out_images.append(reverse_rotate_image_270(reverse_flip_image(res))) 
        # pdb.set_trace()
        if use_mean:
            cnn_output.append(np.round(np.mean(np.array(out_images), axis=0)))
        else:
            cnn_output.append(np.round(np.median(np.array(out_images), axis=0)))
        
    cnn_output = np.array(cnn_output)
    write_3d_images(image_names.pop(0), cnn_output, 'cnn_best') 
    return cnn_output

def predict(downscaled_image, original_image, checkpoint):
    scale_factor = params.scale    
    num_iters = 1 # int(np.log2(scale))
    
    for iter in range(num_iters):
         downscaled_image = upscale(downscaled_image, checkpoint)
         tf.reset_default_graph()
         
    ssim_cnn, psnr_cnn = compute_ssim_psnr_batch(downscaled_image, original_image) 

    return ssim_cnn, psnr_cnn 
    
def compute_performance_indeces(test_images_gt, test_images, checkpoint):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images)):
        # pdb.set_trace()
        ssim_cnn, psnr_cnn = predict(test_images[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn  
        num_images += test_images[index].shape[0]
      
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))

def read_images(test_path):

    test_images_gt = read_all_directory_images_from_directory_test(test_path)
    test_images = read_all_directory_images_from_directory_test(test_path, add_to_path='input_%d' % scale)
    
    return test_images_gt, test_images
    
# checkpoint = tf.train.latest_checkpoint(params.folder_data)  
image_names = ['00001_0007', '00001_0009', '00001_0010', '00001_0011']  
scale = 4
checkpoint = './data_ckpt/model.ckpt35'
use_mean = False

test_path = './data/test_new_images'  
test_images_gt, test_images = read_images(test_path)    
 
compute_performance_indeces(test_images_gt, test_images, checkpoint) 