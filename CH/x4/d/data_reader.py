import numpy as np
import utils
import params
from sklearn.utils import shuffle
import cv2 as cv 
import random 
import pdb


class DataReader:
    
    def __init__(self, train_path, eval_path, test_path, is_training=True, SHOW_IMAGES=False): 
    
        self.rotation_degrees = [0, 180]
        self.SHOW_IMAGES = SHOW_IMAGES        
        if is_training:
            self.train_images_in = utils.read_all_patches_from_directory(train_path,
                                                                         'input_%d_%d_%d' % (params.dim_patch_w,
                                                                                             params.dim_patch_h,
                                                                                             params.scale))
            self.train_images_gt = utils.read_all_patches_from_directory(train_path,
                                                                         'gt_%d_%d_%d' % (params.dim_patch_w,
                                                                                          params.dim_patch_h,
                                                                                          params.scale))

            self.train_images_in, self.train_images_gt = shuffle(self.train_images_in, self.train_images_gt) 
            self.num_train_images = len(self.train_images_in)
            self.dim_patch_in_rows = self.train_images_in.shape[1] 
            self.dim_patch_in_cols = self.train_images_in.shape[2]
            self.dim_patch_gt_rows = self.train_images_gt.shape[1] 
            self.dim_patch_gt_cols = self.train_images_gt.shape[2] 
            self.index_train = 0 
            print('number of train images is %d' % (self.num_train_images))   
        
        else:
            
            self.test_images_gt = utils.read_all_patches_from_directory(test_path, return_np_array=False)
            self.test_images = utils.read_all_patches_from_directory(test_path, 'input', return_np_array=False)
            self.eval_images_gt = utils.read_all_patches_from_directory(eval_path, return_np_array=False)
            self.eval_images = utils.read_all_patches_from_directory(eval_path, 'input', return_np_array=False)  
            self.num_eval_images = len(self.eval_images)
            self.num_test_images = len(self.test_images)
            print('number of eval images is %d' % (self.num_eval_images))
            print('number of test images is %d' % (self.num_test_images))
        
        
    def get_next_batch_train(self, iteration, batch_size=32):
    
        end = self.index_train + batch_size 
        if(iteration == 0): # because we use only full batch
            self.index_train = 0
            end = batch_size 
            self.train_images_in, self.train_images_gt = shuffle(self.train_images_in, self.train_images_gt) 
            
        input_images = np.zeros((batch_size, self.dim_patch_in_rows, self.dim_patch_in_cols, params.num_channels))
        output_images = np.zeros((batch_size, self.dim_patch_gt_rows, self.dim_patch_gt_cols, params.num_channels))
        
        start = self.index_train
        for idx in range(start, end): 
            image_in = self.train_images_in[idx].copy()   
            image_gt = self.train_images_gt[idx].copy()    
            # augumentation
            idx_degree = random.randint(0, len(self.rotation_degrees) - 1) 
            image_in = utils.rotate(image_in, self.rotation_degrees[idx_degree])
            image_gt = utils.rotate(image_gt, self.rotation_degrees[idx_degree])
            input_images[idx - start] = image_in.copy()
            output_images[idx - start] = image_gt.copy()
                
            if(self.SHOW_IMAGES):
                cv.imshow('input', input_images[idx - start]/255)
                cv.imshow('output', output_images[idx - start]/255)
                cv.waitKey(1000)
        
        self.index_train = end
        return input_images, output_images
 
        
