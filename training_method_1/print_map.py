#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:16:59 2018

@author: sw
"""

import numpy as np
#import cv2

import tensorflow as tf
import os
import tifffile as tiff
from HSI_branch import HSI_branch
r = 8

expended_IMAGE_WIDTH = 4768 + 2*r
expended_IMAGE_HEIGHT = 1202 + 2*r
checkpoint_dir = 'ckpt'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')

import sys
def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

def placeholder_inputs(batch_size):
  hsi_pl = tf.placeholder(tf.float32,
                        shape=(batch_size,2*r+1,2*r+1,48))
  
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size,))
  
  return hsi_pl, labels_pl

def get_batch_data(col, batch_size):
  img_hsi = []
  
  for row in range(r, expended_IMAGE_HEIGHT-r, 1):
    temp_h = HSI[row-r:row+r+1,col-r:col+r+1,:]

    
    img_hsi.append(temp_h)

  
  img_hsi = np.asarray(img_hsi)
  
  return img_hsi
  
if __name__ == '__main__':  
  HSI = np.load('expended_HSI.npy')
  gt = tiff.imread('GT.tif')
  idx,idy = np.where(gt==0)
  print(HSI.shape)
  with tf.Graph().as_default():
    hsi_pl, _ = placeholder_inputs(batch_size = 1202)
    HSI_b = HSI_branch()
    logits, outputs, variables = HSI_b(hsi_pl,keep_prob=1)#(500, 2432)    
    outputs1 = tf.nn.softmax(logits)
    
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, 'ckpt/model.ckpt-15000')
    predict_gt = np.ones((expended_IMAGE_HEIGHT-2*r, expended_IMAGE_WIDTH-2*r))
    
    for col in range(r, expended_IMAGE_WIDTH-r, 1):
      img_hsi = get_batch_data(col, batch_size = 1202)
      outputs_value = sess.run(outputs1, feed_dict = {hsi_pl:img_hsi})
      predicts = np.argmax(outputs_value, axis=-1)
      temp = np.max(outputs_value,axis=-1)<=0.95
      predicts[temp]=-1
      predict_gt[:,col-r] = predicts + 1
      view_bar('processing', col-r, expended_IMAGE_WIDTH-2*r)
    predict_gt[idx,idy] = 0
    np.save('predict_map', predict_gt)
    tiff.imsave('sorted_predict_map1.tif',predict_gt)