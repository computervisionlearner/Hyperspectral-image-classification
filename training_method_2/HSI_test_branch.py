#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:26:11 2018

@author: sw
"""

import tensorflow as tf


def _leaky_relu(inputs, slope, name):
  with tf.variable_scope(name):
    return tf.maximum(slope*inputs, inputs)
  
def _weight_variable(name, shape, mean=0):
  """weight_variable generates a weight variable of a given shape."""
  initializer = tf.truncated_normal_initializer(mean=mean,stddev=0.1)
  var = tf.get_variable(name,shape,initializer=initializer, dtype=tf.float32)
  return var


def _bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1)
  var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var  
def _instance_norm(name, inputs):
  """ Instance Normalization
  """
  #equals to tf.nn.batch_normalization() just when batch=1
  with tf.variable_scope(name):
    mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)  # (1, 1, 1, 64) nomornize in axis=0&1
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon) #rsqrt(x)=1./sqrt(x)
    normalized = (inputs-mean)*inv
    return normalized 
  
class HSI_branch:
  
  def __init__(self, height = 17, width =17, channel = 48, classes = 20):
      
    self.width = width
    self.height = height
    self.channel = channel
    self.rate = 0.2
    self.r = 7
    self.classes = classes

    
  def __call__(self, images, keep_prob):
    with tf.variable_scope('branch_1'):
      
      flatten1 = self.conv2d(images)#shape=(500, 1024)
#      features1 = tf.nn.dropout(flatten1, keep_prob)
#      logits1 = tf.layers.dense(features1, 20, name='logits1')
#      variables1 = [var for var in tf.trainable_variables() \
#                    if var.name.startswith('branch_1')]
      
      
    with tf.variable_scope('branch_2'):
      flatten2 = self.conv1d(images)#shape=(500, 608)
#      features2 = tf.nn.dropout(flatten2, keep_prob)
#      logits2 = tf.layers.dense(features2, 20, name='logits2')
#      variables2 = [var for var in tf.trainable_variables() \
#                    if var.name.startswith('branch_2')]  
      
    outputs = tf.concat([flatten1,flatten2],axis=-1,name='concatenate')
    features = tf.nn.dropout(outputs, keep_prob)
    logits = tf.layers.dense(features, 20, name='logits')
    variables = [var for var in tf.trainable_variables()]
    return logits, outputs, variables
  
  
  def conv2d(self, images):
    
    rate = self.rate
    images = images[:,1:,1:,:]  #(batch,16,16,48)
    layers1 = tf.layers.conv2d(images, filters = 64, kernel_size = 3, strides = 1,
                     kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                     padding = 'same', name = 'layers1')
    
    layers1 = _instance_norm('BN1',layers1)
    layers1 = _leaky_relu(layers1, rate, 'activations1')

    
    pool1 = tf.nn.max_pool(layers1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding='SAME', name = 'pool1')#(batch,8,8,64)
    
    layers2 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 3, strides = 1,
                    kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                    padding = 'same',name = 'layers2')
    layers2 = _instance_norm('BN2',layers2)
    layers2 = _leaky_relu(layers2, rate, 'activations2')
    pool2 = tf.nn.max_pool(layers2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding='SAME', name = 'pool2')#(batch,4,4,128)
    layers3_1 = tf.layers.conv2d(pool2, filters = 256, kernel_size = 3, strides = 1,
                    kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                    padding = 'same',name = 'layers3_1')
    layers3_1 = _instance_norm('BN3',layers3_1)

    layers3_1 = _leaky_relu(layers3_1, rate, 'activations3_2')

    flatten1 = tf.contrib.layers.flatten(layers3_1)  #(batch,2048)
    assert flatten1.shape.as_list()[-1]==4096
    return flatten1
  
  
  def conv1d(self, images):
    
    r = self.r
    rate = self.rate
    
#    inputs = images[:,r-6:r+5+1,r-6:r+5+1,:]
    inputs = images[:,r-6:r+5+1,r-6:r+5+1,:]
    
    out1 = tf.layers.conv2d(inputs, filters = 64, kernel_size = 3, strides = 1,
                     kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                     padding = 'same', name = 'out1')  #(batch,9,9,64)
     
    out1 = _instance_norm('BN5', out1)
    
    out1 = _leaky_relu(out1, rate, 'activations4')   
    
    #-------------------------------------------------------------------------------------
    
    pool1 = tf.nn.max_pool(out1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding='SAME', name = 'pool3')   #(batch,5,5,64)
    
    
    out2 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 3, strides = 1,
                    kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                    padding = 'same',name = 'out2')
    
    out2 = _instance_norm('BN6', out2)
    out2 = _leaky_relu(out2, rate, 'activations5')  
 
    pool2 = tf.nn.avg_pool(out2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                padding='SAME', name = 'pool4') #(batch,3,3,128)
    
    layers3_1 = tf.layers.conv2d(pool2, filters = 256, kernel_size = 3, strides = 1,
                    kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                    padding = 'same',name = 'out3')
    layers3_1 = _instance_norm('BN7',layers3_1)

    layers3_1 = _leaky_relu(layers3_1, rate, 'activations6')    
    
    flatten2 = tf.contrib.layers.flatten(pool2)
    
    return flatten2
    
if __name__ == '__main__':
  
  images = tf.ones((100,11,11,50))
  hsi = HSI_branch()
  logits = hsi(images)