#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 23:02:02 2018

@author: sw
"""
import tensorflow as tf
import glob
import os
import numpy as np
import time
import gc
import sys
class Reader():
  def __init__(self, tfrecords_file, height=17, width=17,
    min_queue_examples=5000, batch_size=500, num_threads=3, name = ''):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.tfrecords_file = tfrecords_file
    self.height = height
    self.width =width
    self.min_queue_examples = min_queue_examples
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.reader = tf.TFRecordReader()
    self.name = name
  def feed(self, train_data = True):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer([self.tfrecords_file])
      _, serialized_example = self.reader.read(filename_queue)
  
      features = tf.parse_single_example(
        serialized_example,
        features={
        'label_raw': tf.FixedLenFeature([], tf.string),
        'HSI_raw': tf.FixedLenFeature([], tf.string)
         
            })
  
      HSI = tf.decode_raw(features['HSI_raw'], tf.float32)     
      HSI.set_shape([self.height * self.width * 48])
      HSI = tf.cast(HSI, tf.float32) 
     
      label = tf.decode_raw(features['label_raw'], tf.int8)
      label = tf.reshape(tf.cast(label, tf.int32),shape=(1,))
      reshape_HSI = tf.reshape(HSI, [self.height, self.width, 48])   
  
#--------------------------------------------------------------------------------      

      
      if train_data:
        reshape_HSI = tf.image.random_flip_left_right(reshape_HSI)
        reshape_HSI = tf.image.random_flip_up_down(reshape_HSI)
#        reshape_HSI = tf.image.rot90(reshape_HSI, k = np.random.randint(0,4))
        
        HSIs, labels = tf.train.shuffle_batch([reshape_HSI, label], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size, min_after_dequeue=self.min_queue_examples)
  
      else:
        
        HSIs, labels = tf.train.batch([reshape_HSI, label],
                                               batch_size = self.batch_size,
                                               num_threads=3, capacity=2000 + 3*self.batch_size)
        
#      tf.summary.image('record_inputs', images)
      labels = tf.squeeze(labels)
    return HSIs, labels

def read_train():
#  train_reader = Reader('valid.tfrecord',name='train_data')
#  images_op, labels_op = train_reader.feed(train_data=False)
  
  records = glob.glob('data/*.tfrecord')  
  train_readers = []
  for record in records: 
    record_name = 'train_data' + os.path.basename(record).split('.')[0]        
    train_reader = Reader(record, name = record_name, batch_size=50)
    train_readers.append(train_reader)    
    
  train_imgs_and_labels = [train_reader_.feed(train_data = False) for train_reader_ in train_readers]
  
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  train_images = []
  train_labels = []
  
  for i in range(3000):
    start_time = time.time()        
    images_and_labels = sess.run(train_imgs_and_labels)
#    images,labels=sess.run([images_op, labels_op])
    HSIs = []
    VISs = []
    LIDARs = []
    labels = []
    for hsi, vis, lidar, label in images_and_labels:
      HSIs.extend(hsi)
      VISs.extend(vis)
      LIDARs.extend(lidar)
      labels.extend(label)
       
    HSIs = np.array(HSIs)
    VISs = np.array(VISs)
    LIDARs = np.array(LIDARs)
    labels = np.array(labels)
         
  coord.request_stop()
  coord.join(threads)
  sess.close() 
  
def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()
    
def read_valid():
  train_reader = Reader('data/1.tfrecord',name='valid_data', batch_size=500)
  images_op, labels_op = train_reader.feed(train_data=False)
  
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  valid_images = []
  valid_labels = []
  valid_examples = 1413248
  steps_per_epoch = valid_examples // 500  
#  indexes = np.split(np.load('index.npy'),steps_per_epoch)
  
  for i in range(steps_per_epoch):
#    view_bar('valid ', i, steps_per_epoch)
    start_time = time.time()        
    images,labels=sess.run([images_op, labels_op])  
    view_bar('valid ', i, steps_per_epoch)
#    temp = indexes[i]
#    temp_images = images[temp]
#    temp_labels = labels[temp]
#    
#    temp1 = ((temp_labels==1) | (temp_labels==2) | (temp_labels==3) | (temp_labels==6) | (temp_labels==8) | (temp_labels==18))
#    valid_images.extend(temp_images[temp1])
#    valid_labels.extend(temp_labels[temp1])
#    print('processed label = {} {}/{}'.format(labels[0],i,len(valid_labels)))
    
  duration = time.time() - start_time
#  np.save('valid_images',valid_images)
#  np.save('valid_labels',valid_labels)
  print('cost time = {:.3f}'.format(duration))
  del valid_images
  gc.collect()
  time.sleep(1)
      
  coord.request_stop()
  coord.join(threads)
  sess.close()  
  
if __name__ == '__main__':
  read_valid()
