#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 18:50:05 2018

@author: sw
"""

import numpy as np
import time

from matplotlib import pyplot as plt
import pandas as pd
from HSI_test_branch import HSI_branch
import tensorflow as tf
import os
from datetime import datetime
import logging
from read_record import Reader
import sys

BATCH_SIZE = 2000
IMAGE_HEIGHT = 17
IMAGE_WIDTH = 17

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

checkpoint_dir = 'ckpt'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
train_dir='summary'

def initLogging(logFilename='record.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level= logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
  
initLogging()

def placeholder_inputs(batch_size):
  hsi_pl = tf.placeholder(tf.float32,
                        shape=(batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,48))
  labels_pl = tf.placeholder(tf.int32,shape=(BATCH_SIZE))
  return hsi_pl, labels_pl


def do_eval(sess, HSIs_valid, labels_valid, logits1):

  valid_examples = 1413248

  steps_per_epoch = valid_examples // BATCH_SIZE +1
#  corrects = []
  predicts= []
  labels = []
  indexes = []
  output1 = tf.nn.softmax(logits1)
  for step in range(steps_per_epoch):
    view_bar('valid ', step, steps_per_epoch)     
    output_value1,label = sess.run([output1, labels_valid])   
    predicts_value1 = np.argmax(output_value1,axis=-1)
    predicts.extend(predicts_value1)
    labels.extend(label)
    temp = np.max(output_value1,axis=-1)>0.95
    indexes.extend(temp)
  

  matrix = get_matrix(labels[:valid_examples],predicts[:valid_examples])
  sorted_matrix = get_matrix(np.array(labels)[indexes],np.array(predicts)[indexes])
  draw_table(matrix,'normal')
  draw_table(sorted_matrix,'sorted')
  AA = np.mean(matrix[20,:20])
  AR = np.mean(matrix[:20,20])
  precision = np.mean(np.array(predicts[:valid_examples])==np.array(labels[:valid_examples]))
  kappa = compute_Kappa(matrix[:20,:20])
  sorted_precision = np.mean(np.array(predicts)[indexes]==np.array(labels)[indexes])
#  np.save('index',indexes)
  logging.info('>>fusion matrix has been saved and AA={:.3f},AR={:.3f},precision={:.3f},sorted_precision={:.3f}'.format(AA,AR,precision,sorted_precision))
  logging.info('>> kappa = {}'.format(kappa))


def compute_Kappa(confusion_matrix):
  N = np.sum(confusion_matrix)
  N_observed = np.trace(confusion_matrix)
  Po = 1.0 * N_observed / N
  h_sum = np.sum(confusion_matrix, axis=0)
  v_sum = np.sum(confusion_matrix, axis=1)
  Pe = np.sum(np.multiply(1.0 * h_sum / N, 1.0 * v_sum / N))
  kappa = (Po - Pe) / (1.0 - Pe)
  return kappa

def model_predict(logits):
  with tf.variable_scope('predict') :    
    predicts = tf.argmax(logits, axis=-1, name='predict')
    return predicts

def get_matrix(labels, predicts):
  '''
  列表示groundtruth，行表示预测，此函数返回值是混淆矩阵
  '''
  matrix = np.zeros((21,21))
  for i in range(len(labels)):
    matrix[labels[i],predicts[i]] += 1   
    
  matrix[:20,20] = np.round(np.diag(matrix[:20,:20])/np.sum(matrix[:20,:20],axis=1),2)
  matrix[20,:20] = np.round(np.diag(matrix[:20,:20])/np.sum(matrix[:20,:20],axis=0),2)
  
  return matrix
 
def draw_table(matrix,name):

  idx = pd.Index(np.arange(1,22))
  cols = list(map(str,np.arange(1,22)))

  df = pd.DataFrame(matrix, index=idx, columns=cols)
  plt.figure(figsize=(20,10))
  the_table=plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, 
                    colWidths = [0.03]*df.values.shape[1], loc='center',cellLoc='center')
  
  the_table.set_fontsize(15)
  the_table.scale(2,2.1)
  plt.axis('off')
  plt.savefig('test_{}.png'.format(name),dpi=200)
  
def run_test():
  """Train CAPTCHA for a number of steps."""

  with tf.Graph().as_default():      
    valid_reader = Reader('val.tfrecord', name='valid_data', min_queue_examples=50000, batch_size=BATCH_SIZE, num_threads=13)    
    HSIs_valid, labels_valid = valid_reader.feed(train_data = False)
#    hsi_pl, labels_pl = placeholder_inputs(BATCH_SIZE)
    
    HSI = HSI_branch()
    logits1, outputs1, variables1 = HSI(HSIs_valid, keep_prob=1) #(500, 2432)
    
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:      

      start_time = time.time()        

      do_eval(sess, HSIs_valid, labels_valid, logits1)
      duration = time.time() - start_time
      
      logging.info('>>Take time %.3f(sec)' % duration)
    except KeyboardInterrupt:
        print('INTERRUPTED')
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)

    sess.close()

if __name__ == '__main__':
  run_test()
