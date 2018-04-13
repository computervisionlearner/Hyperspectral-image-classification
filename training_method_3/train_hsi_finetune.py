#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:54:46 2018

@author: sw
"""

import numpy as np
import time

from matplotlib import pyplot as plt
import pandas as pd
from HSI_branch import HSI_branch
import tensorflow as tf
import os
from datetime import datetime
import logging
from read_record import Reader
import sys
import glob

BATCH_SIZE = 512
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
  
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size,))
  
  return hsi_pl, labels_pl


def do_eval(sess, step_name, HSIs_valid, labels_valid,hsi_pl, labels_pl, predict1,predict2,predict):

  valid_examples = 141324
  steps_per_epoch = valid_examples // BATCH_SIZE 
#  corrects = []
  predicts1= []
  predicts2= []
  predicts= []
  labels = []
  for step in range(steps_per_epoch):
    view_bar('valid ', step, steps_per_epoch)     
    hsi,label = sess.run([HSIs_valid, labels_valid])   
    predicts_value1,predicts_value2,predicts_value = sess.run([predict1,predict2,predict], feed_dict= {hsi_pl:hsi, labels_pl:label})
    
    predicts1.extend(predicts_value1)
    predicts2.extend(predicts_value2)
    predicts.extend(predicts_value)
    labels.extend(label)

  matrix = get_matrix(labels,predicts)
  draw_table(matrix,step_name)
  
  AA = np.mean(matrix[20,:20])
  AR = np.mean(matrix[:20,20])
  precision1 = np.mean(np.array(predicts1)==np.array(labels))
  precision2 = np.mean(np.array(predicts2)==np.array(labels))
  precision = np.mean(np.array(predicts)==np.array(labels))
  logging.info('>>fusion matrix has been saved and AA={:.3f},AR={:.3f},precision={:.3f}'.format(AA,AR,precision))
  logging.info('>>precision1={:.3f} precision2={:.3f}'.format(precision1, precision2))
  
def model_loss(logits, labels):
  with tf.variable_scope('caculate_loss') :
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits, name='corss_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.summary.scalar('cross_loss', cross_entropy_mean)
 
  return cross_entropy_mean


def model_training(variables1, variables2, loss, loss2):
  def make_optimizer(loss, variables, name='Adam'):
    """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
        and a linearly decaying rate that goes to zero over the next 100k steps
    """
    global_step = tf.Variable(0, trainable=False)
    learning_rate = 1e-4
    beta1 = 0.5
    learning_step = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name).minimize(loss, global_step=global_step, var_list=variables)
    return learning_step  
  
  variables = [var for var in tf.trainable_variables() if var not in variables1]
  
  optimizer2 = make_optimizer(loss2, variables2, name='Adam_2')
  
  with tf.control_dependencies([optimizer2]):
    optimizer = make_optimizer(loss, variables, name='Adam')

  ema = tf.train.ExponentialMovingAverage(decay=0.95)

  update_losses = ema.apply([loss2, loss])

  return tf.group(update_losses, optimizer)

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
 
def draw_table(matrix, step_name):

  idx = pd.Index(np.arange(1,22))
  cols = list(map(str,np.arange(1,22)))

  df = pd.DataFrame(matrix, index=idx, columns=cols)
  fig, axes = plt.subplots()
  plt.figure(figsize=(20,10))
  the_table=plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, 
                    colWidths = [0.03]*df.values.shape[1], loc='center',cellLoc='center')
  
  the_table.set_fontsize(15)
  the_table.scale(2,2.1)
  plt.axis('off')
  plt.savefig('result_{}.png'.format(step_name),dpi=200)
  
def run_train():
  """Train CAPTCHA for a number of steps."""

  with tf.Graph().as_default():      
    records = glob.glob('../HSI/data/*.tfrecord')  
    train_readers = []
    for record in records: 
      record_name = 'train_data' + os.path.basename(record).split('.')[0]        
      train_reader = Reader(record, name = record_name, batch_size=50)
      train_readers.append(train_reader)  
    
    valid_reader = Reader('tiny_val.tfrecord', name='valid_data', batch_size=BATCH_SIZE)    
    train_imgs_and_labels = [train_reader_.feed(train_data = True) for train_reader_ in train_readers]

    HSIs_valid, labels_valid = valid_reader.feed(train_data = False)
    hsi_pl, labels_pl = placeholder_inputs(BATCH_SIZE)
    
    HSI = HSI_branch()
    logits1, variables1, logits2, variables2, logits, outputs, variables = HSI(hsi_pl,keep_prob=0.5) #(500, 2432)
    
    predicts = model_predict(logits)
    predicts1 = model_predict(logits1)
    predicts2 = model_predict(logits2)
    
    
    loss2 = model_loss(logits2, labels_pl)
    loss = model_loss(logits, labels_pl) + loss2
    train_op = model_training(variables1, variables2, loss, loss2)
    
    summary = tf.summary.merge_all()    
#    saver1 = tf.train.Saver(var_list=variables1)
#    saver2 = tf.train.Saver(var_list=variables2)
    saver = tf.train.Saver(max_to_keep=50)
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    sess.run(init_op)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
#    saver1.restore(sess, 'ckpt/model.ckpt-25000')
#    saver2.restore(sess, 'ckpt/model.ckpt-32000')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:      
      max_step = 80000      
      for step in range(50000,max_step):
        start_time = time.time()        
        images_and_labels = sess.run(train_imgs_and_labels)
        HSIs, labels = [], []
        for hsi, label in images_and_labels:
          HSIs.extend(hsi)
          labels.extend(label)          
       
        shuffle = np.random.permutation(range(len(labels)))
        HSIs = np.array(HSIs)
        labels = np.array(labels)
        
        HSIs = HSIs[shuffle][:BATCH_SIZE]
        labels = labels[shuffle][:BATCH_SIZE]
        
        _, loss_value, summary_str, predicts_value = sess.run([train_op, loss, summary, predicts],\
                                              feed_dict={hsi_pl:HSIs, labels_pl:labels}
                                              )
        
        train_precision = np.mean(predicts_value==labels)
        
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        duration = time.time() - start_time
        
        count = (step % 500) or 500
        message = ('>>Step %d run_train: loss = %.4f precision = %.3f (%.3f sec)'
                % (step, loss_value, train_precision, duration))
        view_bar(message, count, 500)
          #-------------------------------
        if step % 500 == 0:
          logging.info('>>%s Saving in %s' % (datetime.now(), checkpoint_dir))
          saver.save(sess, checkpoint_file, global_step=step)
          
          logging.info('Valid Data Eval:')
          do_eval(sess,step,
                  HSIs_valid, labels_valid, hsi_pl, labels_pl, predicts1,predicts2,predicts,
                  )


    except KeyboardInterrupt:
        print('INTERRUPTED')
        coord.request_stop()

    finally:
        saver.save(sess, checkpoint_file, global_step=step)
        print('Model saved in file :%s'%checkpoint_dir)
        coord.request_stop()
        coord.join(threads)

    sess.close()

if __name__ == '__main__':
  run_train()