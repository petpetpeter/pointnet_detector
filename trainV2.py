import sys
sys.executable

import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import glob

import os
import sys
 
import provider
import tf_utilV2
from modelV2 import *
print("success")

BATCH_SIZE = 16
BATCH_SIZE_EVAL = 16
NUM_POINT = 4096*2
MAX_EPOCH = 1000
BASE_LEARNING_RATE = 0.001
GPU_INDEX = 0
MOMENTUM = 0.9
OPTIMIZER = 'adam'
DECAY_STEP = 300000
DECAY_RATE = 0.5

LOG_DIR = 'log'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
#os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
# LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = NUM_POINT
NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# #Normalize data
# xmax = 3.0
# xmin = -3.0

DATASET_DIR = '/home/hayashi/Documents/pythonWS/pointnet_detector/train_input/augmented_data/'
f = h5py.File(DATASET_DIR+ 'd0.h5','r')
NUM_FRAMES = f['data'].shape[0]

total_data = np.zeros((6*NUM_FRAMES,NUM_POINT, 6))
total_label = np.zeros((6*NUM_FRAMES,NUM_POINT))

for i in range (0,6):
    f = h5py.File(DATASET_DIR+ 'd' +str(i)+'.h5','r')
    data = f['data']
    label = f['label']
    total_data[i*len(data):(i+1)*len(data),:,0:3] = data[:, :, 0:3]
    total_data[i*len(data):(i+1)*len(data),:,3:6] = data[:, :, 3:6]
    total_label[i*len(data):(i+1)*len(data),:] = label[:, :]
    
print(total_data.shape)
print(total_label.shape)

features = ["x","y","z","r","g","b"]
for i in range(6): 
    print(features[i] + "_range :", np.min(total_data[:, :, i]), np.max(total_data[:, :, i]))


X = total_data
y = total_label

X.shape, y.shape

features = ["x","y","z","r","g","b"]
for i in range(6): 
    print(features[i] + "_range :", np.min(total_data[:, :, i]), np.max(total_data[:, :, i]))

from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.1, random_state=42)

print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.compat.v1.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(input=pred, axis=2), tf.cast(labels_pl, dtype=tf.int64))
            accuracy = tf.reduce_sum(input_tensor=tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver()
            
        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.compat.v1.Session(config=config)

        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE_EVAL
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE_EVAL
        end_idx = (batch_idx+1) * BATCH_SIZE_EVAL

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE_EVAL*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE_EVAL)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = int(current_label[i, j])
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         

if __name__ == "__main__":
    train()
    LOG_FOUT.close()