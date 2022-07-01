import sys
from turtle import color
sys.executable
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

import h5py
import provider
import tf_utilV2
from modelV2 import *
from plyfile import PlyData, PlyElement
import open3d as o3d
print("success")

BATCH_SIZE = 1
BATCH_SIZE_EVAL = 1
NUM_POINT = 4096*2
MAX_EPOCH = 50
BASE_LEARNING_RATE = 0.001
GPU_INDEX = 0
MOMENTUM = 0.9
OPTIMIZER = 'adam'
DECAY_STEP = 300000
DECAY_RATE = 0.5

LOG_DIR = './log_1200_burr'
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

# Load ALL data
f = h5py.File('/home/peter/Documents/python_ws/PointNet_Custom_Object_Detection/test_input/test_data.h5')
print(f)

#Choose a frame to test, (0,60)
frame_to_test = 0


test_data = np.zeros((NUM_POINT, 6))
test_label = np.ones((1,NUM_POINT))

xmax = 3.0
xmin = -3.0

data = f['data']
label = f['label']
test_data[:,0:3] = (data[frame_to_test][:, 0:3]- xmin) / (xmax  - xmin )
test_data[:,3:6] = data[frame_to_test][:, 3:6]
test_label[:,:] = label[frame_to_test][:]

    
print(test_data.shape)
print(test_label.shape)

features = ["x","y","z","r","g","b"]
for i in range(6): 
    print(features[i] + "_range :", np.min(test_data[:, i]), np.max(test_data[:, i]))

test_data_min = []
test_data_max = []
for i in range(6):
    test_data_min.append(np.min(test_data[:,i]))
    test_data_max.append(np.max(test_data[:,i]))
    
print(test_data_min)
print(test_data_max)

features = ["x","y","z","r","g","b"]
for i in range(6): 
    print(features[i] + "_range :", np.min(test_data[:, i]), np.max(test_data[:, i]))

def numpy2o3d(np_points,labels):
    pcd = o3d.geometry.PointCloud()
    print(f"np_points.shape: {np_points.shape}")
    pcd.points = o3d.utility.Vector3dVector(np_points)
    colors_array = np.zeros((np_points.shape[0], 3))
    colors_array[np.where(labels == 1), :] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors_array)
    return pcd

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

def evaluate():
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
            learning_rate = get_learning_rate(batch)

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
        merged = tf.compat.v1.summary.merge_all()

        
        ops = {'pointclouds_pl': pointclouds_pl,
       'labels_pl': labels_pl,
       'is_training_pl': is_training_pl,
       'pred': pred,
       'loss': loss,
       'train_op': train_op,
       'merged': merged,
       'step': batch}
        MODEL_PATH = f"{LOG_DIR}/model.ckpt"
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")
        
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        eval_one_epoch(sess, ops, test_writer)


        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    
    #current_data = np.zeros((1,4096, 6))
    current_data  = test_data[0:NUM_POINT,:]
    visual_data = np.copy(current_data)[:,0:3]
    current_label = test_label
    
    current_data = current_data.reshape(1,NUM_POINT, 6)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE_EVAL 
    
    fout = open(f'{LOG_DIR}/'+str(frame_to_test)+'_pred.obj', 'w')
    fout_gt = open(f'{LOG_DIR}/'+str(frame_to_test)+'_gt.obj', 'w')    
    
    t0 = time.time()
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE_EVAL
        end_idx = (batch_idx+1) * BATCH_SIZE_EVAL

        feed_dict = {ops['pointclouds_pl']: current_data[:, :],
                     ops['labels_pl']: current_label[:],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        
        pred_label = np.argmax(pred_val, 2) # BxN
        
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)

        #Visualize the predicted point cloud
        result_pcd = numpy2o3d(visual_data,pred_val)
        o3d.visualization.draw_geometries([result_pcd])
        original_pcd = numpy2o3d(visual_data,current_label)
        o3d.visualization.draw_geometries([original_pcd])

        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE_EVAL*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE_EVAL)
        class_color = [[0,255,0],[0,0,255]]
        print(start_idx, end_idx)
        
        for i in range(start_idx, end_idx):
            print(pred_label.shape)
            pred = pred_label[i-start_idx, :]
            
            pts = current_data[i-start_idx, :, :]
            l = current_label[i-start_idx,:]
            
            
            for j in range(NUM_POINT):
                l = int(current_label[i, j])
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
 
                color = class_color[pred_val[i-start_idx, j]]
                color_gt = class_color[l]
  
                fout.write('v %f %f %f %d %d %d\n' % (pts[j,0], pts[j,1], pts[j,2], color[0], color[1], color[2]))
                fout_gt.write('v %f %f %f %d %d %d\n' % (pts[j,0], pts[j,1], pts[j,2], color_gt[0], color_gt[1], color_gt[2]))
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    print(f"Time taken: {time.time() - t0}")
         

if __name__ == "__main__":
    evaluate()
    LOG_FOUT.close()