import os
import sys
from unicodedata import name
from modelV2 import *
import numpy as np
import open3d as o3d
import copy
import tensorflow as tf
import socket
 
from utils.label_pcd import PointCloudData,read_pcd_file

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

class PointNetDetector:
    def __init__(self,model_path):
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
            self.config = tf.compat.v1.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.config.allow_soft_placement = True
            self.config.log_device_placement = True
            self.sess = tf.compat.v1.Session(config=self.config)
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
            saver.restore(self.sess, MODEL_PATH)
            test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
    
    def predict(self, pointcloud_data):
        np_points = pointcloud_data.points
        np_colors = pointcloud_data.colors
        true_label = pointcloud_data.labels
        pred_label,acc = self.eval_one_epoch(self.sess, self.ops, np_points, np_colors, true_label)
        predict_pcl_data = PointCloudData(np_points,np_colors,pred_label)
        return predict_pcl_data

    def eval_one_epoch(self,sess, ops, xyz, rgb,true_label):
        is_training = False
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        num_point = xyz.shape[0]
        xmax = 3.0
        xmin = -3.0
        current_data = np.zeros((num_point,6))
        current_data[:,0:3]  = (xyz[0:num_point,:]- xmin) / (xmax  - xmin )
        current_data[:,3:6]  = rgb[0:num_point,:]/(255*255)

        current_data = current_data.reshape(1,num_point, 6)
        current_label = np.zeros((1,num_point))

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE_EVAL 

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
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
        return pred_label,correct/num_point

if __name__ == "__main__":
    BATCH_SIZE = 1
    BATCH_SIZE_EVAL = 1
    NUM_POINT = 4096*2
    BASE_LEARNING_RATE = 0.001
    GPU_INDEX = 0
    MOMENTUM = 0.9
    OPTIMIZER = 'adam'
    DECAY_STEP = 300000
    DECAY_RATE = 0.5

    LOG_DIR = 'log'
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
    os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

    MAX_NUM_POINT = 4096
    NUM_CLASSES = 2

    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    #BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
    BN_DECAY_DECAY_STEP = float(DECAY_STEP)
    BN_DECAY_CLIP = 0.99

    HOSTNAME = socket.gethostname()

    print(f"Hello {HOSTNAME}")
    MODEL_PATH = "/home/peter/Documents/python_ws/PointNet_Custom_Object_Detection/log_1200_burr/model.ckpt"
    TEST_FILE = "/home/peter/Documents/python_ws/PointNet_Custom_Object_Detection/test_input/hole4_bigburr.pcd"
    model = PointNetDetector(MODEL_PATH)
    pcd_data = read_pcd_file(TEST_FILE)
    layers = pcd_data.uniform_slice_layer(50)
    for layer in layers:
        predict_pcd_data = model.predict(layer)
        break
    


