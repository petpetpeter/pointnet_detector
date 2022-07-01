import imp
from typing import final
import h5py
import os, os.path
from matplotlib.colors import rgb2hex
import numpy as np
from plyfile import PlyData, PlyElement
import glob
import open3d as o3d
from pathlib import Path
from pyntcloud import PyntCloud
from tqdm import tqdm
import random


class PointCloudData:
    def __init__(self, points, colors, labels, file_name):
        self.points = np.array(points)
        self.colors = np.array(colors)
        self.labels = np.array(labels)
        self.file_name = file_name
    
    def downsample(self, factor):
        self.points = self.points[::factor,:]
        self.colors = self.colors[::factor,:]
        self.labels = self.labels[::factor]
    
    def visualize(self):
        pcd = o3d.geometry.PointCloud()
        points = self.points
        colors = self.colors
        #paint labeled colors in green
        colors[self.labels == 1] = [0, 1, 0]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])


def colorToRGB(RGBint):
    b =  RGBint & 255
    g = (RGBint >> 8) & 255
    r =   (RGBint >> 16) & 255
    return [r/255, g/255, b/255]

def slice_burr_layer(pcl_data,number_of_layer = 100):
    list_of_burr_layers = []
    _, _, z_sparse = zip(*pcl_data.points)
    step_value = (max(z_sparse) - min(z_sparse)) / number_of_layer
    for i in tqdm(range(number_of_layer)):
        layers_points = pcl_data.points[np.where(np.logical_and(z_sparse >= min(z_sparse) + i*step_value, z_sparse < min(z_sparse) + (i+1)*step_value))]
        layers_colors = pcl_data.colors[np.where(np.logical_and(z_sparse >= min(z_sparse) + i*step_value, z_sparse < min(z_sparse) + (i+1)*step_value))]
        layers_labels = pcl_data.labels[np.where(np.logical_and(z_sparse >= min(z_sparse) + i*step_value, z_sparse < min(z_sparse) + (i+1)*step_value))]
        if np.count_nonzero(layers_labels == 1) > 0.05*len(layers_labels):   
            list_of_burr_layers.append(PointCloudData(layers_points,layers_colors,layers_labels,"dummy"))
        
    return list_of_burr_layers

def uniform_slice_layer(pcl_data,number_of_layer = 100):
    np_points = pcl_data.points
    np_colors = pcl_data.colors
    np_labels = pcl_data.labels
    sorter = np.argsort(np_points[:,2])
    sorted_point = np_points[sorter]
    sorted_color = np_colors[sorter]
    sorted_label = np_labels[sorter]
    #test_pc = PointCloudData(sorted_point,sorted_color,sorted_label,"dummy")
    #test_pc.visualize()
    list_of_burr_layers = []
    list_of_no_burr_layers = []
    step_value = len(sorted_point) // number_of_layer
    for i in tqdm(range(number_of_layer)):
        layer_point = sorted_point[i*step_value:(i+1)*step_value]
        layer_color = sorted_color[i*step_value:(i+1)*step_value]
        layer_label = sorted_label[i*step_value:(i+1)*step_value]
        if np.count_nonzero(layer_label == 1) > 0.05*len(layer_label):
            list_of_burr_layers.append(PointCloudData(layer_point,layer_color,layer_label,"dummy"))
        else:
            list_of_no_burr_layers.append(PointCloudData(layer_point,layer_color,layer_label,"dummy"))
    return list_of_burr_layers,list_of_no_burr_layers



def read_pcd_file(file_name,header_line = 10):
    lines = open(file_name).readlines()
    points = []
    colors = []
    labels = []
    for line in lines[header_line:]:
        line = line.split()
        points.append([float(line[0]),float(line[1]),float(line[2])])
        color_rgb = colorToRGB(int(line[3]))
        colors.append(color_rgb)
        labels.append(int(line[4]))    
    pc_data = PointCloudData(points,colors,labels,file_name)
    print(f"filename {pc_data.file_name} color {pc_data.colors[0]}")
    return pc_data

def numpy2o3d_visualizer(points,colors,labels):
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(points)
    #o3d_points.colors = o3d.utility.Vector3dVector(colors)
    labels_color = [0,1,0]
    colors[np.where(labels == 1)] = labels_color
    o3d_points.colors = o3d.utility.Vector3dVector(colors)
    #o3d_points.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([o3d_points])


INPUT_DIR = "/home/hayashi/Documents/pythonWS/pointnet_detector/train_input"
NUMBER_OF_DATA = 4096*2


 	
f = h5py.File(f'{INPUT_DIR}/train_data.h5', 'w')
i = -1


final_burr_layer_list = []
final_no_burr_layer_list = []
for file in os.listdir(INPUT_DIR):
    if file.endswith(".pcd"):
        pc_data = read_pcd_file(os.path.join(INPUT_DIR,file))
        #pc_data.visualize()
        list_of_burr_layers,list_of_no_burr_layers = uniform_slice_layer(pc_data,number_of_layer = 50)
        final_burr_layer_list = final_burr_layer_list + list_of_burr_layers
        final_no_burr_layer_list = final_no_burr_layer_list + list_of_no_burr_layers

number_of_burr_data = len(final_burr_layer_list)
print(f"number of burr data {number_of_burr_data}")
random.shuffle(final_no_burr_layer_list) #shuffle no burr layer
final_no_burr_layer_list = final_no_burr_layer_list[:number_of_burr_data]
mixed_data = final_burr_layer_list + final_no_burr_layer_list
    
NUM_FRAMES = len(mixed_data)
data = np.zeros((NUM_FRAMES, NUMBER_OF_DATA, 6), dtype = np.float32)
label = np.zeros((NUM_FRAMES,  NUMBER_OF_DATA),dtype = np.uint8)
i = -1
for layer in tqdm(mixed_data):
    i += 1
    np_points = layer.points
    np_colors = layer.colors
    np_labels = layer.labels
    print(f"length of points {len(np_points)}")
    #numpy2o3d_visualizer(np_points,np_colors,np_labels)
    
    data[i,:,:] = np.concatenate((np_points,np_colors),axis = 1)
    label[i,:] = np_labels

print(f"final data shaoe:{np.shape(data)}")
print(f"final label shape:{np.shape(label)}")
    
f.create_dataset('data', data = data)
f.create_dataset('label', data = label)
                    

