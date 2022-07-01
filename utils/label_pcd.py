import imp
import os, os.path
from matplotlib.colors import rgb2hex
import numpy as np
from plyfile import PlyData, PlyElement
import glob
import open3d as o3d
from pathlib import Path
from pyntcloud import PyntCloud
from tqdm import tqdm


class PointCloudData:
    def __init__(self, points, colors, labels):
        self.points = np.array(points)
        self.colors = np.array(colors)
        self.labels = np.array(labels)
    
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
    
    def uniform_slice_layer(self,number_of_layer = 100):
        np_points = self.points
        np_colors = self.colors
        np_labels = self.labels
        sorter = np.argsort(np_points[:,2])
        sorted_point = np_points[sorter]
        sorted_color = np_colors[sorter]
        sorted_label = np_labels[sorter]
        #test_pc = PointCloudData(sorted_point,sorted_color,sorted_label,"dummy")
        #test_pc.visualize()
        list_of_layers = []
        step_value = len(sorted_point) // number_of_layer
        for i in tqdm(range(number_of_layer)):
            layer_point = sorted_point[i*step_value:(i+1)*step_value]
            layer_color = sorted_color[i*step_value:(i+1)*step_value]
            layer_label = sorted_label[i*step_value:(i+1)*step_value]
            #if np.count_nonzero(layer_label == 1) > 0.05*len(layer_label):
            list_of_layers.append(PointCloudData(layer_point,layer_color,layer_label,"dummy"))
        return list_of_layers

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
    pc_data = PointCloudData(points,colors,labels)
    print(f"filename {file_name} color {pc_data.colors[0]}")
    return pc_data

def colorToRGB(RGBint):
    b =  RGBint & 255
    g = (RGBint >> 8) & 255
    r =   (RGBint >> 16) & 255
    return [r/255, g/255, b/255]