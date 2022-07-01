import h5py
import os, os.path
from matplotlib.colors import rgb2hex
import numpy as np
from plyfile import PlyData, PlyElement
import glob
import open3d as o3d
from pathlib import Path
from pyntcloud import PyntCloud

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

def colorToRGB(RGBint):
    b =  RGBint & 255
    g = (RGBint >> 8) & 255
    r =   (RGBint >> 16) & 255
    return [r/255, g/255, b/255]

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

NUM_FRAMES = 2
INPUT_DIR = "/home/peter/Documents/py_workspace/PointNet_Custom_Object_Detection/test_input"
NUMBER_OF_DATA = 4096*2

data = np.zeros((NUM_FRAMES, NUMBER_OF_DATA, 6), dtype = np.float32)
label = np.zeros((NUM_FRAMES,  NUMBER_OF_DATA),dtype = np.uint8)
 	
f = h5py.File(f'{INPUT_DIR}/test_data.h5', 'w')
i = -1

for file in os.listdir(INPUT_DIR):
    if file.endswith(".pcd"):
        i += 1
        pc_data = read_pcd_file(os.path.join(INPUT_DIR,file))
        pc_data.downsample(15)
        np_points = pc_data.points
        np_colors = pc_data.colors
        np_labels = pc_data.labels
        print(f"length of points {len(np_points)}")
        #numpy2o3d_visualizer(np_points,np_colors,np_labels)
        data[i,:,:] = np.concatenate((np_points,np_colors),axis = 1)
        label[i,:] = pc_data.labels
        print(f"name of file: {pc_data.file_name}")
    
f.create_dataset('data', data = data)
f.create_dataset('label', data = label)
                    

