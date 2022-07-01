import imp
import os, os.path
from matplotlib.colors import rgb2hex
import numpy as np
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

    def get_xyz_normalize_data(self):
        x_points = self.points[:,0]
        y_points = self.points[:,1]
        z_points = self.points[:,2]
        x_min,x_max = np.min(x_points),np.max(x_points)
        y_min,y_max = np.min(y_points),np.max(y_points)
        z_min,z_max = np.min(z_points),np.max(z_points)
        norm_x_points = (x_points - x_min)/(x_max - x_min)
        norm_y_points = (y_points - y_min)/(y_max - y_min)
        norm_z_points = (z_points - z_min)/(z_max - z_min)
        norm_points = np.concatenate((norm_x_points.reshape(-1,1),norm_y_points.reshape(-1,1),norm_z_points.reshape(-1,1)),axis = 1)
        print(f"shape x {norm_x_points.shape}")
        print(f"shape points {norm_points.shape}")
        norm_data = PointCloudData(norm_points,self.colors,self.labels)
        return norm_data
    
    def get_sphere_normalize_data(self):
        centroid = np.mean(self.points, axis=0)
        norm_points = self.points - centroid
        furthest_distance = np.max(np.linalg.norm(norm_points, axis=-1))
        norm_points = norm_points / furthest_distance 
        norm_data = PointCloudData(norm_points,self.colors,self.labels)
        return norm_data  
    
    def output_ply(self,filename):
        o3d_data = o3d.geometry.PointCloud()
        points = self.points
        colors = self.colors
        #paint labeled colors in green
        colors[self.labels == 1] = [0, 1, 0]
        o3d_data.points = o3d.utility.Vector3dVector(points)
        o3d_data.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename,o3d_data,write_ascii=True)


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
            list_of_layers.append(PointCloudData(layer_point,layer_color,layer_label))
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

if __name__ == "__main__":
    test_file = "/home/hayashi/Documents/pythonWS/pointnet_detector/test_input/hole4_bigburr.pcd"
    pc_data = read_pcd_file(test_file)
    pc_data.visualize()
    norm_pc_data = pc_data.get_sphere_normalize_data()
    norm_pc_data.visualize()
    norm_pc_data.output_ply("./test.ply")