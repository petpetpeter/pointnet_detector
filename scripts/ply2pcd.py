#convert ply file to pcd file
import numpy as np
import os
import sys
import plyfile
import open3d as o3d

INPUT_DIR = '/home/peter/Documents/py_workspace/PointNet_Custom_Object_Detection/dense_hole'
OUTPUT_DIR = f'{INPUT_DIR}/pcd/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_pointcloud_list(input_dir):
    pointcloud_list = []
    for file in os.listdir(input_dir):
        if file.endswith('.ply'):
            pointcloud_list.append(file)
    return pointcloud_list

def ply2pcd(input_filename, output_filename):
    plydata = o3d.io.read_point_cloud(input_filename)
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(plydata.elements[0].data)
    #pcd.colors = o3d.utility.Vector3dVector(plydata.elements[1].data)
    o3d.io.write_point_cloud(output_filename, plydata)

if __name__ == '__main__':
    pointcloud_list = get_pointcloud_list(INPUT_DIR)
    for pointcloud in pointcloud_list:
        input_filename = os.path.join(INPUT_DIR, pointcloud)
        output_filename = os.path.join(OUTPUT_DIR, pointcloud.split('.')[0] + '.pcd')
        ply2pcd(input_filename, output_filename)
