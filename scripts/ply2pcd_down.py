#convert ply file to pcd file
import numpy as np
import os
import sys
import plyfile
import open3d as o3d
from sklearn.utils import shuffle
import tqdm
from random import randrange

INPUT_DIR = '/home/peter/Documents/py_workspace/PointNet_Custom_Object_Detection/dense_hole'
OUTPUT_DIR = f'{INPUT_DIR}/pcd_400k'
number_of_pointcloud = 4096*100

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_pointcloud_list(input_dir):
    pointcloud_list = []
    for file in os.listdir(input_dir):
        if file.endswith('.ply'):
            pointcloud_list.append(file)
    return pointcloud_list

def down_sample(pointcloud, target_number_of_pointcloud):
    current_number_of_pointcloud = len(np.asarray(pointcloud.points))
    print(f"len before points: {current_number_of_pointcloud}")
    down_sample_scale = int(current_number_of_pointcloud / target_number_of_pointcloud)
    print(f"down_sample_scale: {down_sample_scale}")
    pointcloud = pointcloud.uniform_down_sample(down_sample_scale)
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)
    pcl = list(zip(points, colors))
    if len(points) > target_number_of_pointcloud:
        shuffle(pcl)
        pcl = pcl[:target_number_of_pointcloud]
        points, colors = zip(*pcl)
        points = np.asarray(points)
        colors = np.asarray(colors)
        #n_removing = len(points) - number_of_pointcloud
        #for i in tqdm.tqdm(range(n_removing)):
            #ind = randrange(len(points))
            #points = np.delete(points, ind, axis=0)
            #colors = np.delete(colors, ind, axis=0)
            #points = np.delete(points, np.random.randint(0, len(points)),axis=0)
            #colors = np.delete(colors, np.random.randint(0, len(colors)),axis=0)
    print(f"len after points: {len(points)},{points.shape}")
    return points, colors


def ply2pcd(input_filename, output_filename):
    plydata = o3d.io.read_point_cloud(input_filename)
    plydata.normalize_normals()
    
    np_points = np.asarray(plydata.points)
    # max_z = np.max(np_points[:,2])
    # print(f"max z value: {np.max(np_points[:,2])}")
    # if max_z > 20:
    #     plydata = plydata.voxel_down_sample(voxel_size=0.1)
    # else:
    #     plydata = plydata.voxel_down_sample(voxel_size=0.01)
    
    print(f"filename {input_filename} len down: {len(plydata.points)}")
    points,colors = down_sample(plydata, number_of_pointcloud)
    # if len(points) > number_of_pointcloud:
    #     n_removing = len(points) - number_of_pointcloud
    #     for i in tqdm.tqdm(range(n_removing)):
    #         points = np.delete(points, np.random.randint(0, len(points)),axis=0)
    #         colors = np.delete(colors, np.random.randint(0, len(colors)),axis=0)
    #     print(f"len after points: {len(points)},{points.shape}")
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points)
    #     pcd.colors = o3d.utility.Vector3dVector(colors)
    #     o3d.io.write_point_cloud(output_filename, pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_filename, pcd)



if __name__ == '__main__':
    pointcloud_list = get_pointcloud_list(INPUT_DIR)
    for pointcloud in pointcloud_list:
        input_filename = os.path.join(INPUT_DIR, pointcloud)
        output_filename = os.path.join(OUTPUT_DIR, pointcloud.split('.')[0] + '.pcd')
        ply2pcd(input_filename, output_filename)
