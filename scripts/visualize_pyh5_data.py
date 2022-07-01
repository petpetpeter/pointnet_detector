import numpy as np 
import open3d 
import open3d as o3d
 
#dummycomment

from plyfile import PlyData, PlyElement
import h5py
#f = h5py.File('/home/peter/Documents/python_ws/PointNet_Custom_Object_Detection/train_input/train_data.h5','r')
f = h5py.File('/home/peter/Documents/python_ws/PointNet_Custom_Object_Detection/train_input/augmented_data/d2.h5','r')
FRAME_NUMBER = 90

print(f.keys())
data = f['data']
label = f['label']
print(np.unique(label[FRAME_NUMBER]))

xyz = np.zeros((len(data[FRAME_NUMBER]), 3))
colors = np.zeros((len(data[FRAME_NUMBER]), 3))

xyz[:, 0] = data[FRAME_NUMBER][:,0]
xyz[:, 1] = data[FRAME_NUMBER][:,1]
xyz[:, 2] = data[FRAME_NUMBER][:,2]
colors[:, 0] = data[FRAME_NUMBER][:,3]
colors[:, 1] = data[FRAME_NUMBER][:,4]
colors[:, 2] = data[FRAME_NUMBER][:,5]
colors[np.where(label[FRAME_NUMBER] == 1)] = [1,0,0]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)


o3d.visualization.draw_geometries([pcd])

