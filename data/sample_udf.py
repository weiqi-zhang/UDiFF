import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh
import argparse
from pyhocon import ConfigFactory
import time
from fnmatch import fnmatch
import multiprocessing as mp

resolution = 256
def convert_udf_grid(arg):
    mesh_path, mesh_name = arg
    print(mesh_path, '  ', mesh_name)
    pointcloud = trimesh.load(os.path.join(mesh_path, mesh_name)).vertices
    pointcloud = np.asarray(pointcloud)

    shape_scale = 1.1 * np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    object_bbox_min = np.array([-0.5, -0.5, -0.5])
    object_bbox_max = np.array([0.5, 0.5, 0.5])

    ptree = cKDTree(pointcloud)

    N = 32
    X = torch.linspace(object_bbox_min[0], object_bbox_max[0], resolution).split(N)
    Y = torch.linspace(object_bbox_min[1], object_bbox_max[1], resolution).split(N)
    Z = torch.linspace(object_bbox_min[2], object_bbox_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                d = ptree.query(pts)
                val = d[0].reshape(len(xs), len(ys), len(zs))
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    np.save(os.path.join(mesh_path, mesh_name[:-4]) + '.npy', u)

if __name__ == '__main__':
    workers = 1

    pattern = "*.ply"
    root = r'<PATH_TO_DEEPFASHION>'

    args = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                args.append((path, name))
    
    pool = mp.Pool(workers)
    pool.map(convert_udf_grid, args)
    