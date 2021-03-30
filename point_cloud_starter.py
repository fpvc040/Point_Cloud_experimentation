
import numpy as np
import time
import open3d as od
import pandas as pd
import matplotlib.pyplot as plt

## http://www.open3d.org/docs/release/tutorial/Basic/
#Visualization of point cloud
pcd = od.io.read_point_cloud("test_files/KITTI/000002.pcd")
od.visualization.draw_geometries([pcd])
print(np.asarray(pcd.points))

# Voxel grid Downsampling
print(f"Points before downsampling: {len(pcd.points)} ")
pcd = pcd.voxel_down_sample(voxel_size=0.05)
print(f"Points after downsampling: {len(pcd.points)}")# DOWNSAMPLING

#Ransac plane detection
_, inliers = pcd.segment_plane(distance_threshold=0.25,
                                         ransac_n=3,
                                         num_iterations=1000)
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([0, 1.0, 0])
od.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# DBSCAN for object clustering
pcd = outlier_cloud
with od.utility.VerbosityContextManager(
        od.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.45, min_points=7, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = od.utility.Vector3dVector(colors[:, :3])
od.visualization.draw_geometries([pcd])

# Draw bounding boxes
obbs = []
indexes = pd.Series(range(len(labels))).groupby(labels,sort = False).apply(list).tolist()
MAX_POINTS = 300
MIN_POINTS = 40

for i in range(0,len(indexes)):
	nb_points = len(outlier_cloud.select_by_index(indexes[i]).points)
	if nb_points > MIN_POINTS:
		sub_cloud =outlier_cloud.select_by_index(indexes[i])
		obb = sub_cloud.get_axis_aligned_bounding_box()
		obb.color= (0,0,1)
		obbs.append(obb)

list_of_visuals = []
list_of_visuals.append(outlier_cloud)
list_of_visuals.extend(obbs)
list_of_visuals.append(inlier_cloud)
od.visualization.draw_geometries(list_of_visuals)