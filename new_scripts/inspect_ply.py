import open3d as o3d

# Load the saved point cloud
pcd = o3d.io.read_point_cloud("map_with_imu.ply")
print(f"Loaded point cloud has {len(pcd.points)} points.")

# If there are no points, it means the mapping or saving logic failed
if len(pcd.points) == 0:
    print("The point cloud is empty. Check your 3D mapping code.")
else:
    print("The point cloud has data. Ready to visualize.")
