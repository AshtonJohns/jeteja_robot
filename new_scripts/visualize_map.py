import open3d as o3d

def visualize_point_cloud(file_path):
    """Load and display the saved point cloud."""
    pcd = o3d.io.read_point_cloud(file_path)
    print("Loaded point cloud:")
    print(pcd)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="3D Map with IMU")

if __name__ == "__main__":
    # Replace with your actual file path
    visualize_point_cloud("map_with_imu.ply")
