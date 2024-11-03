import open3d as o3d
import numpy as np

# Load the large point cloud
pcd = o3d.io.read_point_cloud("map_with_imu.ply")

# Split the point cloud into smaller chunks
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

chunk_size = len(points) // 10  # Split into 10 chunks

for i in range(10):
    chunk = o3d.geometry.PointCloud()
    start = i * chunk_size
    end = (i + 1) * chunk_size

    chunk.points = o3d.utility.Vector3dVector(points[start:end])
    chunk.colors = o3d.utility.Vector3dVector(colors[start:end])

    print(f"Visualizing chunk {i + 1} with {len(chunk.points)} points.")
    o3d.visualization.draw_geometries([chunk], window_name=f"Chunk {i + 1}")

    input("Press Enter to load the next chunk...")
