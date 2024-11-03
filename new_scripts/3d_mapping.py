import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import time
import csv

# Initialize RealSense pipeline
pipeline = rs.pipeline() #type: ignore
config = rs.config() #type: ignore
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #type: ignore
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #type: ignore
config.enable_stream(rs.stream.accel)  # Enable accelerometer #type: ignore
config.enable_stream(rs.stream.gyro)   # Enable gyroscope #type: ignore

# Start the pipeline
pipeline.start(config)

# IMU Data Logger
imu_log = open("imu_data.csv", "w", newline="")
csv_writer = csv.writer(imu_log)
csv_writer.writerow(["Timestamp", "Accel_x", "Accel_y", "Accel_z", "Gyro_x", "Gyro_y", "Gyro_z"])

def point_cloud_from_frames(color_frame, depth_frame, intrinsics):
    """Convert RealSense frames to an Open3D point cloud."""
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Create Open3D images
    o3d_color = o3d.geometry.Image(color_image)
    o3d_depth = o3d.geometry.Image(depth_image)

    # Generate RGBD Image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, convert_rgb_to_intensity=False
    )

    # Create camera intrinsics
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height,
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy
    )

    # Generate the point cloud from RGBD
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, pinhole_camera_intrinsic
    )
    
    # Transform for horizontal alignment
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    return pcd

def log_imu_data(imu_frame):
    """Log IMU data from accelerometer and gyroscope."""
    data = imu_frame.as_motion_frame().get_motion_data()
    timestamp = imu_frame.get_timestamp()
    csv_writer.writerow([timestamp, data.x, data.y, data.z, data.x, data.y, data.z])

def save_point_cloud(pcd, filename="map_with_imu.ply"):
    """Save the point cloud to a PLY file."""
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved map to {filename}")

def main():
    try:
        print("Starting 3D mapping with IMU logging. Press Ctrl+C to stop.")

        # Get camera intrinsics
        profile = pipeline.get_active_profile()
        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics() #type: ignore

        point_cloud = o3d.geometry.PointCloud()  # Accumulate all points here

        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Check for IMU data (accelerometer and gyroscope)
            for frame in frames:
                if frame.is_motion_frame():
                    log_imu_data(frame)

            if not depth_frame or not color_frame:
                continue

            # Display depth and color frames side-by-side in a horizontal layout
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show the combined image in a window
            cv2.imshow("RGB and Depth Stream", images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Convert frames to point cloud
            pcd = point_cloud_from_frames(color_frame, depth_frame, intrinsics)

            # Accumulate the point cloud
            point_cloud += pcd

            time.sleep(0.1)  # Adjust for desired frame rate

    except KeyboardInterrupt:
        print("\nMapping stopped.")
        save_point_cloud(point_cloud)

    finally:
        pipeline.stop()
        imu_log.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
