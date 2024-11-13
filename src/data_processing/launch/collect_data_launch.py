from datetime import datetime
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Arguments to configure topics and output directory
        DeclareLaunchArgument(
            'topics',
            default_value='/camera/color/image_raw/cmd_vel',
            description='Space-separated list of topics to record'
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value=f"data/rosbags/{datetime.now()}",
            description='Directory where the rosbag will be saved'
        ),
        
        # Execute rosbag2 record process
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', LaunchConfiguration('output_dir'),
                '--storage', 'sqlite3',  # Set the storage format (sqlite3 is default)
                '--compression-mode', 'file',  # Optional: compress the bag file
                '--compression-format', 'zstd',  # Optional: compression format
                LaunchConfiguration('topics')
            ],
            output='screen'
        ),
    ])
