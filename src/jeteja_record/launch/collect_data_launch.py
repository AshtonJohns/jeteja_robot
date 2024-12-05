import os
import yaml
from datetime import datetime
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the topics configuration file
    config_path = os.path.join(
        get_package_share_directory('jeteja_record'),
        'config',
        'topics.yaml'
    )

    rosbag_config = os.path.join(
        get_package_share_directory('jeteja_record'),
        'config',
        'rosbag.yaml'
    )

    # Get the workspace directory from an environment variable or default to the install space
    workspace_dir = os.getenv('ROS_WORKSPACE', os.path.abspath(os.path.join(
        get_package_share_directory('jeteja_launch'), '..', '..', '..', '..', # TODO is this a good practice? 
    )))

    # Define the output directory
    default_output_dir = os.path.join(workspace_dir, 'data', 'rosbags', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Load topics from the configuration file
    with open(config_path, 'r') as file:
        topics_config = yaml.safe_load(file)
    topics = topics_config['topics']  # List of topics from YAML

    # Load rosbag configurations
    with open(rosbag_config, 'r') as file:
        rosbag_config = yaml.safe_load(file)
    split_size = rosbag_config['split_size']

    return LaunchDescription([
        # Declare output directory argument
        DeclareLaunchArgument(
            'output_dir',
            default_value=default_output_dir,
            description='Directory where the rosbag will be saved'
        ),

        # Declare split size for rosbag argument
        DeclareLaunchArgument(
            'split_size',
            default_value=split_size,
            description='Split rosbag once it has reached the size.'
        ),
        
        # Rosbag Manager Node
        Node(
            package='jeteja_record',
            executable='rosbag_manager',
            name='rosbag_manager',
            output='screen',
            parameters=[{
                'topics': topics,  # Pass topics as a parameter
                'split_size': LaunchConfiguration('split_size'),
                'output_dir': LaunchConfiguration('output_dir')  # Pass output directory as a parameter
            }]
        )
    ])
