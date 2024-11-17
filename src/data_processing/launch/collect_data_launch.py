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
        get_package_share_directory('data_processing'),
        'config',
        'topics.yaml'
    )

    # Get the package directory
    package_dir = get_package_share_directory('data_processing')

    # Get the workspace directory (two levels up from the package share directory)
    workspace_dir = os.path.abspath(os.path.join(package_dir, '..', '..'))

    # Define the output directory
    default_output_dir = os.path.join(workspace_dir, 'data', 'rosbags', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Load topics from the configuration file
    with open(config_path, 'r') as file:
        topics_config = yaml.safe_load(file)
    topics = topics_config['topics']  # List of topics from YAML

    return LaunchDescription([
        # Declare output directory argument
        DeclareLaunchArgument(
            'output_dir',
            default_value=default_output_dir,
            description='Directory where the rosbag will be saved'
        ),
        
        # Rosbag Manager Node
        Node(
            package='data_processing',
            executable='rosbag_manager',
            name='rosbag_manager',
            output='screen',
            parameters=[{
                'topics': topics,  # Pass topics as a parameter
                'output_dir': LaunchConfiguration('output_dir')  # Pass output directory as a parameter
            }]
        )
    ])
