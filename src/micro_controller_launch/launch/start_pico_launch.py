from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to pico_main.py within the package
    default_pico_script_path = os.path.join(
        get_package_share_directory('micro_controller_launch'),
        'pico',
        'pico_main.py'
    )

    return LaunchDescription([
        # Argument for specifying the Pico script file
        DeclareLaunchArgument(
            'pico_script',
            default_value=default_pico_script_path,
            description='Path to the MicroPython script to run on the Pico'
        ),
        
        # Command to start the MicroPython script on the Pico with python3 -m mpremote
        ExecuteProcess(
            cmd=['python3', '-m', 'mpremote', 'run', LaunchConfiguration('pico_script')],
            output='screen'
        ),
    ])
