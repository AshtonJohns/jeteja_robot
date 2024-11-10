from os.path import join
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    # Paths to helper scripts and other launch files
    pico_port_script_path = join(
        get_package_share_directory('robot_launch'), 
        'scripts',
        'find_pico_port.py'
    )

    start_pico_launch_path = os.path.join(
        get_package_share_directory('micro_controller_launch'),
        'launch',
        'start_pico_launch.py'
    )

    teleop_config = join(
        get_package_share_directory('robot_launch'),
        'config',
        'teleop_twist_joy.yaml'
    )

    teleop_twist_joy_launch_path = join(
        get_package_share_directory('teleop_twist_joy'),
        'launch',
        'teleop-launch.py'
    )

    realsense2_config = join(
        get_package_share_directory('robot_launch'),
        'config',
        'realsense2_camera.yaml'
    )

    realsense2_camera_launch_path = join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )


    # # Define primary nodes that, if they exit, should trigger Pico process shutdown
    # realsense_camera_node = Node(
    #     package='realsense2_camera',
    #     executable='realsense2_camera_node',
    #     name='realsense_camera',
    #     output='screen',
    #     parameters=[{
    #         'color_width': 640,
    #         'color_height': 480,
    #         'color_fps': 30,
    #     }]
    # )

    remote_control_handler_node = Node(
        package='robot_launch',
        executable='remote_control_handler',
        name='remote_control_handler',
        output='screen',
        parameters=[{
            'max_speed_pwm': LaunchConfiguration('max_speed_pwm'),
            'min_speed_pwm': LaunchConfiguration('min_speed_pwm'),
            'neutral_pwm': LaunchConfiguration('neutral_pwm'),
            'serial_port': LaunchConfiguration('serial_port')
        }]
    )

    # Include the teleop_twist_joy launch file
    teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(teleop_twist_joy_launch_path),
        launch_arguments={'config_filepath': LaunchConfiguration('config_filepath')}.items()
    )

    # Include the realsense2_camera launch file
    rs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense2_camera_launch_path),
        launch_arguments={'config_filepath': LaunchConfiguration('config_filepath')}.items()
    )

    # Instructional message to display the control mappings
    controls_info = """
    --- Robot Controls ---
    - X Button: Stop
    - O Button: Pause
    - Right Stick: Forward/Backward
    - Left Stick: Left/Right
    - R1 : Enable bot moving (hold button)
    - △ : Turbo mode (hold button)
    """

    # # Teleop Node to publish to /cmd_vel
    # teleop_node = Node(
    #     package='teleop_twist_joy',  # Use 'teleop_twist_joy' if using a joystick
    #     executable='teleop_node',  # Executable name
    #     name='teleop_node',
    #     output='screen',
    #     parameters=[teleop_config],
    #     remappings=[('/cmd_vel', '/cmd_vel')]
    # )

    # # Include start_pico_launch.py from micro_controller_launch
    # pico_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(start_pico_launch_path),
    #     launch_arguments={'serial_port': LaunchConfiguration('serial_port')}.items()
    # )

    # # Event handler to stop the Pico launch when any critical node exits
    # shutdown_pico_on_exit = RegisterEventHandler(
    #     OnProcessExit(
    #         target_action=realsense_camera_node,
    #         on_exit=[pico_launch],
    #     )
    # )

    # # Event handler to stop the Pico process on shutdown
    # shutdown_pico_on_shutdown = RegisterEventHandler(
    #     OnShutdown(
    #         on_shutdown=[
    #             pico_launch  # This will ensure the Pico process is shut down on launch file shutdown
    #         ]
    #     )
    # )

    return LaunchDescription([
        DeclareLaunchArgument('config_filepath', default_value=realsense2_config,
                               description='Path to the realsense2_camera configuration file'),
        DeclareLaunchArgument('config_filepath', default_value=teleop_config, 
                              description='Path to the teleop_twist_joy configuration file'),
        # Launch arguments for easy parameter changes
        DeclareLaunchArgument(
            'max_speed_pwm', default_value='2000', description='Maximum PWM for speed'
        ),
        DeclareLaunchArgument(
            'min_speed_pwm', default_value='1000', description='Minimum PWM for speed'
        ),
        DeclareLaunchArgument(
            'neutral_pwm', default_value='1500', description='Neutral PWM'
        ),
        
        # Use the helper script to set the default serial port
        DeclareLaunchArgument(
            'serial_port',
            default_value=Command(['python3 ', pico_port_script_path]),
            description='Serial port for Pico (detected automatically)'
        ),

        # Add the nodes to the launch description
        rs_launch,

        remote_control_handler_node,

        teleop_launch,

        LogInfo(msg=controls_info),

        # Node(
        #     package='joy',
        #     executable='joy_node',
        #     name='joy_node',
        #     output='screen',
        #     parameters=[{
        #         'dev': '/dev/input/js0',
        #         'deadzone': 0.1,
        #         'autorepeat_rate': 20.0
        #     }]
        # ),

        # teleop_node,
        
        # # Include the Pico launch file 
        # pico_launch,
    ])
