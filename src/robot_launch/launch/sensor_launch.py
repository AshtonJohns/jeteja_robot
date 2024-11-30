from os.path import join
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Paths to helper scripts and other launch files

    teleop_config = join(
        get_package_share_directory('robot_launch'),
        'config',
        'teleop_twist_joy.yaml'
    )

    joy_config = join(
        get_package_share_directory('robot_launch'),
        'config',
        'joy.yaml'
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

    remote_control_handler_config = join(
        get_package_share_directory('robot_launch'),
        'config',
        'remote_control_handler.yaml'
    )

    realsense2_camera_launch_path = join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )


    rplidar_launch_path = join(
        get_package_share_directory('rplidar_ros'),
        'launch',
        'rplidar_s2_launch.py'
    )

    # realsense_camera_node = Node(
    #     package='realsense2_camera',
    #     executable='realsense2_camera_node',
    #     name='realsense_camera',
    #     output='screen',
    #     parameters=[realsense2_config]
    # )

    remote_control_handler_node = Node(
        package='robot_launch',
        executable='remote_control_handler',
        name='remote_control_handler',
        output='screen',
        parameters=[remote_control_handler_config]
    )

    declare_teleop_config_arg = DeclareLaunchArgument(
        'teleop_config_filepath',
        default_value=teleop_config,
        description='Path to the teleop_twist_joy configuration file'
    )

    declare_joy_config_arg = DeclareLaunchArgument(
        'joy_config_filepath',
        default_value=joy_config,
        description='Path to the joy configuration file'
    )

    teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(teleop_twist_joy_launch_path),
        launch_arguments={
            'config_filepath': LaunchConfiguration('teleop_config_filepath'),
            # 'joy_config': LaunchConfiguration('joy_config_filepath')
        }.items()
    )

    cmd_vel_fixed_rate_node = Node(
        package='robot_launch',
        executable='cmd_vel_fixed_rate',
        name='cmd_vel_fixed_rate',
        output='screen',
    )

    twist_stamper_node = Node(
        package='twist_stamper',
        executable='twist_stamper',
        name='twist_stamper',
        remappings=[
            ('/cmd_vel_in', '/cmd_vel_fixed_rate'),  # Input: Rate-adjusted topic
            ('/cmd_vel_out', '/cmd_vel_stamped')    # Output: Stamped topic
        ]
    )

    declare_realsense_config_arg = DeclareLaunchArgument(
        'realsense_config_filepath',
        default_value=realsense2_config,
        description='Path to the realsense2_camera configuration file'
    )

    rs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense2_camera_launch_path),
        launch_arguments={'config_file': LaunchConfiguration('realsense_config_filepath')}.items()
    )

    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rplidar_launch_path),
    )

    # Instructional message to display the control mappings
    controls_info = """
    --- Robot Controls ---
    - OPTIONS: Start Pico
    - X: Reset Pico
    - □ : Enable bot moving        (hold button)
    - △ : Turbo mode               (hold button)
    - ○: Pause Recording
    - SHARE: Start Recording
    - Left Stick: Forward/Backward (requires enable)
    - Right Stick: Left/Right      (requires enable)
    """

    return LaunchDescription([

        # declare_joy_config_arg,
        
        declare_teleop_config_arg,

        declare_realsense_config_arg,

        remote_control_handler_node,

        teleop_launch,

        cmd_vel_fixed_rate_node,

        twist_stamper_node,

        rs_launch,

        # rplidar_launch,

        LogInfo(msg=controls_info),

        LogInfo(msg=f"Controller config: {teleop_config}"),

        LogInfo(msg=f"Realsense camera config: {realsense2_config}")

    ])
