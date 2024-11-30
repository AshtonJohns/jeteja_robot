from os.path import join
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition

def generate_launch_description():

    # Declare arguments for toggling nodes
    manual_control_arg = DeclareLaunchArgument(
        'manual', default_value='false',
        description='Set to true to enable remote_control node'
    )

    autopilot_control_arg = DeclareLaunchArgument(
        'autopilot', default_value='false',
        description='Set to true to enable autopilot_control node'
    )

    # Paths to helper scripts and other launch files
    teleop_config = join(
        get_package_share_directory('jeteja_launch'),
        'config',
        'teleop_twist_joy.yaml'
    )

    joy_config = join(
        get_package_share_directory('jeteja_launch'),
        'config',
        'joy.yaml'
    )

    teleop_twist_joy_launch_path = join(
        get_package_share_directory('teleop_twist_joy'),
        'launch',
        'teleop-launch.py'
    )

    realsense2_config = join(
        get_package_share_directory('jeteja_launch'),
        'config',
        'realsense2_camera.yaml'
    )

    remote_control_handler_config = join(
        get_package_share_directory('jeteja_launch'),
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


    # Shared nodes between 'manual' and 'autopilot' arguments
    shared_nodes = [

        DeclareLaunchArgument(  # teleop config
            'teleop_config_filepath',
            default_value=teleop_config,
            description='Path to the teleop_twist_joy configuration file'
        ),

        DeclareLaunchArgument(  # joy config
            'joy_config_filepath',
            default_value=joy_config,
            description='Path to the joy configuration file'
        ),

        IncludeLaunchDescription(   # teleop launch
            PythonLaunchDescriptionSource(teleop_twist_joy_launch_path),
            launch_arguments={
                'config_filepath': LaunchConfiguration('teleop_config_filepath'),
                # 'joy_config': LaunchConfiguration('joy_config_filepath')
            }.items()
        ),

        DeclareLaunchArgument( # realsense config
        'realsense_config_filepath',
        default_value=realsense2_config,
        description='Path to the realsense2_camera configuration file'
        ),

        IncludeLaunchDescription(   # realsense launch
            PythonLaunchDescriptionSource(realsense2_camera_launch_path),
            launch_arguments={'config_file': LaunchConfiguration('realsense_config_filepath')}.items()
        ),

        IncludeLaunchDescription(   # rplidar
            PythonLaunchDescriptionSource(rplidar_launch_path),
        )
            
    ]

    manual_nodes = [
            Node(   # remote_control_handler node
            package='jeteja_launch',
            executable='remote_control_handler',
            name='remote_control_handler',
            output='screen',
            parameters=[],
            # parameters=[remote_control_handler_config],
            condition=IfCondition(LaunchConfiguration('manual'))
        ),

        Node(   # twist_stamper node
            package='twist_stamper',
            executable='twist_stamper',
            name='twist_stamper',
            remappings=[
                ('/cmd_vel_in', '/cmd_vel_fixed_rate'),  # Input: Rate-adjusted topic
                ('/cmd_vel_out', '/cmd_vel_stamped')    # Output: Stamped topic
            ],
            condition=IfCondition(LaunchConfiguration('manual'))
        ),

        Node(   # cmd_vel_fixed_rate node
            package='jeteja_launch',
            executable='cmd_vel_fixed_rate',
            name='cmd_vel_fixed_rate',
            output='screen',
            condition=IfCondition(LaunchConfiguration('manual'))
        ),
    ]

    # Instructional message to display the control mappings
    controls_info = LogInfo(
        msg="""
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
    )

    # Log which mode has been entered
    log_manual = LogInfo(
        condition=IfCondition(LaunchConfiguration('manual')),
        msg='Mode: Manual Control Enabled'
    )

    log_autopilot = LogInfo(
        condition=IfCondition(LaunchConfiguration('autopilot')),
        msg='Mode: Autopilot Enabled'
    )

    # log_no_mode = LogInfo(
    #     condition=IfCondition('not ' + LaunchConfiguration('manual') + ' and not ' + LaunchConfiguration('autopilot')),
    #     msg='No mode selected; launching shared nodes only.'
    # )

    return LaunchDescription([

        # --- ARGUMENTS ---
        manual_control_arg, # argument for manual 

        autopilot_control_arg,  # argument for autopilot

        # --- NODES ---

        *shared_nodes, # Shared nodes between autopilot and manual modes

        *manual_nodes, # Nodes strictly for manual control

        # --- INFORMATION FOR USER ---

        controls_info,

        log_manual,

        log_autopilot,

        # log_no_mode,

        # LogInfo(msg=f"Controller config: {teleop_config}"),

        # LogInfo(msg=f"Realsense camera config: {realsense2_config}")

    ])
