from setuptools import find_packages, setup

package_name = 'jeteja_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    py_modules=['scripts.pico_handler', 'scripts.lower_control'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/autopilot.yaml']),
        ('share/' + package_name + '/config', ['config/joy.yaml']),
        ('share/' + package_name + '/config', ['config/master_config.py']),
        ('share/' + package_name + '/config', ['config/realsense2_camera.yaml']),
        ('share/' + package_name + '/config', ['config/remote_control_handler.yaml']),
        ('share/' + package_name + '/config', ['config/teleop_twist_joy.yaml']),
        ('share/' + package_name + '/scripts', ['scripts/image_processing.py']),        
        ('share/' + package_name + '/scripts', ['scripts/lower_control.py']),  
        ('share/' + package_name + '/scripts', ['scripts/main.py']),
        ('share/' + package_name + '/scripts', ['scripts/model_inference_handler.py']),        
        ('share/' + package_name + '/scripts', ['scripts/pico_handler.py']),
        ('share/' + package_name + '/launch', ['launch/sensor_launch.py']),
    ],
    install_requires=['setuptools', 
                      'pyserial',
                      'launch', 
                      'launch_ros', 
                      'rclpy',
                      'jeteja_launch_msgs',
                      'opencv-python',],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ashtonmjohns@gmail.com',
    description='Package for launching and controlling a robot with ROS 2 and a Pico microcontroller',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'remote_control_handler = jeteja_launch.remote_control_handler:main',
            'cmd_vel_fixed_rate = jeteja_launch.cmd_vel_fixed_rate:main',
            'cmd_vel_to_pwm = jeteja_launch.cmd_vel_to_pwm:main',
            'autopilot_inference_handler = jeteja_launch.autopilot_inference_handler:main',
        ],
    },
)
