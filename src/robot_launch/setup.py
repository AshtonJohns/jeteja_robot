from setuptools import find_packages, setup

package_name = 'robot_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/sensor_launch.py']),
        # ('share/' + package_name + '/scripts', ['scripts/find_pico_port.py']),
        # ('share/' + package_name + '/scripts', ['scripts/run.py']),
        ('share/' + package_name + '/scripts', ['scripts/main.py']),
        # ('share/' + package_name + '/scripts', ['scripts/reset.py']),
        ('share/' + package_name + '/config', ['config/teleop_twist_joy.yaml']),
        ('share/' + package_name + '/config', ['config/realsense2_camera.yaml']),
        ('share/' + package_name + '/config', ['config/remote_control_handler.yaml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ashtonmjohns@gmail.com',
    description='Package for launching and controlling a robot with ROS 2 and a Pico microcontroller',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'remote_control_handler = robot_launch.remote_control_handler:main',
        ],
    },
)
