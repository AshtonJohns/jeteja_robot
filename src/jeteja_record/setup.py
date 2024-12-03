from setuptools import find_packages, setup

package_name = 'jeteja_record'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/collect_data_launch.py']),
        ('share/' + package_name + '/config', ['config/topics.yaml']), 
    ],
    install_requires=[
        'setuptools',       
        'launch', 
        'launch_ros', 
        'rclpy'
    ],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ashtonmjohns@gmail.com',
    description='Package for data extraction from rosbags for behavior cloning',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rosbag_manager = jeteja_record.rosbag_manager:main',
        ],
    },
)
