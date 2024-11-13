from setuptools import find_packages, setup

package_name = 'data_processing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/collect_data_launch.py']),
        # ('share/' + package_name + '/data_processing', ['data_processing/extract_bags.py'])
    ],
    install_requires=[
        'setuptools',         # Core dependency for ROS 2 packages
        'opencv-python',       # Required for OpenCV functions (cv2)
        'argparse'             # Argument parsing library
    ],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ashtonmjohns@gmail.com',
    description='Package for data extraction from rosbags for behavior cloning',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'extract_bags = data_processing.extract_bags:main',  # Entry point for extract_bags.py
        ],
    },
)
