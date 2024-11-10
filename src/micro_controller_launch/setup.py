from setuptools import find_packages, setup
import os

package_name = 'micro_controller_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the launch directory
        ('share/' + package_name + '/launch', ['launch/start_pico_launch.py']),
        # Include the pico_main.py file in the install
        ('share/' + package_name + '/pico', ['pico/pico_main.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ashtonmjohns@gmail.com',
    description='Launch a MicroPython script on the Pico using mpremote',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Add any additional console scripts here
        ],
    },
)
