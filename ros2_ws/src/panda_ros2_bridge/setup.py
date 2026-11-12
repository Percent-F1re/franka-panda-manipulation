from setuptools import setup
from glob import glob

name = 'panda_ros2_bridge'
setup(
    name=name, version='0.1.0', packages=[name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + name]),
        ('share/' + name, ['package.xml']),
        ('share/' + name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='Project maintainer',
    maintainer_email='maintainer@example.invalid',
    description='WSL2 ROS2 telemetry and reset bridge for robosuite Panda',
    license='Proprietary',
    entry_points={'console_scripts': [
        'bridge_node = panda_ros2_bridge.bridge_node:main',
        'monitor_node = panda_ros2_bridge.monitor_node:main',
        'joint_adapter_node = panda_ros2_bridge.joint_adapter_node:main',
    ]},
)
