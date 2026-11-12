"""Optional full Panda URDF visualization for the existing ROS2 bridge.

Starts NO MuJoCo simulation, controller, reset service or UDP listener.
Run alongside the original Bridge or alongside rosbag2 playback.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_prefixed_panda_urdf():
    pkg = get_package_share_directory('moveit_resources_panda_description')
    urdf_file = Path(pkg) / 'urdf' / 'panda.urdf'
    if not urdf_file.is_file():
        raise RuntimeError(f'Cannot find Panda URDF at {urdf_file}')
    root = ET.parse(str(urdf_file)).getroot()
    if root.tag != 'robot':
        raise RuntimeError('Invalid Panda URDF root element')
    # Existing Bridge publishes world -> panda_hand from MuJoCo. Rename all
    # URDF links to viz_panda_* to avoid duplicate TF child_frame_id.
    for link in root.findall('link'):
        link.set('name', 'viz_' + link.get('name'))
    for joint in root.findall('joint'):
        for tag in ('parent', 'child'):
            el = joint.find(tag)
            if el is not None:
                el.set('link', 'viz_' + el.get('link'))
    # Package mesh URLs remain unchanged, resolving from installed ROS package.
    return ET.tostring(root, encoding='unicode')


def generate_launch_description():
    description = load_prefixed_panda_urdf()
    return LaunchDescription([
        DeclareLaunchArgument('base_x', default_value='0.0'),
        DeclareLaunchArgument('base_y', default_value='0.0'),
        DeclareLaunchArgument('base_z', default_value='0.0'),
        DeclareLaunchArgument('base_yaw', default_value='0.0'),
        Node(
            package='panda_ros2_bridge', executable='joint_adapter_node',
            name='panda_joint_adapter', output='screen',
            parameters=[{'use_sim_time': True,
                         'default_gripper_opening_m': 0.04}]),
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='panda_viz_robot_state_publisher', output='screen',
            remappings=[('/joint_states', '/panda/urdf_joint_states')],
            parameters=[{'robot_description': ParameterValue(description, value_type=str),
                         'use_sim_time': True,
                         'publish_frequency': 50.0,
                         'ignore_timestamp': True}]),
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='panda_viz_world_to_base', output='screen',
            arguments=['--x', LaunchConfiguration('base_x'),
                       '--y', LaunchConfiguration('base_y'),
                       '--z', LaunchConfiguration('base_z'),
                       '--yaw', LaunchConfiguration('base_yaw'),
                       '--frame-id', 'world',
                       '--child-frame-id', 'viz_panda_link0']),
    ])
