from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='15555'),
        DeclareLaunchArgument('token', default_value='local_demo_only'),
        Node(package='panda_ros2_bridge', executable='bridge_node',
             output='screen', parameters=[{
                 'port': LaunchConfiguration('port'),
                 'token': LaunchConfiguration('token'),
             }]),
        Node(package='panda_ros2_bridge', executable='monitor_node', output='screen'),
    ])
