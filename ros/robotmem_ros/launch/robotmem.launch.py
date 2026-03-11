"""robotmem ROS 2 Launch 文件

用法:
    ros2 launch robotmem_ros robotmem.launch.py
    ros2 launch robotmem_ros robotmem.launch.py collection:=robot1
    ros2 launch robotmem_ros robotmem.launch.py ns:=/robot1 collection:=robot1
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_node(context, *args, **kwargs):
    """OpaqueFunction: collection 非空时才覆盖 params.yaml 默认值"""
    params_file = LaunchConfiguration('params_file').perform(context)
    ns = LaunchConfiguration('ns').perform(context)
    collection = LaunchConfiguration('collection').perform(context)

    parameters = [params_file]
    if collection:
        parameters.append({'collection': collection})

    return [Node(
        package='robotmem_ros',
        executable='robotmem_node',
        name='robotmem',
        namespace=ns,
        parameters=parameters,
        output='screen',
    )]


def generate_launch_description():
    pkg_dir = get_package_share_directory('robotmem_ros')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='', description='Namespace for multi-robot'),
        DeclareLaunchArgument('collection', default_value='', description='Override collection'),
        DeclareLaunchArgument('params_file', default_value=params_file, description='Params YAML'),
        OpaqueFunction(function=_launch_node),
    ])
