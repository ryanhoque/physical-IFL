from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='block_pusher',
            namespace='block_pusher',
            executable='cloud_executor',
            name='block_pusher'
        )
    ])