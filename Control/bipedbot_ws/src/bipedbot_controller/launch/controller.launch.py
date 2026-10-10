from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="true")
    wheel_radius_arg = DeclareLaunchArgument("wheel_radius", default_value="0.037")
    wheel_separation_arg = DeclareLaunchArgument("wheel_separation", default_value="0.12")

    use_sim_time = LaunchConfiguration("use_sim_time")
    wheel_radius = LaunchConfiguration("wheel_radius")
    wheel_separation = LaunchConfiguration("wheel_separation")

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    wheel_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["simple_velocity_controller", "--controller-manager", "/controller_manager"],
    )

    simple_controller = Node(
        package="bipedbot_controller",
        executable="simple_controller.py",
        parameters=[{
            # LaunchConfiguration gives strings, so declare the real type
            "wheel_radius": ParameterValue(wheel_radius, value_type=float),
            "wheel_separation": ParameterValue(wheel_separation, value_type=float),
            "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
        }],
    )


    return LaunchDescription([
        use_sim_time_arg,
        wheel_radius_arg,
        wheel_separation_arg,
        joint_state_broadcaster_spawner,
        wheel_controller_spawner,
        simple_controller,
    ])