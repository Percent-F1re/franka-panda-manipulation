"""Map existing MuJoCo /joint_states into MoveIt Panda URDF joint names.

Consumes published telemetry only; does not send actions to MuJoCo.
Works during live simulation and rosbag2 replay.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from panda_ros2_bridge.joint_mapping import translate_joint_state


class PandaJointAdapter(Node):
    def __init__(self):
        super().__init__('panda_joint_adapter')
        self.declare_parameter('default_gripper_opening_m', 0.04)
        self.opening = float(self.get_parameter('default_gripper_opening_m').value)
        self.pub = self.create_publisher(JointState, '/panda/urdf_joint_states', 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.create_subscription(Float32, '/panda/gripper_opening', self.on_opening, 10)
        self.bad_joint_log_count = 0
        self.get_logger().info('Listening to /joint_states; publishing /panda/urdf_joint_states')

    def on_opening(self, msg: Float32):
        self.opening = float(msg.data)

    def on_joints(self, src: JointState):
        try:
            names, q, dq = translate_joint_state(
                src.name, src.position, src.velocity, self.opening)
        except (ValueError, TypeError) as exc:
            if self.bad_joint_log_count < 5:
                self.get_logger().warn(f'Ignored incompatible JointState: {exc}')
                self.bad_joint_log_count += 1
            return
        dst = JointState()
        dst.header = src.header
        dst.name = names
        dst.position = q
        dst.velocity = dq
        self.pub.publish(dst)


def main(args=None):
    rclpy.init(args=args)
    node = PandaJointAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
