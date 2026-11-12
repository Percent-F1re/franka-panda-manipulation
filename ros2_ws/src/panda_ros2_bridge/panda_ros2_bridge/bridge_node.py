"""WSL2 rclpy bridge: UDP telemetry -> ROS standard msgs, TF2, RViz Marker.

ROS2 reset Service -> UDP reset request -> Windows demo runner only.
No telemetry is fed back into the SAC policy or the physics controllers.
"""
from __future__ import annotations

import json
import socket
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from builtin_interfaces.msg import Time as RosTime
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32, Float32MultiArray, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

PROTOCOL = 'panda_bridge_v1'


def stamp(t):
    t = max(0.0, float(t))
    secs = int(t)
    return RosTime(sec=secs, nanosec=min(999999999, int(round((t - secs) * 1_000_000_000))))


def position_orientation(msg, xyz, quat_wxyz):
    msg.position.x, msg.position.y, msg.position.z = map(float, xyz)
    msg.orientation.w = float(quat_wxyz[0])
    msg.orientation.x = float(quat_wxyz[1])
    msg.orientation.y = float(quat_wxyz[2])
    msg.orientation.z = float(quat_wxyz[3])


class Bridge(Node):
    def __init__(self):
        super().__init__('panda_ros2_bridge')
        self.declare_parameter('port', 15555)
        self.declare_parameter('token', 'local_demo_only')
        self.port = self.get_parameter('port').value
        self.token = self.get_parameter('token').value
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(('0.0.0.0', int(self.port)))
        self.udp.setblocking(False)
        self.last_client = None
        self.last_run_id = None
        self.last_phase = None
        self.trajectory = []
        self.clock_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.clock_pub = self.create_publisher(Clock, '/clock', self.clock_qos)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.ee_pub = self.create_publisher(PoseStamped, '/panda/ee_pose', 10)
        self.cube_pub = self.create_publisher(PoseStamped, '/panda/cube_pose', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/panda/target_pose', 10)
        self.obs_pub = self.create_publisher(Float32MultiArray, '/panda/observation', 10)
        self.action_pub = self.create_publisher(Float32MultiArray, '/panda/sac_action', 10)
        self.gripper_pub = self.create_publisher(Float32, '/panda/gripper_opening', 10)
        self.status_pub = self.create_publisher(String, '/panda/task_status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/panda/markers', 10)
        self.tf = TransformBroadcaster(self)
        self.reset_srv = self.create_service(Trigger, '/panda/reset_episode', self.handle_reset)
        # Wall-clock executor timer, even while /clock is paused or changing.
        self.create_timer(0.01, self.poll_udp)
        self.get_logger().info(f'Waiting for Windows telemetry on UDP 0.0.0.0:{self.port}')

    def handle_reset(self, _request, response):
        if not self.last_client:
            response.success = False
            response.message = 'No Windows demo runner detected; start ros2_demo_runner.py first.'
            return response
        rid = uuid.uuid4().hex[:12]
        wire = dict(protocol=PROTOCOL, token=self.token, kind='command',
                    command='reset', request_id=rid)
        self.udp.sendto(json.dumps(wire).encode('utf-8'), self.last_client)
        response.success = True
        response.message = f'Reset REQUEST sent id={rid}; confirm reset_applied on /panda/task_status.'
        return response

    def poll_udp(self):
        # Drain with cap so a high-rate remote can't starve ROS callbacks.
        for _ in range(64):
            try:
                raw, source = self.udp.recvfrom(16000)
            except BlockingIOError:
                break
            if len(raw) >= 16000:
                continue
            try:
                p = json.loads(raw.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                continue
            if p.get('protocol') != PROTOCOL or p.get('token') != self.token:
                continue
            if p.get('kind') == 'event':
                self.status_pub.publish(String(data=f"{p.get('event')} run={p.get('run_id')} id={p.get('request_id')}"))
                continue
            if p.get('kind') != 'state':
                continue
            self.last_client = source  # only registered sender can receive reset requests
            try:
                self.publish_state(p)
            except (ValueError, KeyError, TypeError) as exc:
                self.get_logger().warn(f'Bad telemetry packet: {exc}')

    def publish_state(self, p):
        t = stamp(p['clock_s'])
        self.clock_pub.publish(Clock(clock=t))
        joint = JointState()
        joint.header.frame_id = 'world'
        joint.header.stamp = t
        joint.name = list(p['joint_names'])
        joint.position = list(map(float, p['joint_q']))
        joint.velocity = list(map(float, p['joint_dq']))
        if len(joint.name) != 7 or len(joint.position) != 7 or len(joint.velocity) != 7:
            raise ValueError('Expected 7 Panda joints')
        self.joint_pub.publish(joint)
        if 'gripper_opening_m' in p:
            self.gripper_pub.publish(Float32(data=float(p['gripper_opening_m'])))
        identity = [1.0, 0.0, 0.0, 0.0]
        pose_specs = [
            (self.ee_pub, 'panda_hand', p['hand_xyz'], p['hand_quat_wxyz']),
            (self.cube_pub, 'panda_cube', p['cube_xyz'], p['cube_quat_wxyz']),
            (self.target_pub, 'panda_target', p['target_xyz'], identity),
        ]
        for publisher, child_frame, xyz, quat in pose_specs:
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.header.stamp = t
            position_orientation(pose.pose, xyz, quat)
            publisher.publish(pose)
            tr = TransformStamped()
            tr.header = pose.header
            tr.child_frame_id = child_frame
            tr.transform.translation.x = pose.pose.position.x
            tr.transform.translation.y = pose.pose.position.y
            tr.transform.translation.z = pose.pose.position.z
            tr.transform.rotation = pose.pose.orientation
            self.tf.sendTransform(tr)

        self.obs_pub.publish(Float32MultiArray(data=list(map(float, p['observation']))))
        self.action_pub.publish(Float32MultiArray(data=list(map(float, p['sac_action']))))
        if len(p['observation']) != 46 or len(p['sac_action']) != 4:
            raise ValueError('Unexpected SAC dimensions')
        phase = f"controller={p['controller']} run={p['run_id']} seed={p['seed']} " \
                f"step={p['step']} phase={p['phase']} original_success={p['original_is_success']}"
        self.status_pub.publish(String(data=phase))
        if p['phase'] != self.last_phase or p['run_id'] != self.last_run_id:
            self.get_logger().info(phase)
            self.last_phase = p['phase']
        if p['run_id'] != self.last_run_id:
            self.trajectory.clear()
            self.last_run_id = p['run_id']
        if not self.trajectory or p['hand_xyz'] != self.trajectory[-1]:
            self.trajectory.append(list(p['hand_xyz']))
            if len(self.trajectory) > 1500:
                del self.trajectory[:len(self.trajectory) - 1500]
        self.publish_markers(p, t)

    def publish_markers(self, p, t):
        from geometry_msgs.msg import Point
        markers = MarkerArray()
        for idx, (namespace, xyz, side, rgb) in enumerate([
            ('cube', p['cube_xyz'], p['cube_side_m'], (0.97, 0.55, 0.12)),
            ('target', p['target_xyz'], p['cube_side_m'], (0.12, 0.76, 0.32)),
        ]):
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = t
            m.ns = namespace
            m.id = idx
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, xyz)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = max(0.01, float(side))
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 0.70 if idx == 1 else 1.0
            markers.markers.append(m)
        line = Marker()
        line.header.frame_id = 'world'
        line.header.stamp = t
        line.ns = 'hand_trajectory'
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.006
        line.color.r = 0.13
        line.color.g = 0.45
        line.color.b = 0.97
        line.color.a = 1.0
        for pt in self.trajectory:
            point = Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
            line.points.append(point)
        markers.markers.append(line)
        self.marker_pub.publish(markers)

    def destroy_node(self):
        self.udp.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
