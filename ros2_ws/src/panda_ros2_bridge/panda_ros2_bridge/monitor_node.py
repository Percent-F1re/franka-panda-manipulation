"""Actual subscriber example: report changes in Panda task state."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Monitor(Node):
    def __init__(self):
        super().__init__('panda_task_monitor')
        self.last = ''
        self.create_subscription(String, '/panda/task_status', self.on_status, 10)

    def on_status(self, msg):
        # Suppress repetition while retaining service event messages.
        content = msg.data
        if content.startswith('reset_applied'):
            self.get_logger().info(content)
            return
        parts = content.split()
        summary = ' '.join(x for x in parts if x.startswith(('controller=', 'run=', 'phase=')))
        if summary != self.last:
            self.get_logger().info(summary)
            self.last = summary


def main(args=None):
    rclpy.init(args=args)
    node = Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
