import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ConstantRateCmdVelPublisher(Node):
    def __init__(self):
        super().__init__('constant_rate_cmd_vel_publisher')

        # Subscribe to the original `/cmd_vel`
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',  # Input topic
            self.cmd_vel_callback,
            10
        )

        # Publish to `/cmd_vel_fixed_rate`
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel_fixed_rate',  # Output topic
            10
        )

        # Publish messages at a constant rate (e.g., 30 Hz)
        self.timer = self.create_timer(1 / 55.0, self.publish_last_cmd_vel)  # 50 Hz

        # State to hold the last received `/cmd_vel` message
        self.last_cmd_vel = None

    def cmd_vel_callback(self, msg):
        # Store the last received `/cmd_vel` message
        self.last_cmd_vel = msg

    def publish_last_cmd_vel(self):
        # Publish the last received message or a zero message
        if self.last_cmd_vel is not None:
            self.cmd_vel_publisher.publish(self.last_cmd_vel)
        else:
            zero_cmd_vel = Twist()
            self.cmd_vel_publisher.publish(zero_cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = ConstantRateCmdVelPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()