import rclpy
import scripts.lower_control as lower_control
from rclpy.node import Node
from geometry_msgs.msg import Twist
from jeteja_launch_msgs.msg import PwmSignals

class CmdVelToPWMNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_pwm_node')

        # Subscribe to /cmd_vel topic
        self.create_subscription(Twist, '/cmd_vel_fixed_rate', self.cmd_vel_callback, 10)

        # Publisher for /pwm_signals topic
        self.pwm_pub = self.create_publisher(PwmSignals, '/pwm_signals', 10)

        self.get_logger().info("CmdVelToPWMNode is publishing to /pwm_signals.")

    def cmd_vel_callback(self, msg):
        """Callback to process /cmd_vel and convert to PWM."""
        linear_x = msg.linear.x 
        angular_z = msg.angular.z

        # Convert to PWM
        motor_pwm = lower_control.calculate_motor_duty_cycle(linear_x)
        steering_pwm = lower_control.calculate_steering_duty_cycle(angular_z)

        # Publish PWM values
        pwm_msg = PwmSignals()
        pwm_msg.stamp = self.get_clock().now().to_msg()  # Add timestamp
        pwm_msg.motor_pwm = motor_pwm
        pwm_msg.steering_pwm = steering_pwm

        self.pwm_pub.publish(pwm_msg)
        # self.get_logger().info(f"Published PWM: motor={motor_pwm}, steering={steering_pwm}")


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToPWMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down CmdVelToPWMNode.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
