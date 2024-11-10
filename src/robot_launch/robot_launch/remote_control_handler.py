import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import serial
import subprocess
from ament_index_python.packages import get_package_share_directory
import os

class RemoteControlHandler(Node):
    def __init__(self):
        super().__init__('remote_control_handler')

        # Retrieve or auto-detect serial port
        self.declare_parameter('serial_port', 'auto')
        serial_port = self.get_serial_port()

        # Set up serial connection to Pico
        try:
            self.serial = serial.Serial(serial_port, baudrate=115200, timeout=1)
            self.get_logger().info(f"Connected to Pico on {serial_port}")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect to Pico on {serial_port}: {e}")
            rclpy.shutdown()
            return

        # PWM parameters
        self.declare_parameter('max_speed_pwm', 2000)
        self.declare_parameter('min_speed_pwm', 1000)
        self.declare_parameter('neutral_pwm', 1500)
        self.max_speed_pwm = self.get_parameter('max_speed_pwm').get_parameter_value().integer_value
        self.min_speed_pwm = self.get_parameter('min_speed_pwm').get_parameter_value().integer_value
        self.neutral_pwm = self.get_parameter('neutral_pwm').get_parameter_value().integer_value

        # State variables
        self.emergency_stop = False

        # Subscribe to /cmd_vel and /joy
        self.cmd_vel_subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.joy_subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

    def get_serial_port(self):
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        if serial_port == 'auto':
            # Get the path to find_pico_port.py
            pico_port_script_path = os.path.join(
                get_package_share_directory('robot_launch'),
                'scripts',
                'find_pico_port.py'
            )
            # Run find_pico_port.py and capture output
            try:
                result = subprocess.run(
                    ['python3', pico_port_script_path],
                    capture_output=True,
                    text=True
                )
                output = result.stdout.strip()
                if output:
                    return output
                else:
                    self.get_logger().error("Failed to detect Pico port.")
                    return "/dev/ttyUSB0"  # Fallback
            except Exception as e:
                self.get_logger().error(f"Error detecting Pico port: {e}")
                return "/dev/ttyUSB0"  # Fallback
        return serial_port

    def cmd_vel_callback(self, msg):
        if not self.emergency_stop:
            linear_x = msg.linear.x
            angular_z = msg.angular.z

            # Convert to PWM
            speed_pwm = self.convert_speed_to_pwm(linear_x)
            steering_pwm = self.convert_steering_to_pwm(angular_z)

            # Send to Pico
            self.send_pwm_to_pico(speed_pwm, steering_pwm)
        else:
            # If emergency stop is active, send neutral PWM values
            self.send_pwm_to_pico(self.neutral_pwm, self.neutral_pwm)

    def joy_callback(self, joy_msg):
        # Assume button index 0 is the emergency stop button
        # Button index 1 could be a pause/resume command
        emergency_button = joy_msg.buttons[0]  # Example button for emergency stop
        pause_button = joy_msg.buttons[1]      # Example button for pause/resume recording

        # Emergency stop toggling
        if emergency_button == 1:  # Emergency stop button pressed
            self.emergency_stop = True
            self.get_logger().info("Emergency stop activated. Robot stopped.")
            self.send_pwm_to_pico(self.neutral_pwm, self.neutral_pwm)
        
        elif pause_button == 1:  # Pause recording command (or other action)
            self.handle_pause_resume()

    def handle_pause_resume(self):
        # Logic for pausing/resuming (e.g., triggering a topic or service call)
        # This could publish a message to a topic or call a service to control rosbag recording
        self.get_logger().info("Pause/Resume command received.")
        # Implement pause/resume functionality here if needed

    def convert_speed_to_pwm(self, linear_x):
        if linear_x > 0:
            return int(self.neutral_pwm + (linear_x * (self.max_speed_pwm - self.neutral_pwm)))
        elif linear_x < 0:
            return int(self.neutral_pwm + (linear_x * (self.neutral_pwm - self.min_speed_pwm)))
        else:
            return self.neutral_pwm

    def convert_steering_to_pwm(self, angular_z):
        return int(self.neutral_pwm + (angular_z * (self.max_speed_pwm - self.neutral_pwm)))

    def send_pwm_to_pico(self, speed_pwm, steering_pwm):
        command = f"SPEED:{speed_pwm};STEER:{steering_pwm}\n"
        self.serial.write(command.encode('utf-8'))

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RemoteControlHandler()
        rclpy.spin(node)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()