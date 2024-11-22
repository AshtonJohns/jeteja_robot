import serial
from time import time
import subprocess
import os
import rclpy
from std_msgs.msg import String
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Joy
from ament_index_python.packages import get_package_share_directory

class PicoHandler(object):
    def __init__(self):
        pass

    def kill(self):
        try:
            self.terminate()
        except:
            pass

    def reset(self):
        self.kill()
        command = ["python", "-m", "mpremote", "reset"]
        try:
            res = subprocess.Popen(command)
            return 1
        except:
            return 0
            
    def run(self,path):
        command = ["python", "-m", "mpremote", "run", path]
        try:
            res = subprocess.Popen(command)
            self.terminate = res.terminate
            return 1
        except:
            return 0
        
    def get_pico_port(self):
        # Run `mpremote connect list` and capture the output
        result = subprocess.run(['python3', '-m', 'mpremote', 'connect', 'list'], capture_output=True, text=True)
        output = result.stdout

        # Parse the output to find the serial port (assuming the first line has the required port)
        lines = output.splitlines()
        for line in lines:
            if line.startswith('/dev'):
                return line.split()[0]  # Extract the port (e.g., /dev/ttyACM0)
        return 0

class RemoteControlHandler(Node):
    def __init__(self):
        super().__init__('remote_control_handler')

        # Pico process execution
        self.pico_execute = PicoHandler()

        # Retrieve or auto-detect serial port
        self.declare_parameter('serial_port', 'auto')
        self.set_serial()

        # Get main.py path for pico 
        self.pico_port_script_path = os.path.join(
                get_package_share_directory('robot_launch'),
                'scripts',
                'main.py'
            )
        
        # Declare all parameters with default value 0
        self.declare_parameter('motor_min_duty_cycle', 0)
        self.declare_parameter('motor_neutral_duty_cycle', 0)
        self.declare_parameter('motor_max_duty_cycle', 0)

        self.declare_parameter('steering_min_duty_cycle', 0)
        self.declare_parameter('steering_neutral_duty_cycle', 0)
        self.declare_parameter('steering_max_duty_cycle', 0)

        # Motor PWM duty cycle parameters
        self.motor_min_duty_cycle = self.get_parameter('motor_min_duty_cycle').get_parameter_value().integer_value
        self.motor_neutral_duty_cycle = self.get_parameter('motor_neutral_duty_cycle').get_parameter_value().integer_value
        self.motor_max_duty_cycle = self.get_parameter('motor_max_duty_cycle').get_parameter_value().integer_value

        # Steering PWM duty cycle parameters
        self.steering_min_duty_cycle = self.get_parameter('steering_min_duty_cycle').get_parameter_value().integer_value
        self.steering_neutral_duty_cycle = self.get_parameter('steering_neutral_duty_cycle').get_parameter_value().integer_value
        self.steering_max_duty_cycle = self.get_parameter('steering_max_duty_cycle').get_parameter_value().integer_value

        # State variables (True == alive, False == dead)
        self.microcontroller_state = False
        self.recording_state = False
        self.enable_recording_state = False

        # Subscribers and publishers
        self.cmd_vel_subscription = self.create_subscription(Twist, '/cmd_vel_fixed_rate', self.cmd_vel_callback, 15) # not timestamped, but a steady publish rate
        # self.cmd_vel_subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 15) # not timestamped
        # self.cmd_vel_subscription = self.create_subscription(TwistStamped, '/cmd_vel_stamped', self.cmd_vel_callback, 30) # timestamped, 30 hz for 60 fps
        # NOTE we don't need the time stamped /cmd_vel for the pico
        self.joy_subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.recording_status_pub = self.create_publisher(String, '/recording_status', 10)

    def set_serial(self):
        try:
            self.close_serial()
            serial_port = self.get_serial_port()
            self.serial = serial.Serial(serial_port, baudrate=115200, timeout=1)
            self.get_logger().info(f"Connected to Pico on {serial_port}")
            return 1
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect to Pico on {serial_port}: {e}")
            return 0
        
    def close_serial(self):
        try:
            self.serial.close()
            return 1
        except:
            return 0

    def get_serial_port(self):
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        if serial_port == 'auto':
            return self.pico_execute.get_pico_port()
        return serial_port

    def cmd_vel_callback(self, msg):
        if self.microcontroller_state:
            # linear_x = msg.twist.linear.x
            # angular_z = msg.twist.angular.z
            linear_x = msg.linear.x
            angular_z = msg.angular.z
            # Convert to PWM duty cycle
            speed_duty_cycle = self.calculate_motor_duty_cycle(linear_x)
            steering_duty_cycle = self.calculate_steering_duty_cycle(angular_z)

            # Send to Pico
            self.send_duty_cycle_to_pico(speed_duty_cycle, steering_duty_cycle)
        else:
            # If emergency stop is active, send neutral duty cycle values
            self.send_duty_cycle_to_pico(self.motor_neutral_duty_cycle, self.steering_neutral_duty_cycle)

    def joy_callback(self, joy_msg):
        emergency_button = joy_msg.buttons[0]
        pause_recording = joy_msg.buttons[1]
        pico_start_button = joy_msg.buttons[9]
        start_recording = joy_msg.buttons[8]

        if emergency_button == 1:
            self.get_logger().info("Emergency buttoned pressed")
            res = self.pico_execute.reset()
            self.close_serial()
            if res == 1:
                self.microcontroller_state = False
        
        elif pause_recording == 1:
            self.handle_pause()

        elif start_recording == 1:
            self.handle_resume()

        elif pico_start_button == 1:
            self.get_logger().info("Pico start button pressed")
            if not self.microcontroller_state:
                res = self.pico_execute.run(self.pico_port_script_path)
                if res == 1:
                    res = self.set_serial()
                    if res == 1:
                        self.microcontroller_state = True

    def handle_resume(self):
        if not self.enable_recording_state and not self.recording_state:
            self.get_logger().info("Resume data collection.")
            self.recording_status_pub.publish(String(data='resume'))
            self.recording_state = True
            self.start_lockout_timer()

    def handle_pause(self):
        if not self.enable_recording_state and self.recording_state:
            self.get_logger().info("Pause data collection.")
            self.recording_status_pub.publish(String(data='pause'))
            self.recording_state = False
            self.start_lockout_timer()

    def start_lockout_timer(self):
        """Start a lockout timer to prevent further actions for 10 seconds."""
        self.enable_recording_state = True
        self.get_logger().info("Lockout enabled for 10 seconds.")

        # Create a timer
        self.create_timer(10.0, self.unlock_buttons)

    def unlock_buttons(self):
        """Unlock buttons after the lockout period."""
        self.enable_recording_state = False
        self.get_logger().info("Lockout period ended. Buttons re-enabled.")

    def calculate_motor_duty_cycle(self, value):
        if value > 0:
            return int(self.motor_neutral_duty_cycle + (value * (self.motor_max_duty_cycle - self.motor_neutral_duty_cycle)))
        elif value < 0:
            return int(self.motor_neutral_duty_cycle + (value * (self.motor_neutral_duty_cycle - self.motor_min_duty_cycle)))
        else:
            return self.motor_neutral_duty_cycle

    def calculate_steering_duty_cycle(self, value):
        if value > 0:
            return int(self.steering_neutral_duty_cycle + (value * (self.steering_max_duty_cycle - self.steering_neutral_duty_cycle)))
        elif value < 0:
            return int(self.steering_neutral_duty_cycle + (value * (self.steering_neutral_duty_cycle - self.steering_min_duty_cycle)))
        else:
            return self.steering_neutral_duty_cycle

    def send_duty_cycle_to_pico(self, speed_duty_cycle, steering_duty_cycle):
        command = f"SPEED:{speed_duty_cycle};STEER:{steering_duty_cycle}\n"
        if self.microcontroller_state:
            self.serial.write(command.encode('utf-8'))
            self.get_logger().info(f"Sent: {command}")
        else:
            self.get_logger().info(f"Pico is not alive!")

def main(args=None):
    import traceback
    rclpy.init(args=args)
    node = None
    try:
        node = RemoteControlHandler()
        rclpy.spin(node)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
