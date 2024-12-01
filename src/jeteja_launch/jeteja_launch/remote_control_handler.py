import os
import traceback
import rclpy
import scripts.lower_control as lower_control
from std_msgs.msg import String
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Joy
from ament_index_python.packages import get_package_share_directory
from scripts.pico_handler import PicoConnection

class RemoteControlHandler(Node):
    def __init__(self):
        super().__init__('remote_control_handler')

        # Get main.py path for pico 
        pico_port_script_path = os.path.join(
                get_package_share_directory('jeteja_launch'),
                'scripts',
                'main.py'
            )
        
        # Pico and serial communication process execution
        self.get_logger().info(pico_port_script_path)
        self.pico_execute = PicoConnection(pico_port_script_path)

        # State variables (True == alive, False == dead)
        self.recording_state = False
        self.enable_recording_state = False
        self.pico_enable_state = False

        # Timers to delete
        self.pico_timer = None
        self.recording_timer = None

        # Subscribers and publishers
        self.cmd_vel_subscription = self.create_subscription(Twist, '/cmd_vel_fixed_rate', self.cmd_vel_callback, 15) # not timestamped, but a steady publish rate
        # self.cmd_vel_subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 15) # not timestamped
        # self.cmd_vel_subscription = self.create_subscription(TwistStamped, '/cmd_vel_stamped', self.cmd_vel_callback, 30) # timestamped, 30 hz for 60 fps
        # NOTE we don't need the time stamped /cmd_vel for the pico
        self.joy_subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.recording_status_pub = self.create_publisher(String, '/recording_status', 10)
        

    def cmd_vel_callback(self, msg):
        if self.pico_execute.get_state():
            # linear_x = msg.twist.linear.x
            # angular_z = msg.twist.angular.z
            linear_x = msg.linear.x
            angular_z = msg.angular.z
            # Convert to PWM duty cycle
            speed_duty_cycle = lower_control.calculate_motor_duty_cycle(linear_x)
            steering_duty_cycle = lower_control.calculate_steering_duty_cycle(angular_z)
            # Send to Pico
            self.send_duty_cycle_to_pico(speed_duty_cycle, steering_duty_cycle)
        else:
            self.send_duty_cycle_to_pico(lower_control.MOTOR_NEUTRAL_DUTY_CYCLE, lower_control.STEERING_NEUTRAL_DUTY_CYCLE)

    def joy_callback(self, joy_msg):
        emergency_button = joy_msg.buttons[0]
        pause_recording = joy_msg.buttons[1]
        pico_start_button = joy_msg.buttons[9]
        start_recording = joy_msg.buttons[8]

        if emergency_button == 1:
            self.get_logger().info("Emergency buttoned pressed")
            self.pico_execute.close()
        
        elif pause_recording == 1:
            self.get_logger().info("Pause button pressed")
            self.handle_pause()

        elif start_recording == 1:
            self.get_logger().info('Recording button pressed')
            self.handle_resume()

        elif pico_start_button == 1:
            self.pico_enable_state = False
            self.get_logger().info("Pico start button pressed")
            self.start_lockout_timer(5, pico=True)
            self.pico_execute.connect()


    def handle_resume(self):
        if not self.enable_recording_state and not self.recording_state:
            self.get_logger().info("Resume data collection.")
            self.recording_status_pub.publish(String(data='resume'))
            self.recording_state = True
            self.start_lockout_timer(seconds=10,recording=True)

    def handle_pause(self):
        if not self.enable_recording_state and self.recording_state:
            self.get_logger().info("Pause data collection.")
            self.recording_status_pub.publish(String(data='pause'))
            self.recording_state = False
            self.start_lockout_timer(seconds=10,recording=True)

    def start_lockout_timer(self,seconds,**kwargs):
        """Start a lockout timer to prevent further actions for seconds.
        
        :param seconds: number of seconds for timer

        :param kwargs:
        - recording: recording timer (boolean)
        - pico: pico timer for sending messages (boolean)
        """
        recording = kwargs.get('recording',False)
        pico = kwargs.get('pico', False)
        self.enable_recording_state = True
        if recording:
            self.get_logger().info(f"Recording lockout enabled for {seconds} seconds.")
            if self.recording_timer is None: 
                timer = self.create_timer(seconds, self.unlock_buttons) # Create a timer
                self.recording_timer = timer
        elif pico:
            self.get_logger().info(f"Pico state lockout enable for {seconds} seconds")
            if self.pico_timer is None: 
                timer = self.create_timer(seconds, self.unlock_pico)
                self.pico_timer = timer

    def unlock_buttons(self):
        """Unlock buttons after the lockout period."""
        self.enable_recording_state = False
        self.get_logger().info("Lockout period ended for recording.")
        self.destroy_timer(self.recording_timer)
        self.recording_timer = None

    def unlock_pico(self):
        """
        Allow pico to start after lockout period.
        """
        self.pico_enable_state = True
        self.get_logger().info("Lockout period ended for pico.")
        self.destroy_timer(self.pico_timer)
        self.pico_timer = None
        
    def send_duty_cycle_to_pico(self, speed_duty_cycle, steering_duty_cycle):
        command = lower_control.create_command_message(speed_duty_cycle, steering_duty_cycle)
        if self.pico_execute.get_state():
            if self.pico_enable_state:
                self.pico_execute.write(command)
                self.get_logger().info(f"Sent: {command}")
            # else:
            #     self.get_logger().info(f"Pico is not alive!")

def main(args=None):
    
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
