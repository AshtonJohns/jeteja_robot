import traceback
import rclpy
import scripts.lower_control as lower_control
import config.master_config as master_config
from std_msgs.msg import String
from rclpy.node import Node
from sensor_msgs.msg import Joy
from scripts.pico_handler import PicoConnection
from jeteja_launch_msgs.msg import PwmSignals

class RemoteControlHandler(Node):
    def __init__(self):
        super().__init__('remote_control_handler')

        # Run 'manual' or 'autopilot' mode
        self.declare_parameter('manual', False)
        self.declare_parameter('autopilot', False)

        manual_mode = self.get_parameter('manual').value
        autopilot_mode = self.get_parameter('autopilot').value

        self.get_logger().info(f"Manual mode: {manual_mode}")
        self.get_logger().info(f"Autopilot mode: {autopilot_mode}")

        self.manual_mode = False
        self.autopilot_mode = False
        self.mode_discovery(manual_mode,autopilot_mode)

        # Master configs
        self.button_mapping = master_config.JOY_CONTROLLER_CONFIG_MAP
        
        # Pico and serial communication process execution
        self.pico_execute = PicoConnection()

        # State variables (True == alive, False == dead)
        self.pico_enable_state = False

        # Timers to delete
        self.pico_timer = None
        self.recording_timer = None

        # Subscribers and publishers
        # self.cmd_vel_subscription = self.create_subscription(Twist, '/cmd_vel_fixed_rate', self.cmd_vel_callback, 15) # not timestamped, but a steady publish rate
        # self.cmd_vel_subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 15) # not timestamped
        # self.cmd_vel_subscription = self.create_subscription(TwistStamped, '/cmd_vel_stamped', self.cmd_vel_callback, 30) # timestamped, 30 hz for 60 fps
        # NOTE we don't need the time stamped /cmd_vel for the pico
        self.pwm_signals_subscription = self.create_subscription(PwmSignals, '/pwm_signals', self.send_duty_cycle_to_pico, 15)
        self.joy_subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.recording_status_pub = self.create_publisher(String, '/recording_status', 10)

    def mode_discovery(self, manual_mode, autopilot_mode):
        if manual_mode and autopilot_mode:
            raise Exception("You cannot run in both manual and autopilot modes.")
        elif manual_mode:
            self.manual_mode = True
        elif autopilot_mode:
            self.autopilot_mode = True
        else:
            raise Exception("You must run either manual or autopilot modes.")

    def get_button_names(self, joy_msg:Joy):
        btn_names = []
        for btn_name, btn_idx in self.button_mapping.items():
            if joy_msg.buttons[btn_idx]:
                btn_names.append(btn_name)
        return btn_names

    def joy_callback(self, joy_msg):
        enabled_btns = self.get_button_names(joy_msg)

        # self.get_logger().info(" ".join(enabled_btns))

        if "emergency_btn" in enabled_btns:
            self.get_logger().info("Emergency buttoned pressed")
            self.pico_execute.close()
        
        elif "pause_recording_btn" in enabled_btns:
            self.get_logger().info("Pause button pressed")
            self.handle_pause()

        elif "start_recording_btn" in enabled_btns:
            self.get_logger().info('Recording button pressed')
            self.handle_resume()

        elif "pico_start_btn" in enabled_btns:
            self.pico_enable_state = False
            self.get_logger().info("Pico start button pressed")
            self.start_lockout_timer(5, pico=True)
            self.pico_execute.connect()


    def handle_resume(self):
        self.get_logger().info("Resume data collection.")
        self.recording_status_pub.publish(String(data='resume'))

    def handle_pause(self):
        self.get_logger().info("Pause data collection.")
        self.recording_status_pub.publish(String(data='pause'))

    def start_lockout_timer(self,seconds,**kwargs):
        """Start a lockout timer to prevent further actions for seconds.
        
        :param seconds: number of seconds for timer

        :param kwargs:
        - recording: recording timer (boolean)
        - pico: pico timer for sending messages (boolean)
        """
        recording = kwargs.get('recording',False)
        pico = kwargs.get('pico', False)
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
        self.enable_recording_state = True
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
        
    def send_duty_cycle_to_pico(self, msg):
        if self.pico_execute.get_state(): # Check if Pico is running and serial connection is established
            motor_pwm = msg.motor_pwm
            steering_pwm = msg.steering_pwm
            command = lower_control.create_command_message(motor_pwm, steering_pwm)
            if self.pico_enable_state: # Has surpassed the lockdown timer and is enabled
                self.pico_execute.write(command)
                # self.get_logger().info(f"Sent: {command}")
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
