import machine
import utime
import sys
import select
import utime

# Set up PWM pins for speed and steering control
speed_pwm = machine.PWM(machine.Pin(15))  
steering_pwm = machine.PWM(machine.Pin(0))
speed_pwm.freq(50)
steering_pwm.freq(50)

# LED for indicating status
led = machine.Pin('LED', machine.Pin.OUT)

def set_pwm(pwm, duty_cycle):
    pwm.duty_u16(duty_cycle)

def process_command(command):
    commands = command.split(";")
    for cmd in commands:
        if cmd.startswith("SPEED:"):
            speed_value = int(cmd.split(":")[1])
            set_pwm(speed_pwm, speed_value)  # Use set_pwm for speed
        elif cmd.startswith("STEER:"):
            steer_value = int(cmd.split(":")[1])
            set_pwm(steering_pwm, steer_value)  # Use set_pwm for steering
        
try:
    led.toggle()
    while True:
        # Check for incoming data via sys.stdin
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            data = sys.stdin.readline().strip()
            process_command(data)
        utime.sleep(0.01)
except Exception as e:
    machine.reset()
finally:
    led.toggle()
    # machine.reset()
