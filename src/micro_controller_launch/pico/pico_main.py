import machine
import utime

# Set up PWM pins for speed and steering control
speed_pwm = machine.PWM(machine.Pin(15))  # Adjust GPIO as needed
steering_pwm = machine.PWM(machine.Pin(14))
speed_pwm.freq(50)
steering_pwm.freq(50)

# LED
led = machine.Pin('LED', machine.Pin.OUT)

# Set up serial communication
uart = machine.UART(0, baudrate=115200, tx=machine.Pin(0), rx=machine.Pin(1))

def process_command(command):
    if command.startswith("SPEED:"):
        speed_value = int(command.split(":")[1])
        speed_pwm.duty_u16(speed_value)
    elif command.startswith("STEER:"):
        steer_value = int(command.split(":")[1])
        steering_pwm.duty_u16(steer_value)

try:
    led.toggle()
    while True:
        if uart.any():
            data = uart.readline().decode('utf-8').strip()
            process_command(data)
        utime.sleep(0.01)
except:
    led.toggle()
    machine.reset()