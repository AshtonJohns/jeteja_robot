"""
Upload this script to the pico board, then rename it to main.py.
Read dutycycle in nanoseconds via USB BUS.
"""
import sys
import select
from time import sleep
from machine import Pin, PWM, reset

# SETUP
steering = PWM(Pin(0))

steering.freq(50)
throttle = PWM(Pin(15))
throttle.freq(50)
led = Pin('LED')


# LOOP
try:
    led.toggle()
    sleep(3)  # ESC calibrate
    poller = select.poll()
    poller.register(sys.stdin, select.POLLIN)
    print("I'm listening...")
    event = poller.poll()

    while True:
        # read data from serial
        
        for msg, _ in event:
            buffer = msg.readline().rstrip().split(',')
            # print(buffer) # debug
            # print(len(buffer)) # debug
            if len(buffer) == 2:
                ns_st, ns_th = int(buffer[0]), int(buffer[1])
                if ns_st == ns_th == 'END':
                    break
                print(ns_st, ns_th) # debug
                steering.duty_ns(ns_st)
                throttle.duty_ns(ns_th)

except:
    led.toggle()
finally:
    print('Pico reset')
    reset()