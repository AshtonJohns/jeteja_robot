from pico_handler import PicoConnection

pico = PicoConnection('src/robot_launch/scripts/main_log.py')

while True:

    button = input('Press WASD: ')

    res = "NONE"

    if button == 'W':
        res = pico.connect()
    elif button == 'A':
        res = pico.close()
    elif button == 'S':
        res = pico.get_state()
    elif button == 'D':
        command = f"SPEED:{941};STEER:{941}\n"
        pico.write(command)

    print(res)