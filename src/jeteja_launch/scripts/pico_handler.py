import subprocess
import serial

class PicoConnection(object):
    def __init__(self, path) -> None:
        self.picohandler = PicoHandler()
        self.serialhandler = SerialConnectionHandler()
        self.mainpy_path = path
        self.state_alive = False

    def connect(self):
        if not self.get_state():
            port = self.picohandler.get_pico_port()
            self.serialhandler.set_serial(port, 115200, 1)
            self.picohandler.run(self.mainpy_path)
            self.set_state_alive()

    def write(self, command):
        if self.get_state():
            self.serialhandler.write(command)

    def close(self):
        if self.get_state():
            self.picohandler.reset()
            self.serialhandler.close_serial()
            self.set_state_dead()

    def get_state(self):
        return self.state_alive
    
    def set_state_alive(self):
        self.state_alive = True

    def set_state_dead(self):
        self.state_alive = False


class SerialConnectionHandler(object):
    def __init__(self) -> None:
        self.serial = serial.Serial()

    def set_serial(self, serial_port, baudrate, timeout):
        self.serial.port = serial_port
        self.port = serial_port
        self.serial.baudrate = baudrate
        self.baudrate = baudrate
        self.serial.timeout = timeout
        self.timeout = timeout
        self.serial.open()

    def write(self, command:str):
        self.serial.write(command.encode('utf-8'))

    def close_serial(self):
        self.serial.close()

class PicoHandler(object):
    def __init__(self):
        pass

    def kill(self):
        self.terminate()

    def reset(self):
        self.kill()
        command = ["python", "-m","mpremote", "reset"]
        res = subprocess.Popen(command)
            
    def run(self,path):
        command = ["python", "-m", "mpremote", "run", path]
        res = subprocess.Popen(command)
        self.terminate = res.terminate
        
    def get_pico_port(self):
        # Run `mpremote connect list` and capture the output
        try:
            result = subprocess.run(['python', '-m','mpremote', 'devs'], capture_output=True, text=True)
            output = result.stdout

            # Parse the output to find the serial port (assuming the first line has the required port)
            lines = output.splitlines()
            for line in lines:
                if 'MicroPython' in line:
                    return line.split()[0]  # Extract the port (e.g., /dev/ttyACM0)
        except:
            raise serial.PortNotOpenError()