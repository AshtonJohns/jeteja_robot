import subprocess

def get_pico_port():
    try:
        # Run `mpremote connect list` and capture the output
        result = subprocess.run(['python3', '-m', 'mpremote', 'connect', 'list'], capture_output=True, text=True)
        output = result.stdout

        # Parse the output to find the serial port (assuming the first line has the required port)
        lines = output.splitlines()
        for line in lines:
            if line.startswith('/dev'):
                return line.split()[0]  # Extract the port (e.g., /dev/ttyACM0)
    except Exception as e:
        print(f"Error finding Pico port: {e}")

    # Return None if the port is not found
    return None

if __name__ == "__main__":
    pico_port = get_pico_port()
    if pico_port:
        print(pico_port)
    else:
        print("No Pico device found")
