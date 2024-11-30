import subprocess
from os.path import dirname

pico_main_path = f"{dirname(__file__)}/main.py"

def run(path):
    res = subprocess.run( # run main.py
            ["python", "-m", "mpremote", "run", path],
            check=True,
            capture_output=True,
            text=True,
        )
    return res

if __name__ == '__main__':
    run(pico_main_path)