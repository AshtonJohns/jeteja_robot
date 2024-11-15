import subprocess

def reset():
    command = ["python", "-m", "mpremote", "reset"]
    res = subprocess.run(command,
                         check=True,
                        capture_output=True,
                        text=True,)
    
if __name__ == '__main__':
    reset()