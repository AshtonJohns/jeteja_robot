import subprocess

def delete_all_files_on_pico():
    # Command to delete all files from the Pico root directory
    delete_command = """
import os
for file in os.listdir():
    os.remove(file)
"""

    try:
        result = subprocess.run(
            ["python3", "-m", "mpremote", "exec", delete_command],
            capture_output=True,
            text=True,
            check=True
        )
        print("All files deleted successfully")
    except subprocess.CalledProcessError as e:
        print("Error deleting files on Pico")

if __name__ == "__main__":
    delete_all_files_on_pico()
