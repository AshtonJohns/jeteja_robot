import subprocess
from os import makedirs
from os.path import dirname

export_path = f"{dirname(__file__)}/../logs"

def export_pico_contents(destination_folder=export_path):
    # Ensure the destination folder exists
    makedirs(destination_folder, exist_ok=True)

    # Command to list all files in the Pico's root directory
    list_files_command = """
import os
for file in os.ilistdir():
    print(file)
"""

    try:
        # Run the command to list files
        result = subprocess.run(
            ["python3", "-m", "mpremote", "exec", list_files_command],
            capture_output=True,
            text=True,
            check=True
        )
        file_list = result.stdout.split("\n")

        # Download each file
        for filename in file_list: 
            
            filename = filename[1:filename.find(',')]
            filename = filename.replace("'","")
            
            try:
                subprocess.run(
                    ["python", "-m", "mpremote", "cp", f":{filename}", export_path],
                    check=True,
                    capture_output=True
                )
            except:
                subprocess.run(
                    ["python", "-m", "mpremote", "cp", "-r", f":{filename}", export_path],
                    check=True,
                    capture_output=True
                )

        print(f"All files have been exported to {destination_folder}")
    except subprocess.CalledProcessError as e:
        print("Error exporting files:", e.stderr)

if __name__ == '__main__':
    export_pico_contents(export_path)
