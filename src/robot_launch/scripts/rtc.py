import subprocess
from datetime import datetime

def set_rtc_on_pico():
    # Get the current date and time
    now = datetime.now()
    # Format as (year, month, day, weekday, hour, minute, second, microsecond)
    # Note: `weekday` is zero-based (0=Monday, 6=Sunday)
    rtc_tuple = (
        now.year,
        now.month,
        now.day,
        now.weekday(),
        now.hour,
        now.minute,
        now.second,
        0  # microseconds are set to 0
    )
    
    # Format the tuple for inline execution
    rtc_command = f"import machine; machine.RTC().datetime({rtc_tuple})"
    
    # Run the mpremote command to set the RTC on the Pico
    try:
        result = subprocess.run(
            ["python3", "-m", "mpremote", "exec", rtc_command],
            capture_output=True,
            text=True,
            check=True
        )
        print("RTC set successfully")
    except subprocess.CalledProcessError as e:
        print("Error setting RTC on Pico:", e.stderr)

if __name__ == '__main__':
    set_rtc_on_pico()
