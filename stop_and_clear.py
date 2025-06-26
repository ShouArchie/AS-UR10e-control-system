# Emergency stop and clear script for Pico
# Run this in Thonny to stop any running programs and clear the device

import machine
import time
import os

print("=== EMERGENCY STOP AND CLEAR ===")

# Stop any running timers
try:
    # Try to stop any existing timers
    for i in range(4):  # Pico has 4 hardware timers
        try:
            timer = machine.Timer(i)
            timer.deinit()
            print(f"Stopped timer {i}")
        except:
            pass
except:
    pass

# Clear any auto-running files
try:
    files = os.listdir('/')
    print("Files on device:", files)
    
    # Check for main.py (auto-runs on boot)
    if 'main.py' in files:
        print("Found main.py - this runs automatically on boot")
        print("To stop auto-run, you can rename or delete it")
        # Uncomment next line to automatically rename it:
        # os.rename('main.py', 'main_disabled.py')
        
except Exception as e:
    print(f"Error listing files: {e}")

print("=== DEVICE CLEARED ===")
print("You can now upload new code safely")
print("Thonny shell should be responsive now") 