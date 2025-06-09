import pygame
import time

# Initialize pygame
pygame.init()
pygame.joystick.init()

print("Simple Space Mouse Test")
print(f"Found {pygame.joystick.get_count()} input devices")

if pygame.joystick.get_count() == 0:
    print("No devices found!")
    exit()

# Use first device
joy = pygame.joystick.Joystick(0)
joy.init()

print(f"Using device: {joy.get_name()}")
print(f"Axes: {joy.get_numaxes()}, Buttons: {joy.get_numbuttons()}")
print("\nMove your Space Mouse (press Ctrl+C to exit):")

try:
    while True:
        pygame.event.pump()
        
        # Read all axes
        values = []
        for i in range(joy.get_numaxes()):
            val = joy.get_axis(i)
            values.append(f"{val:.3f}")
        
        # Only print if any value is non-zero
        if any(abs(float(v)) > 0.001 for v in values):
            print(" | ".join([f"Axis{i}:{v}" for i, v in enumerate(values)]))
        
        time.sleep(0.05)
        
except KeyboardInterrupt:
    print("\nExiting...")
    pygame.quit() 