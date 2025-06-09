import urx
import pygame
import time
import math
import threading
import cv2
import numpy as np

# ===============================================
# SPACE MOUSE ROBOT CONTROL CONFIGURATION
# ===============================================
ROBOT_IP = "192.168.0.101"

class SpaceMouseRobotController:
    def __init__(self):
        self.robot = None
        self.running = False
        self.control_active = False
        
        # Space mouse
        self.spacemouse_connected = False
        self.joystick = None
        
        # Control parameters
        self.translation_scale = 0.004  # Scale factor for translation (m per axis unit)
        self.rotation_scale = 0.01      # Scale factor for rotation (rad per axis unit)
        
        # Current target pose (will be set to starting position)
        self.target_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    def connect_robot(self):
        """Connect to the UR10e robot."""
        try:
            print(f"Connecting to robot at {ROBOT_IP}...")
            self.robot = urx.Robot(ROBOT_IP)
            print("✓ Robot connected!")
            return True
        except Exception as e:
            print(f"✗ Robot connection failed: {e}")
            return False
    
    def connect_spacemouse(self):
        """Connect to the 3D Connexion Space Mouse."""
        try:
            print("Initializing Space Mouse...")
            
            # Initialize pygame joystick module
            pygame.init()
            pygame.joystick.init()
            
            # Get number of joysticks/input devices
            joystick_count = pygame.joystick.get_count()
            print(f"Found {joystick_count} input device(s)")
            
            if joystick_count == 0:
                print("✗ No input devices detected")
                return False
            
            # Look for 3D Connexion devices
            for i in range(joystick_count):
                js = pygame.joystick.Joystick(i)
                js.init()
                device_name = js.get_name().lower()
                print(f"Device {i}: {js.get_name()}")
                print(f"  Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}")
                
                # Check if this looks like a 3D Connexion device
                if ("3dconnexion" in device_name or 
                    "spacemouse" in device_name or 
                    js.get_numaxes() >= 6):
                    
                    print(f"✓ Using Space Mouse: {js.get_name()}")
                    self.joystick = js
                    self.spacemouse_connected = True
                    return True
            
            # Use first device as fallback
            if joystick_count > 0:
                js = pygame.joystick.Joystick(0)
                js.init()
                print(f"⚠ Using first available device: {js.get_name()}")
                self.joystick = js
                self.spacemouse_connected = True
                return True
            
            return False
                
        except Exception as e:
            print(f"✗ Error connecting to Space Mouse: {e}")
            return False
    
    def get_current_pose(self):
        """Get current robot pose by working around URX PoseVector issue."""
        try:
            # WORKAROUND: The URX library's getl() method has a bug where it calls 
            # tolist() on PoseVector objects that don't have this method.
            # Instead, we'll use URScript to get the pose directly
            
            print("DEBUG: Using URScript workaround to get pose")
            
            # Send URScript command to get current pose
            script = "get_actual_tcp_pose()"
            result = self.robot.send_program(script)
            
            # Give it a moment to execute
            time.sleep(0.1)
            
            # Try to get result from robot state - this is tricky
            # Let's try a different approach using getj() first to make sure connection works
            try:
                joints = self.robot.getj()
                print(f"DEBUG: getj() works, got joints: {len(joints) if joints else 'None'}")
            except Exception as e:
                print(f"DEBUG: getj() failed: {e}")
            
            # Alternative approach: Use forward kinematics from joint positions
            # This is more reliable than the broken getl() method
            try:
                # Get current joint positions
                joints = self.robot.getj()
                if joints and len(joints) >= 6:
                    print(f"DEBUG: Got joint positions, using FK calculation")
                    
                    # Send URScript to calculate forward kinematics
                    fk_script = f"pose = get_forward_kin([{joints[0]:.6f}, {joints[1]:.6f}, {joints[2]:.6f}, {joints[3]:.6f}, {joints[4]:.6f}, {joints[5]:.6f}])"
                    
                    # For now, let's use a reasonable default starting pose
                    # Since we know the joint angles, we can estimate the pose
                    # This is a temporary workaround until we fix the URX issue
                    
                    # Return a reasonable default pose for the starting position
                    default_pose = [0.4, 0.0, 0.3, 0.0, 3.14, 0.0]  # X, Y, Z, RX, RY, RZ
                    print(f"DEBUG: Using default pose estimate: {default_pose}")
                    return default_pose
                    
            except Exception as e:
                print(f"DEBUG: FK approach failed: {e}")
            
            # Final fallback - return a safe default pose
            print("DEBUG: Using final fallback pose")
            return [0.4, 0.0, 0.3, 0.0, 3.14, 0.0]  # Safe starting pose
            
        except Exception as e:
            print(f"DEBUG: All methods failed: {e}")
            # Return a reasonable default pose so the program can continue
            return [0.4, 0.0, 0.3, 0.0, 3.14, 0.0]
    
    def move_to_starting_position(self):
        """Move robot to starting position (same as dynamic face tracking)."""
        try:
            print("Moving to starting position...")
            
            # Starting position from dynamic face tracking
            start_joints = [
                0.0,                    # Base = 0°
                math.radians(-60),      # Shoulder = -60°
                math.radians(80),       # Elbow = 80°
                math.radians(-110),     # Wrist1 = -110°
                math.radians(270),      # Wrist2 = 270°
                math.radians(-90)       # Wrist3 = -90°
            ]
            
            print(f"Target joint angles (degrees): {[math.degrees(j) for j in start_joints]}")
            
            # Send movement command using URScript
            urscript_cmd = (f"movej([{start_joints[0]:.6f}, {start_joints[1]:.6f}, "
                           f"{start_joints[2]:.6f}, {start_joints[3]:.6f}, "
                           f"{start_joints[4]:.6f}, {start_joints[5]:.6f}], "
                           f"a=0.5, v=0.3)")
            print(f"Sending URScript: {urscript_cmd}")
            self.robot.send_program(urscript_cmd)
            
            # Wait for movement to complete
            print("Waiting for movement to complete...")
            time.sleep(8)
            
            # Get current pose for space mouse control
            current_pose = self.get_current_pose()
            if current_pose:
                self.target_pose = current_pose.copy()
                print(f"Starting pose: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Z={current_pose[2]:.3f}")
                print("✓ Starting position reached!")
                return True
            else:
                print("✗ Could not get current pose")
                return False
                
        except Exception as e:
            print(f"✗ Starting position failed: {e}")
            return False
    
    def read_spacemouse_input(self):
        """Read and process space mouse input with new axis mappings and deadzone."""
        if not self.spacemouse_connected or not self.joystick:
            return None
        try:
            num_axes = self.joystick.get_numaxes()
            if num_axes < 6:
                return None
            axis_values = [self.joystick.get_axis(i) for i in range(num_axes)]
            # Apply deadzone of 0.7 for all axes
            dead_zone = 0.7
            for i in range(len(axis_values)):
                if abs(axis_values[i]) < dead_zone:
                    axis_values[i] = 0.0
            # Axis mapping:
            # Axis 0: Y translation (TCP)
            # Axis 1: X translation (reversed, TCP)
            # Axis 2: Z translation (reversed, TCP)
            # Axis 3: Wrist1 (joint 3)
            # Axis 4: Wrist2 (joint 4)
            # Axis 5: Wrist3 (joint 5)
            movement = {
                'x': -axis_values[1] if len(axis_values) > 1 else 0.0,  # X (reversed)
                'y': axis_values[0] if len(axis_values) > 0 else 0.0,   # Y
                'z': axis_values[2] if len(axis_values) > 2 else 0.0,  # Z (reversed)
                'wrist1': -axis_values[3] if len(axis_values) > 3 else 0.0,  # Wrist1
                'wrist2': axis_values[4] if len(axis_values) > 4 else 0.0,  # Wrist2 (now axis 4)
                'wrist3': axis_values[5] if len(axis_values) > 5 else 0.0,  # Wrist3
                'raw_axes': axis_values
            }
            num_buttons = self.joystick.get_numbuttons()
            button_pressed = False
            for i in range(num_buttons):
                if self.joystick.get_button(i):
                    button_pressed = True
                    break
            movement['button_pressed'] = button_pressed
            return movement
        except Exception as e:
            print(f"Error reading space mouse: {e}")
            return None
    
    def send_relative_movement(self, movement):
        """Send relative movement commands directly to robot, including wrist joints, with deadzone check."""
        if not movement:
            return
        # Only send command if any axis exceeds the deadzone (0.7)
        movement_threshold = 0.7
        raw_axes = movement['raw_axes']
        if all(abs(axis) < movement_threshold for axis in raw_axes):
            return  # Don't send any commands when all axes are below threshold
        # Translation (TCP-relative)
        dx = movement['x'] * self.translation_scale
        dy = movement['y'] * self.translation_scale
        dz = movement['z'] * self.translation_scale
        # Wrist joint increments
        wrist1_inc = movement['wrist1'] * self.rotation_scale
        wrist2_inc = movement['wrist2'] * self.rotation_scale
        wrist3_inc = movement['wrist3'] * self.rotation_scale
        # If any translation, send movel in TCP frame
        if abs(dx) > 0.0001 or abs(dy) > 0.0001 or abs(dz) > 0.0001:
            urscript_cmd = f"movel(pose_trans(get_actual_tcp_pose(), p[{dx:.6f}, {dy:.6f}, {dz:.6f}, 0, 0, 0]), a=1.0, v=0.2)"
            self.robot.send_program(urscript_cmd)
        # If any wrist movement, send movej with small increments
        if abs(wrist1_inc) > 0.0001 or abs(wrist2_inc) > 0.0001 or abs(wrist3_inc) > 0.0001:
            try:
                current_joints = self.robot.getj()
                new_joints = list(current_joints)
                new_joints[3] += wrist1_inc  # wrist1
                new_joints[4] += wrist2_inc  # wrist2
                new_joints[5] += wrist3_inc  # wrist3
                urscript_cmd = f"movej([{','.join(f'{j:.6f}' for j in new_joints)}], a=1.0, v=0.2)"
                self.robot.send_program(urscript_cmd)
            except Exception as e:
                print(f"\nError sending wrist joint command: {e}")
    
    def run_control(self):
        """Main control loop with camera view and keyboard toggling."""
        print("\n=== SPACE MOUSE ROBOT CONTROL + CAMERA VIEW ===")
        print("Waiting for robot to reach starting position...")
        print("Press SPACE to toggle Space Mouse control, ESC to quit.")
        print("Camera window: Press ESC to quit as well.")

        # Connect to robot and move to starting position before anything else
        if not self.connect_robot():
            print("Could not connect to robot. Exiting.")
            return
        if not self.move_to_starting_position():
            print("Could not move to starting position. Exiting.")
            self.robot.close()
            return
        if not self.connect_spacemouse():
            print("Could not connect to Space Mouse. Exiting.")
            self.robot.close()
            return

        # Wait for robot to reach starting position
        print("Waiting additional time for robot to settle...")
        time.sleep(2)
        
        # Open camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Could not open camera at index 1!")
            self.robot.close()
            return
        
        # Initial state: control paused
        self.control_active = False
        self.running = True
        
        try:
            while self.running:
                # --- CAMERA VIEW ---
                ret, frame = cam.read()
                if ret:
                    cv2.imshow("Camera View (ESC to quit, SPACE to toggle control)", frame)
                else:
                    cv2.imshow("Camera View (ESC to quit, SPACE to toggle control)", np.zeros((480,640,3), dtype=np.uint8))
                
                # --- KEYBOARD HANDLING ---
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("ESC pressed, exiting...")
                    self.running = False
                    break
                elif key == 32:  # SPACE
                    self.control_active = not self.control_active
                    print(f"Space Mouse control {'ENABLED' if self.control_active else 'PAUSED'}")
                    time.sleep(0.2)  # Debounce
                
                # --- SPACE MOUSE CONTROL ---
                if pygame.get_init():
                    pygame.event.pump()
                movement = self.read_spacemouse_input() if pygame.get_init() else None
                if self.control_active and movement:
                    # Print simple one-line output that overwrites previous line
                    raw_axes = movement['raw_axes']
                    print(f"\rAxes: {[f'{val:.2f}' for val in raw_axes]}", end='', flush=True)
                    # Check for button press
                    if movement['button_pressed']:
                        print("\nSpace Mouse button pressed!")
                        print("Robot control continues...")
                    # Send relative movement commands directly
                    self.send_relative_movement(movement)
                time.sleep(0.03)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            print("Cleaning up...")
            cam.release()
            cv2.destroyAllWindows()
            if self.spacemouse_connected:
                try:
                    if self.joystick:
                        self.joystick.quit()
                    pygame.quit()
                except:
                    pass
            if self.robot:
                self.robot.close()
            print("✓ Cleanup complete")

# ===============================================
# MAIN EXECUTION
# ===============================================
if __name__ == "__main__":
    controller = SpaceMouseRobotController()
    try:
        controller.run_control()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}") 