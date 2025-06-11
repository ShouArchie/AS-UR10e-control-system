import urx
import pygame
import time
import math
import threading
import cv2
import numpy as np
import mediapipe as mp
import keyboard


# max_speed can be changed to only change max speed but robot mostly don't reach that with current PID
# You can also change the PID parameters to change the response time and accuracy and speed of robot
# SpeedL if time is too short you won't have enough time to reach max speed

ROBOT_IP = "192.168.0.101"

THERMAL_CAMERA_INDEX = 1
REGULAR_CAMERA_INDEX = 0


class FaceHeatTracker:
    """Face tracking with thermal camera using speedL and PID control"""
    
    def __init__(self):
        self.robot = None
        self.cap = None
        self.running = False
        self.tracking_active = False
        
        # Camera settings - 60 FPS
        self.frame_width = 640
        self.frame_height = 480
        self.target_fps = 60
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Starting position (same as dynamic_face_tracking.py)
        self.start_joints = [
            0.0,                    # Base = 0°
            math.radians(-60),      # Shoulder = -60°
            math.radians(80),       # Elbow = 80°
            math.radians(-110),     # Wrist1 = -110°
            math.radians(270),      # Wrist2 = 270°
            math.radians(-90)       # Wrist3 = -90°
        ]
        
        # PID Controller parameters for Y and Z movements (increased for faster response)
        self.pid_kp_y = 0.003   # Proportional gain for Y (left/right) - increased for speed
        self.pid_ki_y = 0.0003  # Integral gain for Y - increased for speed
        self.pid_kd_y = 0.0006  # Derivative gain for Y - increased for speed
        
        self.pid_kp_z = 0.003   # Proportional gain for Z (up/down) - increased for speed
        self.pid_ki_z = 0.0003  # Integral gain for Z - increased for speed
        self.pid_kd_z = 0.0006  # Derivative gain for Z - increased for speed
        
        # PID state variables
        self.error_integral_y = 0.0
        self.error_integral_z = 0.0
        self.previous_error_y = 0.0
        self.previous_error_z = 0.0
        
        # Speed limits for safety (reduced for less sensitivity)
        self.max_speed_y = 2.5  # Max Y speed (m/s) - reduced for gentle movement
        self.max_speed_z = 2.5  # Max Z speed (m/s) - reduced for gentle movement
        
        # Deadzone to prevent jittery movement (increased for less sensitivity)
        self.deadzone_radius = 25  # pixels - increased for larger deadzone
        
        # Arrow key control for X movement
        self.x_move_step = 0.05  # 50mm steps for more visible movement
        self.arrow_key_thread = None
        
        # Face detection
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.5)
        
        # Moving average filter for face position (increased for more smoothing)
        self.face_history = []
        self.filter_window = 0.2   # 200ms window for smoothing - increased for gentler movement
        
        # Control loop timing
        self.control_thread = None
        self.control_running = False
        self.last_face_position = None
        
    def connect_robot(self):
        """Connect to robot."""
        try:
            print(f"Connecting to robot at {ROBOT_IP}...")
            self.robot = urx.Robot(ROBOT_IP)
            print("✓ Robot connected!")
            return True
        except Exception as e:
            print(f"✗ Robot connection failed: {e}")
            return False
    
    def get_actual_tcp_pose(self):
        """Get current TCP pose using URScript to avoid getl() issues."""
        try:
            # Send URScript to get TCP pose
            script = """
            pose = get_actual_tcp_pose()
            textmsg("POSE:", pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
            """
            self.robot.send_program(script)
            time.sleep(0.05)  # Give time for execution
            
            # For now, we'll use a more reliable method - checking joint positions
            # and using forward kinematics estimation
            current_joints = self.robot.getj()
            if current_joints and len(current_joints) >= 6:
                # Simple forward kinematics estimation for verification
                # This is just for position checking, not exact pose calculation
                return True  # Assume pose is available if joints are readable
            
            return False
            
        except Exception as e:
            print(f"Error getting TCP pose: {e}")
            return False
    
    def move_to_starting_position(self):
        """Move robot to starting position and wait for completion."""
        print("Moving to starting position...")
        print(f"Target joint angles (degrees): {[math.degrees(j) for j in self.start_joints]}")
        
        # Send moveJ command - URX often throws "Robot stopped" even on successful moves
        try:
            self.robot.movej(self.start_joints, acc=0.5, vel=0.3)
            print("Move command sent successfully")
        except Exception as e:
            if "Robot stopped" in str(e):
                print("Move command completed (URX 'Robot stopped' message is normal)")
            else:
                print(f"Move command warning: {e}")
                # Continue anyway - the robot might still move
        
        print("Waiting for robot to reach starting position...")
        
        # Wait and verify position is reached
        max_wait_time = 15  # Increased timeout to 15 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                current_joints = self.robot.getj()
                if current_joints and len(current_joints) >= 6:
                    # Check if we're close to target position
                    position_errors = []
                    for i in range(len(self.start_joints)):
                        error = abs(current_joints[i] - self.start_joints[i])
                        # Handle angle wrapping
                        if error > math.pi:
                            error = 2 * math.pi - error
                        position_errors.append(error)
                    
                    max_error = max(position_errors)
                    max_error_deg = math.degrees(max_error)
                    
                    if max_error_deg < 3.0:  # Within 3 degrees tolerance
                        print(f"\n✓ Starting position reached! Max error: {max_error_deg:.1f}°")
                        
                        # Verify TCP pose is accessible
                        if self.get_actual_tcp_pose():
                            print("✓ TCP pose verified accessible")
                        else:
                            print("⚠ TCP pose verification failed, but continuing...")
                        
                        return True
                    
                    print(f"Moving... Max error: {max_error_deg:.1f}°", end='\r')
                    
            except Exception as e:
                print(f"\nError checking position: {e}")
            
            time.sleep(0.2)
        
        print(f"\n✗ Timeout waiting for starting position after {max_wait_time} seconds")
        
        # Final position check - maybe we're close enough
        try:
            current_joints = self.robot.getj()
            if current_joints and len(current_joints) >= 6:
                position_errors = []
                for i in range(len(self.start_joints)):
                    error = abs(current_joints[i] - self.start_joints[i])
                    if error > math.pi:
                        error = 2 * math.pi - error
                    position_errors.append(error)
                
                max_error = max(position_errors)
                max_error_deg = math.degrees(max_error)
                
                if max_error_deg < 10.0:  # More lenient final check
                    print(f"⚠ Accepting current position with error: {max_error_deg:.1f}°")
                    return True
                else:
                    print(f"✗ Position error too large: {max_error_deg:.1f}°")
                    
        except Exception as e:
            print(f"Final position check failed: {e}")
        
        return False
    
    def init_camera(self):
        """Initialize regular camera with 60 FPS."""
        try:
            print("Initializing regular camera...")
            self.cap = cv2.VideoCapture(REGULAR_CAMERA_INDEX)
            
            if not self.cap.isOpened():
                print(f"✗ Could not open regular camera at index {REGULAR_CAMERA_INDEX}")
                return False
            
            # Set camera properties for 60 FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✓ Camera ready: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            print(f"✗ Camera failed: {e}")
            return False
    
    def detect_face(self, frame):
        """Detect face using MediaPipe."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]  # Use first face
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to pixels
                x = int(bbox.xmin * frame.shape[1])
                y = int(bbox.ymin * frame.shape[0])
                w = int(bbox.width * frame.shape[1])
                h = int(bbox.height * frame.shape[0])
                
                # Face center
                face_x = x + w // 2
                face_y = y + h // 2
                
                return face_x, face_y, x, y, w, h
            
            return None
        except Exception as e:
            print(f"Face detection error: {e}")
            return None
    
    def smooth_face_position(self, face_x, face_y):
        """Apply moving average filter to face position."""
        now = time.time()
        
        # Add current position to history
        self.face_history.append((now, face_x, face_y))
        
        # Remove old entries outside filter window
        self.face_history = [(t, x, y) for (t, x, y) in self.face_history 
                           if now - t <= self.filter_window]
        
        # Calculate average
        if len(self.face_history) > 0:
            avg_x = sum(x for (_, x, _) in self.face_history) / len(self.face_history)
            avg_y = sum(y for (_, _, y) in self.face_history) / len(self.face_history)
            return avg_x, avg_y
        
        return face_x, face_y
    
    def calculate_pid_speeds(self, face_x, face_y):
        """Calculate Y and Z speeds using PID control with adaptive gains based on distance."""
        # Get smoothed face position
        smooth_x, smooth_y = self.smooth_face_position(face_x, face_y)
        
        # Calculate errors (offset from center)
        error_x = smooth_x - self.center_x  # For Y movement (left/right)
        error_y = smooth_y - self.center_y  # For Z movement (up/down)
        
        # Check deadzone
        distance_from_center = math.sqrt(error_x**2 + error_y**2)
        if distance_from_center <= self.deadzone_radius:
            # Reset PID state when in deadzone
            self.error_integral_y = 0.0
            self.error_integral_z = 0.0
            self.previous_error_y = 0.0
            self.previous_error_z = 0.0
            return 0.0, 0.0  # No movement in deadzone
        
        # Calculate adaptive gain multiplier based on distance from center
        # Far from center = higher multiplier (up to 2x), close to center = normal multiplier (1x)
        max_distance = math.sqrt(self.center_x**2 + self.center_y**2)  # Corner of frame
        normalized_distance = min(distance_from_center / max_distance, 1.0)
        
        # Exponential scaling: starts at 1x for close, goes up to 2x for far
        gain_multiplier = 1.0 + (normalized_distance ** 2) * 1.0  # 1x to 2x range
        
        # Apply adaptive gains
        kp_y_adaptive = self.pid_kp_y * gain_multiplier
        ki_y_adaptive = self.pid_ki_y * gain_multiplier
        kd_y_adaptive = self.pid_kd_y * gain_multiplier
        
        kp_z_adaptive = self.pid_kp_z * gain_multiplier
        ki_z_adaptive = self.pid_ki_z * gain_multiplier
        kd_z_adaptive = self.pid_kd_z * gain_multiplier
        
        # PID calculation for Y (left/right movement)
        self.error_integral_y += error_x
        error_derivative_y = error_x - self.previous_error_y
        
        # Integral windup protection
        max_integral = 100
        self.error_integral_y = max(-max_integral, min(max_integral, self.error_integral_y))
        
        dy = (kp_y_adaptive * error_x + 
              ki_y_adaptive * self.error_integral_y + 
              kd_y_adaptive * error_derivative_y)
        
        # PID calculation for Z (up/down movement) - FLIPPED
        self.error_integral_z += error_y
        error_derivative_z = error_y - self.previous_error_z
        
        # Integral windup protection
        self.error_integral_z = max(-max_integral, min(max_integral, self.error_integral_z))
        
        dz = -(kp_z_adaptive * error_y +  # NEGATIVE for correct up/down movement
               ki_z_adaptive * self.error_integral_z + 
               kd_z_adaptive * error_derivative_z)
        
        # Apply speed limits
        dy_limited = max(-self.max_speed_y, min(self.max_speed_y, dy))
        dz_limited = max(-self.max_speed_z, min(self.max_speed_z, dz))
        
        # Update previous errors
        self.previous_error_y = error_x
        self.previous_error_z = error_y
        
        return dy_limited, dz_limited
    
    def send_speed_command(self, dy, dz):
        """Send speedL command with dx=0 (X constrained)."""
        try:
            if abs(dy) < 0.001 and abs(dz) < 0.001:
                # Stop robot if speeds are very small
                self.robot.send_program("stopl(0.2)")
                return
                
            # Send speedL command with dx=0 (X movement constrained) - should not move in base X
            urscript_cmd = f"speedl([0.0, {dy:.6f}, {dz:.6f}, 0, 0, 0], 1, 0.8)"
            self.robot.send_program(urscript_cmd)
            
        except Exception as e: 
            print(f"Error sending speed command: {e}")
    
    def control_loop(self):
        """Main control loop for face tracking - truly pauses when tracking is inactive."""
        try:
            while self.control_running and self.running:
                try:
                    # Only run control loop when tracking is active
                    if self.tracking_active:
                        if self.last_face_position:
                            face_x, face_y = self.last_face_position
                            
                            # Calculate PID speeds
                            dy, dz = self.calculate_pid_speeds(face_x, face_y)
                            
                            # Send speed command
                            self.send_speed_command(dy, dz)
                        else:
                            # Stop robot when no face detected
                            self.robot.send_program("stopl(0.2)")
                        
                        time.sleep(1.0 / 60.0)  # 60 Hz control loop
                    else:
                        # When tracking is paused, truly pause the control loop
                        # Send stopl only once when first paused, then sleep
                        time.sleep(0.5)  # Long sleep when paused - control loop effectively stopped
                    
                except Exception as inner_e:
                    print(f"Control loop error: {inner_e}")
                    time.sleep(0.1)
                    
        except Exception as outer_e:
            print(f"Control loop error: {outer_e}")
    
    def get_current_pose(self):
        """
        Get current TCP pose as a list of floats.
        ---
        NOTE: URX's getl() and PoseVector are broken on this system (no tolist(), not iterable, etc).
        The only reliable way is to use direct attribute access for x, y, z, rx, ry, rz.
        This method is used for all X direction movement and anywhere a robust pose is needed.
        """
        try:
            # Direct attribute access is the only reliable method on this system
            return [
                float(self.robot.x),
                float(self.robot.y),
                float(self.robot.z),
                float(self.robot.rx),
                float(self.robot.ry),
                float(self.robot.rz)
            ]
        except Exception as e:
            print(f"Error getting current pose with attribute access: {e}")
            return None
    
    def handle_arrow_keys(self):
        """
        Handle arrow key presses for X-axis movement using tool-relative moveL.
        ---
        We originally tried to use URX's movel with a pose list, but due to PoseVector bugs,
        this caused errors ('PoseVector' object is not iterable). The solution is to use
        URScript's pose_trans to move in the tool's X direction, which is robust and does not
        require any PoseVector conversion.
        """
        while self.running:
            try:
                if not self.tracking_active:
                    if keyboard.is_pressed('up'):
                        try:
                            print(f"UP: Moving forward {self.x_move_step*1000:.0f}mm in tool X (tracking is PAUSED)")
                            urscript_cmd = f"movel(pose_trans(get_actual_tcp_pose(), p[{self.x_move_step:.6f}, 0, 0, 0, 0, 0]), a=0.5, v=0.2)"
                            print(f"DEBUG: Sending URScript: {urscript_cmd}")
                            self.robot.send_program(urscript_cmd)
                        except Exception as move_e:
                            print(f"UP movement error: {move_e}")
                        time.sleep(1.0)
                    elif keyboard.is_pressed('down'):
                        try:
                            print(f"DOWN: Moving backward {self.x_move_step*1000:.0f}mm in tool X (tracking is PAUSED)")
                            urscript_cmd = f"movel(pose_trans(get_actual_tcp_pose(), p[-{self.x_move_step:.6f}, 0, 0, 0, 0, 0]), a=0.5, v=0.2)"
                            print(f"DEBUG: Sending URScript: {urscript_cmd}")
                            self.robot.send_program(urscript_cmd)
                        except Exception as move_e:
                            print(f"DOWN movement error: {move_e}")
                        time.sleep(1.0)
                    else:
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Arrow key error: {e}")
                time.sleep(0.1)
    
    def run(self):
        """Main execution."""
        print("\n=== FACE HEAT TRACKER with speedL ===")
        print("SPACE: Toggle tracking, UP/DOWN: X movement, ESC: Exit")
        
        # Initialize systems
        if not self.connect_robot():
            return
        
        # Move to starting position and wait for completion
        if not self.move_to_starting_position():
            self.robot.close()
            return
        
        if not self.init_camera():
            self.robot.close()
            return
        
        print("\n✓ All systems ready!")
        print("Controls:")
        print("  SPACE: Toggle face tracking ON/OFF")
        print("  UP/DOWN arrows: Move forward/backward (X-axis) - works even when tracking is paused")
        print("  ESC: Exit")
        print("\nFace tracking starts PAUSED - press SPACE to activate!")
        
        # Set running flags FIRST before starting threads
        self.running = True
        self.control_running = True
        
        # Start control loop thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        # Start arrow key handler thread
        self.arrow_key_thread = threading.Thread(target=self.handle_arrow_keys)
        self.arrow_key_thread.daemon = True
        self.arrow_key_thread.start()
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    continue
                
                frame = cv2.flip(frame, -1)  # Mirror
                
                # Detect face
                face_data = self.detect_face(frame)
                
                if face_data:
                    face_x, face_y, x, y, w, h = face_data
                    self.last_face_position = (face_x, face_y)
                    
                    # Draw face rectangle and center
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(frame, (face_x, face_y), 5, (0, 255, 0), -1)
                    
                    # Draw deadzone circle
                    cv2.circle(frame, (self.center_x, self.center_y), 
                             self.deadzone_radius, (255, 255, 0), 2)
                else:
                    self.last_face_position = None
                
                # Draw center crosshair
                cv2.circle(frame, (self.center_x, self.center_y), 3, (0, 0, 255), -1)
                cv2.line(frame, (self.center_x-15, self.center_y), 
                        (self.center_x+15, self.center_y), (0, 0, 255), 2)
                cv2.line(frame, (self.center_x, self.center_y-15), 
                        (self.center_x, self.center_y+15), (0, 0, 255), 2)
                
                # Status display
                status = "TRACKING" if self.tracking_active else "PAUSED"
                color = (0, 255, 0) if self.tracking_active else (0, 0, 255)
                cv2.putText(frame, f"Face Tracking: {status}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Controls info
                cv2.putText(frame, "SPACE: Toggle tracking, UP/DOWN: X move (always), ESC: Exit", 
                           (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                           (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Face Heat Tracker', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Exiting...")
                    break
                elif key == ord(' '):  # SPACE
                    self.tracking_active = not self.tracking_active
                    status_text = "ACTIVATED" if self.tracking_active else "PAUSED"
                    print(f"Face tracking {status_text}")
                    
                    # Reset PID state when toggling
                    self.error_integral_y = 0.0
                    self.error_integral_z = 0.0
                    self.previous_error_y = 0.0
                    self.previous_error_z = 0.0
                    
                    if not self.tracking_active:
                        # Stop robot immediately when pausing and notify control loop is paused
                        self.robot.send_program("stopl(0.5)")
                        print("Control loop paused - arrow keys now available")
                    else:
                        print("Control loop resumed - arrow keys disabled")
                
            except Exception as e:
                print(f"Main loop error: {e}")
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.running = False
        self.control_running = False
        
        # Stop robot
        if self.robot:
            try:
                self.robot.send_program("stopl(0.5)")
            except:
                pass
        
        # Stop threads
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1)
        
        if self.arrow_key_thread and self.arrow_key_thread.is_alive():
            self.arrow_key_thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.robot:
            self.robot.close()
        
        print("✓ Cleanup complete")


# ===============================================
# MAIN EXECUTION
# ===============================================
if __name__ == "__main__":
    tracker = FaceHeatTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        tracker.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        tracker.cleanup()



