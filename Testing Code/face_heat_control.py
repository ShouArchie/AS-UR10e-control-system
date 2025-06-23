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

THERMAL_CAMERA_INDEX = 0
REGULAR_CAMERA_INDEX = 1


class FaceHeatTracker:
    """Face tracking with thermal camera using speedL and PID control"""
    
    def __init__(self):
        self.robot = None
        self.cap = None
        self.thermal_cap = None
        self.running = False
        self.face_tracking_active = False
        self.thermal_tracking_active = False
        
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
        
        # PID Controller parameters for Y and Z movements (reduced for less overshoot)
        self.pid_kp_y = 0.002   # Proportional gain for Y (left/right) - reduced for less overshoot
        self.pid_ki_y = 0.0003  # Integral gain for Y
        self.pid_kd_y = 0.0006  # Derivative gain for Y
        
        self.pid_kp_z = 0.002   # Proportional gain for Z (up/down) - reduced for less overshoot
        self.pid_ki_z = 0.0003  # Integral gain for Z
        self.pid_kd_z = 0.0006  # Derivative gain for Z
        
        # Separate PID parameters for thermal tracking (much more conservative)
        self.thermal_pid_kp_y = 0.0008  # Lower proportional gain for thermal Y
        self.thermal_pid_ki_y = 0.0001  # Lower integral gain for thermal Y
        self.thermal_pid_kd_y = 0.0003  # Lower derivative gain for thermal Y
        
        self.thermal_pid_kp_z = 0.0008  # Lower proportional gain for thermal Z
        self.thermal_pid_ki_z = 0.0001  # Lower integral gain for thermal Z
        self.thermal_pid_kd_z = 0.0003  # Lower derivative gain for thermal Z
        
        # PID state variables
        self.error_integral_y = 0.0
        self.error_integral_z = 0.0
        self.previous_error_y = 0.0
        self.previous_error_z = 0.0
        
        # Speed limits for safety (different for face vs thermal tracking)
        self.max_speed_y_face = 2.5  # Max Y speed for face tracking (m/s)
        self.max_speed_z_face = 2.5  # Max Z speed for face tracking (m/s)
        self.max_speed_y_thermal = 0.5  # Max Y speed for thermal tracking (m/s)
        self.max_speed_z_thermal = 0.5  # Max Z speed for thermal tracking (m/s)
        
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
        self.last_thermal_position = None
        
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
            
            print(f"✓ Regular camera ready: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            print(f"✗ Regular camera failed: {e}")
            return False
    
    def init_thermal_camera(self):
        """Initialize thermal camera."""
        try:
            print("Initializing thermal camera...")
            self.thermal_cap = cv2.VideoCapture(THERMAL_CAMERA_INDEX)
            
            if not self.thermal_cap.isOpened():
                print(f"✗ Could not open thermal camera at index {THERMAL_CAMERA_INDEX}")
                return False
            
            # Set camera properties
            self.thermal_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.thermal_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.thermal_cap.set(cv2.CAP_PROP_FPS, 25)  # Set thermal camera to 25 Hz
            
            # Get actual properties
            actual_width = int(self.thermal_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.thermal_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.thermal_cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✓ Thermal camera ready: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            print(f"✗ Thermal camera failed: {e}")
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
    
    def find_hottest_point(self, frame):
        """Find the hottest point in thermal image (adapted from thermal_heat_tracking.py)."""
        try:
            # If the frame is colored, convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Threshold to find hot regions (use a high threshold, e.g., 90% of max)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
            thresh_val = maxVal * 0.9
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

            # Find contours of hot regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area > 200 pixels
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
            if not valid_contours:
                # Fallback to hottest pixel if no valid region
                x, y = maxLoc
                return x, y, maxVal

            # Find the contour with the greatest total heat
            max_heat = -1
            best_cx, best_cy, best_max = 0, 0, 0
            for cnt in valid_contours:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                region_vals = gray[mask == 255]
                total_heat = np.sum(region_vals)
                if total_heat > max_heat:
                    max_heat = total_heat
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    region_max = float(np.max(region_vals)) if region_vals.size > 0 else maxVal
                    best_cx, best_cy, best_max = cx, cy, region_max

            return best_cx, best_cy, best_max
        except Exception as e:
            print(f"Thermal detection error: {e}")
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
    
    def calculate_pid_speeds(self, target_x, target_y, is_thermal=False, center_x=None, center_y=None):
        """Calculate Y and Z speeds using PID control with adaptive gains based on distance."""
        # Use provided center coordinates or default to camera center
        if center_x is None:
            center_x = self.center_x
        if center_y is None:
            center_y = self.center_y
            
        # Get smoothed position (only for face tracking)
        if not is_thermal:
            smooth_x, smooth_y = self.smooth_face_position(target_x, target_y)
        else:
            smooth_x, smooth_y = target_x, target_y
        
        # Calculate errors (offset from center)
        error_x = smooth_x - center_x  # For Y movement (left/right)
        error_y = smooth_y - center_y  # For Z movement (up/down)
        
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
        
        # Select PID gains based on tracking mode
        if is_thermal:
            # Use thermal-specific PID gains (more conservative)
            base_kp_y = self.thermal_pid_kp_y
            base_ki_y = self.thermal_pid_ki_y
            base_kd_y = self.thermal_pid_kd_y
            base_kp_z = self.thermal_pid_kp_z
            base_ki_z = self.thermal_pid_ki_z
            base_kd_z = self.thermal_pid_kd_z
        else:
            # Use face tracking PID gains
            base_kp_y = self.pid_kp_y
            base_ki_y = self.pid_ki_y
            base_kd_y = self.pid_kd_y
            base_kp_z = self.pid_kp_z
            base_ki_z = self.pid_ki_z
            base_kd_z = self.pid_kd_z
        
        # Apply adaptive gains
        kp_y_adaptive = base_kp_y * gain_multiplier
        ki_y_adaptive = base_ki_y * gain_multiplier
        kd_y_adaptive = base_kd_y * gain_multiplier
        
        kp_z_adaptive = base_kp_z * gain_multiplier
        ki_z_adaptive = base_ki_z * gain_multiplier
        kd_z_adaptive = base_kd_z * gain_multiplier
        
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
        
        # Apply speed limits based on tracking mode
        if is_thermal:
            max_speed_y = self.max_speed_y_thermal
            max_speed_z = self.max_speed_z_thermal
        else:
            max_speed_y = self.max_speed_y_face
            max_speed_z = self.max_speed_z_face
        
        dy_limited = max(-max_speed_y, min(max_speed_y, dy))
        dz_limited = max(-max_speed_z, min(max_speed_z, dz))
        
        # Update previous errors
        self.previous_error_y = error_x
        self.previous_error_z = error_y
        
        return dy_limited, dz_limited
    
    def send_speed_command(self, dy, dz):
        """Send speedL command with dx=0 (X constrained) and different acceleration based on tracking mode."""
        try:
            if abs(dy) < 0.001 and abs(dz) < 0.001:
                # Stop robot if speeds are very small
                self.robot.send_program("stopl(0.2)")
                return
            
            # Set acceleration and time based on tracking mode
            if self.thermal_tracking_active:
                acceleration = 0.1  # Lower acceleration for thermal tracking
                time_param = 0.3    # Shorter time for thermal tracking to reduce overshooting
            else:
                acceleration = 1.0  # Default acceleration for face tracking
                time_param = 0.8    # Default time for face tracking
                
            # Send speedL command with dx=0 (X movement constrained) - should not move in base X
            urscript_cmd = f"speedl([0.0, {dy:.6f}, {dz:.6f}, 0, 0, 0], {acceleration}, {time_param})"
            self.robot.send_program(urscript_cmd)
            
        except Exception as e: 
            print(f"Error sending speed command: {e}")
    
    def control_loop(self):
        """Main control loop for face and thermal tracking - truly pauses when both tracking modes are inactive."""
        try:
            while self.control_running and self.running:
                try:
                    # Only run control loop when either tracking mode is active
                    if self.face_tracking_active or self.thermal_tracking_active:
                        target_position = None
                        is_thermal = False
                        
                        # Determine which tracking mode to use (face takes priority if both somehow active)
                        if self.face_tracking_active and self.last_face_position:
                            target_position = self.last_face_position
                            is_thermal = False
                        elif self.thermal_tracking_active and self.last_thermal_position:
                            target_position = self.last_thermal_position
                            is_thermal = True
                        
                        if target_position:
                            target_x, target_y = target_position
                            
                            # Calculate PID speeds with appropriate center coordinates
                            if is_thermal:
                                # Use thermal center coordinates (will be set in main loop)
                                dy, dz = self.calculate_pid_speeds(target_x, target_y, is_thermal, 
                                                                 getattr(self, 'thermal_center_x', self.center_x),
                                                                 getattr(self, 'thermal_center_y', self.center_y))
                            else:
                                # Use regular camera center coordinates
                                dy, dz = self.calculate_pid_speeds(target_x, target_y, is_thermal)
                            
                            # Send speed command
                            self.send_speed_command(dy, dz)
                        else:
                            # Stop robot when no target detected
                            self.robot.send_program("stopl(0.2)")
                        
                        time.sleep(1.0 / 60.0)  # 60 Hz control loop
                    else:
                        # When both tracking modes are paused, truly pause the control loop
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
        Only works when both face and thermal tracking are OFF.
        """
        while self.running:
            try:
                # Arrow keys only work when BOTH tracking modes are off
                if not self.face_tracking_active and not self.thermal_tracking_active:
                    if keyboard.is_pressed('up'):
                        try:
                            print(f"UP: Moving forward {self.x_move_step*1000:.0f}mm in tool X (both tracking modes OFF)")
                            urscript_cmd = f"movel(pose_trans(get_actual_tcp_pose(), p[{self.x_move_step:.6f}, 0, 0, 0, 0, 0]), a=0.5, v=0.2)"
                            print(f"DEBUG: Sending URScript: {urscript_cmd}")
                            self.robot.send_program(urscript_cmd)
                        except Exception as move_e:
                            print(f"UP movement error: {move_e}")
                        time.sleep(1.0)
                    elif keyboard.is_pressed('down'):
                        try:
                            print(f"DOWN: Moving backward {self.x_move_step*1000:.0f}mm in tool X (both tracking modes OFF)")
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
        print("F: Toggle face tracking, T: Toggle thermal tracking, SPACE: EMERGENCY STOP, UP/DOWN: X movement, ESC: Exit")
        
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
        
        if not self.init_thermal_camera():
            self.robot.close()
            return
        
        print("\n✓ All systems ready!")
        print("Controls:")
        print("  F: Toggle face tracking ON/OFF")
        print("  T: Toggle thermal tracking ON/OFF")
        print("  SPACE: EMERGENCY STOP")
        print("  UP/DOWN arrows: Move forward/backward (X-axis) - works only when both tracking modes are OFF")
        print("  ESC: Exit")
        print("\nBoth tracking modes start OFF - press F for face tracking or T for thermal tracking!")
        print("Note: Only one tracking mode can be active at a time. Max speed: Face=2.5m/s, Thermal=0.5m/s")
        
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
                # Capture from both cameras
                ret_regular, regular_frame = self.cap.read()
                ret_thermal, thermal_frame = self.thermal_cap.read()
                
                if not ret_regular:
                    print("Failed to capture regular frame")
                    continue
                    
                if not ret_thermal:
                    print("Failed to capture thermal frame")
                    continue
                
                # Process thermal frame (flip and crop)
                thermal_frame = cv2.rotate(thermal_frame, cv2.ROTATE_180)
                thermal_frame = thermal_frame[80:, :-80]
                
                # Resize frames to ensure they're the same height
                target_height = 480
                regular_frame = cv2.resize(regular_frame, (640, target_height))
                thermal_frame = cv2.resize(thermal_frame, (640, target_height))
                
                # After resizing, recalculate thermal center (since resize changes dimensions)
                thermal_center_x = thermal_frame.shape[1] // 2
                thermal_center_y = thermal_frame.shape[0] // 2
                
                # Store thermal center coordinates for use in control loop
                self.thermal_center_x = thermal_center_x
                self.thermal_center_y = thermal_center_y
                
                # Process face detection on regular camera
                face_data = self.detect_face(regular_frame)
                if face_data:
                    face_x, face_y, x, y, w, h = face_data
                    self.last_face_position = (face_x, face_y)
                    
                    # Draw face rectangle and center on regular frame
                    cv2.rectangle(regular_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(regular_frame, (face_x, face_y), 5, (0, 255, 0), -1)
                else:
                    self.last_face_position = None
                
                # Process thermal detection on thermal camera
                thermal_data = self.find_hottest_point(thermal_frame)
                if thermal_data:
                    hot_x, hot_y, max_val = thermal_data
                    self.last_thermal_position = (hot_x, hot_y)
                    
                    # Draw hottest point on thermal frame
                    cv2.circle(thermal_frame, (hot_x, hot_y), 8, (0, 0, 255), 2)
                    cv2.putText(thermal_frame, f"Hot: ({hot_x}, {hot_y}) {max_val:.1f}", (hot_x+10, hot_y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    self.last_thermal_position = None
                
                # Draw center crosshairs and deadzones on both frames with correct centers
                # Regular camera frame - use original center
                cv2.circle(regular_frame, (self.center_x, self.center_y), 3, (0, 0, 255), -1)
                cv2.line(regular_frame, (self.center_x-15, self.center_y), 
                        (self.center_x+15, self.center_y), (0, 0, 255), 2)
                cv2.line(regular_frame, (self.center_x, self.center_y-15), 
                        (self.center_x, self.center_y+15), (0, 0, 255), 2)
                cv2.circle(regular_frame, (self.center_x, self.center_y), 
                         self.deadzone_radius, (255, 255, 0), 2)
                
                # Thermal camera frame - use thermal center
                cv2.circle(thermal_frame, (thermal_center_x, thermal_center_y), 3, (0, 0, 255), -1)
                cv2.line(thermal_frame, (thermal_center_x-15, thermal_center_y), 
                        (thermal_center_x+15, thermal_center_y), (0, 0, 255), 2)
                cv2.line(thermal_frame, (thermal_center_x, thermal_center_y-15), 
                        (thermal_center_x, thermal_center_y+15), (0, 0, 255), 2)
                cv2.circle(thermal_frame, (thermal_center_x, thermal_center_y), 
                         self.deadzone_radius, (255, 255, 0), 2)
                
                # Add tracking status indicators to each frame
                # Regular camera frame
                face_color = (0, 255, 0) if self.face_tracking_active else (128, 128, 128)
                cv2.putText(regular_frame, "FACE CAMERA", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
                if self.face_tracking_active:
                    cv2.putText(regular_frame, "ACTIVE", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Add border to indicate active tracking
                    cv2.rectangle(regular_frame, (0, 0), (regular_frame.shape[1]-1, regular_frame.shape[0]-1), 
                                 (0, 255, 0), 3)
                
                # Thermal camera frame
                thermal_color = (0, 0, 255) if self.thermal_tracking_active else (128, 128, 128)
                cv2.putText(thermal_frame, "THERMAL CAMERA", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, thermal_color, 2)
                if self.thermal_tracking_active:
                    cv2.putText(thermal_frame, "ACTIVE", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # Add border to indicate active tracking
                    cv2.rectangle(thermal_frame, (0, 0), (thermal_frame.shape[1]-1, thermal_frame.shape[0]-1), 
                                 (0, 0, 255), 3)
                
                # Combine frames side by side
                combined_frame = np.hstack((regular_frame, thermal_frame))
                
                # Add overall status at the top of combined frame
                status_text = ""
                if self.face_tracking_active:
                    status_text = "FACE TRACKING ACTIVE (Max Speed: 2.5 m/s)"
                    status_color = (0, 255, 0)
                elif self.thermal_tracking_active:
                    status_text = "THERMAL TRACKING ACTIVE (Max Speed: 0.5 m/s)"
                    status_color = (0, 0, 255)
                else:
                    status_text = "TRACKING OFF - Arrow keys available"
                    status_color = (255, 255, 0)
                
                cv2.putText(combined_frame, status_text, (10, combined_frame.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                # Controls info
                cv2.putText(combined_frame, "F: Face | T: Thermal | SPACE: Emergency Stop | UP/DOWN: X move | ESC: Exit", 
                           (10, combined_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (255, 255, 255), 2)
                
                # Show combined frame
                cv2.imshow('Face & Thermal Tracker', combined_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Exiting...")
                    break
                elif key == ord('f') or key == ord('F'):  # F - Face tracking toggle
                    if self.thermal_tracking_active:
                        # Turn off thermal tracking first
                        self.thermal_tracking_active = False
                        print("Thermal tracking turned OFF")
                    
                    self.face_tracking_active = not self.face_tracking_active
                    status_text = "ACTIVATED" if self.face_tracking_active else "DEACTIVATED"
                    print(f"Face tracking {status_text}")
                    
                    # Reset PID state when toggling
                    self.error_integral_y = 0.0
                    self.error_integral_z = 0.0
                    self.previous_error_y = 0.0
                    self.previous_error_z = 0.0
                    
                    if not self.face_tracking_active:
                        # Stop robot immediately when pausing
                        self.robot.send_program("stopl(0.5)")
                        print("Face tracking paused - arrow keys available when thermal is also off")
                    else:
                        print("Face tracking active - arrow keys disabled")
                        
                elif key == ord(' '):  # SPACE - Emergency Stop
                    print("EMERGENCY STOP ACTIVATED!")
                    # Turn off all tracking modes
                    self.face_tracking_active = False
                    self.thermal_tracking_active = False
                    
                    # Reset PID state
                    self.error_integral_y = 0.0
                    self.error_integral_z = 0.0
                    self.previous_error_y = 0.0
                    self.previous_error_z = 0.0
                    
                    # Send emergency stop command
                    self.robot.send_program("stopl(0.1)")  # Quick stop
                    print("All tracking stopped - Robot emergency stopped - Arrow keys available")
                        
                elif key == ord('t') or key == ord('T'):  # T - Thermal tracking toggle
                    if self.face_tracking_active:
                        # Turn off face tracking first
                        self.face_tracking_active = False
                        print("Face tracking turned OFF")
                    
                    self.thermal_tracking_active = not self.thermal_tracking_active
                    status_text = "ACTIVATED" if self.thermal_tracking_active else "DEACTIVATED"
                    print(f"Thermal tracking {status_text}")
                    
                    # Reset PID state when toggling
                    self.error_integral_y = 0.0
                    self.error_integral_z = 0.0
                    self.previous_error_y = 0.0
                    self.previous_error_z = 0.0
                    
                    if not self.thermal_tracking_active:
                        # Stop robot immediately when pausing
                        self.robot.send_program("stopl(0.5)")
                        print("Thermal tracking paused - arrow keys available when face is also off")
                    else:
                        print("Thermal tracking active - arrow keys disabled")
                
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



