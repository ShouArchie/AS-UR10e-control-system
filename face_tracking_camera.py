import urx
import cv2
import numpy as np
import time
import math
import threading
import keyboard
import mediapipe as mp

# ===== CONFIGURATION VARIABLES =====
# Easy-to-edit settings at the top of the file

# Robot connection settings
ROBOT_IP = "192.168.0.101"  # Change this to your robot's IP address

# Robot movement speed settings
FACE_TRACKING_ACCELERATION = 0.5   # Acceleration for face tracking movements (0.1-1.0, higher = faster)
FACE_TRACKING_VELOCITY = 0.4       # Velocity for face tracking movements (0.05-0.5, higher = faster)

ARROW_KEY_ACCELERATION = 0.3       # Acceleration for arrow key movements (0.1-1.0, higher = faster)  
ARROW_KEY_VELOCITY = 0.15          # Velocity for arrow key movements (0.05-0.5, higher = faster)

# Movement sensitivity settings
ARROW_KEY_DISTANCE_STEP = 0.15     # How far each arrow key press moves (0.05-0.3, higher = bigger steps)
SHOULDER_MULTIPLIER = 1.0          # Shoulder joint adjustment multiplier for arrow keys (0.5-2.0)
ELBOW_MULTIPLIER = 1.5             # Elbow joint adjustment multiplier for arrow keys (0.5-2.0)

# Face tracking sensitivity settings  
FACE_TRACKING_SENSITIVITY = 0.015  # How sensitive face tracking is (0.005-0.02, higher = bigger movements)
FACE_BASE_MULTIPLIER = 5.0         # Base joint adjustment for left/right face movement (2.0-8.0, higher = faster)
FACE_SHOULDER_MULTIPLIER = 2.5     # Shoulder joint adjustment for up/down face movement (1.0-4.0, higher = faster)  
FACE_ELBOW_MULTIPLIER = 2.0        # Elbow joint adjustment for up/down face movement (1.0-3.0, higher = faster)

# Dead zone settings
CENTER_DEAD_ZONE_RADIUS = 40      # Circular dead zone around center in pixels (10-50, smaller = more responsive)
MOVEMENT_DEAD_ZONE = 20           # Linear dead zone for individual movements (5-25, smaller = more responsive)
RATE_LIMIT_DELAY = 0.05           # Delay between movement commands in seconds (0.02-0.1, smaller = more frequent)

# ===== END CONFIGURATION =====

class FaceTrackingCamera:
    """
    Face tracking camera system for UR10e robot using Google's MediaPipe.
    
    Features:
    - Face tracking up/down controls blue arrow (up/down movements)
    - Face tracking left/right controls green arrow (left/right movements)
    - Arrow keys control red arrow (forward/back toward/away from user)
    - Tool-relative movements ensure consistent behavior regardless of tool orientation
    - Google MediaPipe for smooth and accurate face detection
    - Maintains wrist1 joint relationship: 90° + elbow_angle - abs(shoulder_angle)
    """
    
    def __init__(self, robot_ip=ROBOT_IP, camera_index=0):
        """Initialize the face tracking camera system."""
        self.robot = None
        self.robot_ip = robot_ip
        self.camera_index = camera_index
        self.cap = None
        
        # MediaPipe face detection (Google's version)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (2 meters), 1 for full-range (5 meters)
            min_detection_confidence=0.5
        )
        
        # Tracking parameters
        self.tracking_active = False
        self.last_face_center = None
        self.stable_distance = 0.5  # 50cm stable distance from face
        self.current_distance = self.stable_distance
        
        # Movement parameters - using configuration variables for better responsiveness
        self.movement_sensitivity = FACE_TRACKING_SENSITIVITY  # Using configurable sensitivity
        self.max_movement_per_step = 0.05  # Increased from 0.02 for larger steps
        self.movement_smoothing = 0.2  # Reduced for more responsive movement
        
        # Movement smoothing for better tracking
        self.last_movement = [0.0, 0.0, 0.0]  # Store last movement for smoothing
        self.movement_filter_strength = 0.5  # Reduced from 0.7 for less smoothing, more responsiveness
        self.min_movement_threshold = 0.0003  # Further reduced threshold for more responsive movement
        
        # Camera frame parameters
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Dead zone to prevent jittery movements - using configuration variables
        self.dead_zone_pixels = MOVEMENT_DEAD_ZONE  # Using configurable dead zone
        # Circular dead zone radius - if face is within this radius of center, no movement commands sent
        self.center_dead_zone_radius = CENTER_DEAD_ZONE_RADIUS  # Using configurable center dead zone
        
        # Current robot position
        self.current_pose = None
        self.base_orientation = None  # Base orientation for camera pointing
        
        # Threading
        self.running = False
        self.camera_thread = None
        self.keyboard_thread = None
        
    def connect_robot(self):
        """Connect to the robot."""
        try:
            print(f"Connecting to UR10e face tracking robot at {self.robot_ip}...")
            self.robot = urx.Robot(self.robot_ip)
            print("✓ Robot connected successfully!")
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"✗ Robot connection failed: {e}")
            return False
    
    def disconnect_robot(self):
        """Disconnect from the robot."""
        if self.robot:
            self.robot.close()
            print("Face tracking robot disconnected.")
    
    def initialize_camera(self):
        """Initialize the camera."""
        try:
            print(f"Initializing camera {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"✗ Could not open camera {self.camera_index}")
                return False
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            print("✓ Camera initialized successfully!")
            return True
        except Exception as e:
            print(f"✗ Camera initialization failed: {e}")
            return False
    
    def release_camera(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Camera released.")
    
    def get_robot_pose(self):
        """Get robot pose as a list of floats, handling PoseVector objects robustly."""
        try:
            pose = self.robot.getl()
            
            # Handle different possible formats
            if hasattr(pose, 'array'):
                # Try different ways to access the array
                try:
                    if hasattr(pose.array, 'tolist'):
                        return pose.array.tolist()
                    else:
                        return list(pose.array)
                except:
                    # If array access fails, try to convert directly
                    return [float(x) for x in pose.array]
            elif hasattr(pose, 'get_array'):
                array_data = pose.get_array()
                if hasattr(array_data, 'tolist'):
                    return array_data.tolist()
                else:
                    return list(array_data)
            elif hasattr(pose, 'tolist'):
                return pose.tolist()
            elif hasattr(pose, '__iter__'):
                # If it's iterable, convert to list
                return [float(x) for x in pose]
            else:
                # Last resort: try to access as attributes
                print("Using attribute access fallback for pose")
                return [float(pose[i]) for i in range(6)]
            
        except Exception as e:
            print(f"Error getting robot pose with getl(): {e}")
            
            # Ultimate fallback: use URScript to get pose
            try:
                print("Trying URScript fallback for pose...")
                urscript_cmd = "get_actual_tcp_pose()"
                # This approach gets the pose through URScript
                # We'll need to use a different method since direct URScript return is complex
                
                # Alternative: try individual property access with float conversion
                x = float(self.robot.x) if hasattr(self.robot.x, '__float__') else self.robot.x.array[0] if hasattr(self.robot.x, 'array') else 0.0
                y = float(self.robot.y) if hasattr(self.robot.y, '__float__') else self.robot.y.array[0] if hasattr(self.robot.y, 'array') else 0.0
                z = float(self.robot.z) if hasattr(self.robot.z, '__float__') else self.robot.z.array[0] if hasattr(self.robot.z, 'array') else 0.0
                rx = float(self.robot.rx) if hasattr(self.robot.rx, '__float__') else self.robot.rx.array[0] if hasattr(self.robot.rx, 'array') else 0.0
                ry = float(self.robot.ry) if hasattr(self.robot.ry, '__float__') else self.robot.ry.array[0] if hasattr(self.robot.ry, 'array') else 0.0
                rz = float(self.robot.rz) if hasattr(self.robot.rz, '__float__') else self.robot.rz.array[0] if hasattr(self.robot.rz, 'array') else 0.0
                
                return [x, y, z, rx, ry, rz]
                
            except Exception as e2:
                print(f"Fallback pose retrieval also failed: {e2}")
                # Return last known pose or zeros
                if hasattr(self, 'last_known_pose'):
                    print("Using last known pose")
                    return self.last_known_pose
                else:
                    print("Using zero pose as last resort")
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def safe_translate(self, movement, acc=0.3, vel=0.15):
        """Safely translate robot position using URScript to bypass PoseVector issues."""
        try:
            move_x, move_y, move_z = movement
            
            # Get current pose
            current_pose = self.get_robot_pose()
            self.last_known_pose = current_pose.copy()  # Store for fallback
            
            # Calculate new position
            new_x = current_pose[0] + move_x
            new_y = current_pose[1] + move_y
            new_z = current_pose[2] + move_z
            
            # Keep same orientation
            new_rx, new_ry, new_rz = current_pose[3], current_pose[4], current_pose[5]
            
            # Use URScript movel command directly to bypass translate() PoseVector issues
            urscript_cmd = f"movel(p[{new_x:.6f}, {new_y:.6f}, {new_z:.6f}, {new_rx:.6f}, {new_ry:.6f}, {new_rz:.6f}], a={acc}, v={vel})"
            print(f"Sending safe translate: {urscript_cmd}")
            self.robot.send_program(urscript_cmd)
            
        except Exception as e:
            print(f"Error in safe_translate: {e}")
    
    def move_robot_smooth(self, movement):
        """Move robot smoothly to track face using joint-based movement to preserve wrist1 relationship."""
        if movement is None or not self.tracking_active:
            return
        
        # Check if robot is still executing the previous movement command
        try:
            if self.robot.is_program_running():
                # Robot is still moving, skip this update to prevent choppy movement
                if not hasattr(self, '_movement_skip_counter'):
                    self._movement_skip_counter = 0
                self._movement_skip_counter += 1
                
                if self._movement_skip_counter % 20 == 0:  # Print every 20th skip
                    print(f"Waiting for robot to complete previous movement... (skipped {self._movement_skip_counter} updates)")
                return
            else:
                # Reset skip counter when robot is ready for new commands
                if hasattr(self, '_movement_skip_counter') and self._movement_skip_counter > 0:
                    print(f"Robot ready for new movement after {self._movement_skip_counter} skipped updates")
                    self._movement_skip_counter = 0
        except Exception as e:
            print(f"Error checking robot program status: {e}")
            # Continue with movement if we can't check status
        
        # Rate limiting: only send commands every few frames to prevent choppy movement
        if not hasattr(self, '_last_robot_command_time'):
            self._last_robot_command_time = 0
        
        current_time = time.time()
        if current_time - self._last_robot_command_time < RATE_LIMIT_DELAY:  # Using configurable rate limiting
            return
        
        # Check if movement is significant enough to warrant a robot command
        if (abs(movement[0]) < self.min_movement_threshold and 
            abs(movement[2]) < self.min_movement_threshold):
            return  # Skip insignificant movements
        
        try:
            move_x, move_y, move_z = movement
            
            # Get current joint positions to maintain relationships
            current_joints = self.robot.getj()
            
            # Extract current joint angles
            base_angle = current_joints[0]          # Joint 0 (base)
            shoulder_angle = current_joints[1]      # Joint 1 (shoulder) 
            elbow_angle = current_joints[2]         # Joint 2 (elbow)
            wrist1_angle = current_joints[3]        # Joint 3 (wrist 1)
            wrist2_angle = current_joints[4]        # Joint 4 (wrist 2)
            wrist3_angle = current_joints[5]        # Joint 5 (wrist 3)
            
            print(f"\n=== ROBOT MOVEMENT DEBUG ===")
            print(f"Face movement input: left/right={move_x:.6f}, up/down={move_z:.6f}")
            print(f"Current joint positions (degrees): {[f'{math.degrees(j):.1f}°' for j in current_joints]}")
            
            # Map face movements to joint adjustments:
            # Face left/right (move_x) → Green arrow → Base joint rotation
            # Face up/down (move_z) → Blue arrow → Shoulder/elbow adjustments (FLIPPED)
            
            # Calculate joint adjustments using configurable multipliers for more aggressive tracking
            base_adjustment = move_x * FACE_BASE_MULTIPLIER          # Using configurable base multiplier (5.0)
            shoulder_adjustment = -move_z * FACE_SHOULDER_MULTIPLIER  # Using configurable shoulder multiplier (2.5)
            elbow_adjustment = move_z * FACE_ELBOW_MULTIPLIER         # Using configurable elbow multiplier (2.0)
            
            print(f"Joint adjustments (degrees): Base={math.degrees(base_adjustment):.3f}°, Shoulder={math.degrees(shoulder_adjustment):.3f}°, Elbow={math.degrees(elbow_adjustment):.3f}°")
            
            # Calculate new joint angles
            new_base_angle = base_angle + base_adjustment
            new_shoulder_angle = shoulder_angle + shoulder_adjustment  
            new_elbow_angle = elbow_angle + elbow_adjustment
            
            # Apply the wrist1 formula to maintain relationship
            # wrist1_angle = 90° + elbow_angle - abs(shoulder_angle)
            shoulder_angle_deg = math.degrees(new_shoulder_angle)
            elbow_angle_deg = math.degrees(new_elbow_angle)
            wrist1_angle_deg = 90 + elbow_angle_deg - abs(shoulder_angle_deg)
            new_wrist1_angle = math.radians(-wrist1_angle_deg)  # Convert back to radians with correct sign
            
            # Apply the wrist3 formula: wrist3 = -90° + base_angle
            base_angle_deg = math.degrees(new_base_angle)
            wrist3_angle_deg = -90 + base_angle_deg
            new_wrist3_angle = math.radians(wrist3_angle_deg)
            
            print(f"Wrist1 formula: 90° + {elbow_angle_deg:.1f}° - abs({shoulder_angle_deg:.1f}°) = {wrist1_angle_deg:.1f}°")
            print(f"Wrist3 formula: -90° + {base_angle_deg:.1f}° = {wrist3_angle_deg:.1f}°")
            
            # Create new joint positions maintaining the relationships
            new_joints = [
                new_base_angle,       # Adjusted base for left/right tracking
                new_shoulder_angle,   # Adjusted shoulder for up/down tracking
                new_elbow_angle,      # Adjusted elbow for up/down tracking
                new_wrist1_angle,     # Calculated using formula
                wrist2_angle,         # Keep wrist 2 unchanged
                new_wrist3_angle      # Calculated using wrist3 formula
            ]
            
            print(f"NEW joint positions (degrees): {[f'{math.degrees(j):.1f}°' for j in new_joints]}")
            
            # Get current Cartesian position for reference
            try:
                current_pose = self.get_robot_pose()
                print(f"Current Cartesian position: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Z={current_pose[2]:.3f}")
            except:
                print("Could not get current Cartesian position")
            
            print(f"Joint adjustments: Base: {math.degrees(base_adjustment):.2f}°, Shoulder: {math.degrees(shoulder_adjustment):.2f}°, Elbow: {math.degrees(elbow_adjustment):.2f}°")
            
            # Use movej to maintain exact joint relationships and prevent unwanted rotations
            # Using configurable acceleration and velocity for face tracking movements
            print(f"Sending movej command with acc={FACE_TRACKING_ACCELERATION}, vel={FACE_TRACKING_VELOCITY}...")
            self.robot.movej(new_joints, acc=FACE_TRACKING_ACCELERATION, vel=FACE_TRACKING_VELOCITY)
            
            # Update the last command time
            self._last_robot_command_time = current_time
            
            print(f"=== END DEBUG ===\n")
            
        except Exception as e:
            print(f"Error moving robot: {e}")
    
    def update_orientation_to_face(self, face_center):
        """Update robot orientation to point red arrow directly at the face."""
        if face_center is None or not self.tracking_active:
            return
        
        try:
            # Calculate the orientation needed to point red arrow at face
            target_orientation = self.calculate_face_pointing_orientation(face_center)
            
            if target_orientation is None:
                return
            
            # Get current position using helper method to avoid PoseVector issues
            current_pose = self.get_robot_pose()
            
            # Apply smoothing to orientation changes to prevent jerky movements
            orientation_smoothing = 0.1  # Lower value for smoother orientation changes
            
            current_rx, current_ry, current_rz = current_pose[3], current_pose[4], current_pose[5]
            target_rx, target_ry, target_rz = target_orientation
            
            # Smooth orientation changes
            new_rx = current_rx + (target_rx - current_rx) * orientation_smoothing
            new_ry = current_ry + (target_ry - current_ry) * orientation_smoothing
            new_rz = current_rz + (target_rz - current_rz) * orientation_smoothing
            
            # Keep current position, only change orientation
            new_x, new_y, new_z = current_pose[0], current_pose[1], current_pose[2]
            
            # Move robot to new orientation
            urscript_cmd = f"movel(p[{new_x}, {new_y}, {new_z}, {new_rx}, {new_ry}, {new_rz}], a=0.3, v=0.2)"
            self.robot.send_program(urscript_cmd)
            
        except Exception as e:
            print(f"Error updating orientation to face: {e}")
    
    def handle_distance_control(self):
        """Handle manual distance control with arrow keys along red arrow (forward/back)."""
        while self.running:
            try:
                if keyboard.is_pressed('up'):
                    # Move back along red arrow (away from user) - using configurable distance step
                    self.adjust_distance(ARROW_KEY_DISTANCE_STEP)  # Positive = move back/away
                    time.sleep(0.08)  # Reduced delay for more responsive control
                elif keyboard.is_pressed('down'):
                    # Move forward along red arrow (toward user) - using configurable distance step
                    self.adjust_distance(-ARROW_KEY_DISTANCE_STEP)  # Negative = move forward/closer
                    time.sleep(0.08)  # Reduced delay for more responsive control
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"Error in distance control: {e}")
                time.sleep(0.1)
    
    def adjust_distance(self, distance_change):
        """Adjust the distance using joint-based movement to preserve the wrist1 and wrist3 relationship formulas."""
        try:
            # Check if robot is still executing the previous movement command
            try:
                if self.robot.is_program_running():
                    # Robot is still moving, skip this command to prevent choppy movement
                    print("Robot still moving, waiting for completion before distance adjustment...")
                    return
            except Exception as e:
                print(f"Error checking robot program status: {e}")
                # Continue with movement if we can't check status
            
            # Get current joint positions to maintain relationships
            current_joints = self.robot.getj()
            print(f"Current joints before movement: {[f'{math.degrees(j):.1f}°' for j in current_joints]}")
            
            # Extract current joint angles
            base_angle = current_joints[0]          # Joint 0 (base)
            shoulder_angle = current_joints[1]      # Joint 1 (shoulder) 
            elbow_angle = current_joints[2]         # Joint 2 (elbow)
            wrist1_angle = current_joints[3]        # Joint 3 (wrist 1)
            wrist2_angle = current_joints[4]        # Joint 4 (wrist 2)
            wrist3_angle = current_joints[5]        # Joint 5 (wrist 3)
            
            # For red arrow movement (forward/back), we primarily adjust the shoulder and elbow
            # while maintaining the wrist1 and wrist3 relationship formulas
            
            # Calculate movement adjustments for forward/back motion
            # Positive distance_change = move away, negative = move closer
            shoulder_adjustment = distance_change * SHOULDER_MULTIPLIER  # Using configurable shoulder multiplier
            elbow_adjustment = -distance_change * ELBOW_MULTIPLIER       # Using configurable elbow multiplier
            
            # Calculate new joint angles
            new_shoulder_angle = shoulder_angle + shoulder_adjustment
            new_elbow_angle = elbow_angle + elbow_adjustment
            
            # Apply the wrist1 formula to maintain relationship
            # wrist1_angle = 90° + elbow_angle - abs(shoulder_angle)
            shoulder_angle_deg = math.degrees(new_shoulder_angle)
            elbow_angle_deg = math.degrees(new_elbow_angle)
            wrist1_angle_deg = 90 + elbow_angle_deg - abs(shoulder_angle_deg)
            new_wrist1_angle = math.radians(-wrist1_angle_deg)  # Convert back to radians with correct sign
            
            # Apply the wrist3 formula: wrist3 = -90° + base_angle
            base_angle_deg = math.degrees(base_angle)  # Base doesn't change for distance adjustment
            wrist3_angle_deg = -90 + base_angle_deg
            new_wrist3_angle = math.radians(wrist3_angle_deg)
            
            print(f"Wrist1 formula: 90° + {elbow_angle_deg:.1f}° - abs({shoulder_angle_deg:.1f}°) = {wrist1_angle_deg:.1f}°")
            print(f"Wrist3 formula: -90° + {base_angle_deg:.1f}° = {wrist3_angle_deg:.1f}°")
            
            # Create new joint positions maintaining the relationships
            new_joints = [
                base_angle,           # Keep base joint unchanged
                new_shoulder_angle,   # Adjusted shoulder
                new_elbow_angle,      # Adjusted elbow  
                new_wrist1_angle,     # Calculated using formula
                wrist2_angle,         # Keep wrist 2 unchanged
                new_wrist3_angle      # Calculated using wrist3 formula
            ]
            
            print(f"New joints: {[f'{math.degrees(j):.1f}°' for j in new_joints]}")
            
            # Use movej to maintain exact joint relationships and prevent unwanted rotations
            # Using configurable acceleration and velocity for arrow key movements
            self.robot.movej(new_joints, acc=ARROW_KEY_ACCELERATION, vel=ARROW_KEY_VELOCITY)
            
            direction = "closer" if distance_change < 0 else "further"
            print(f"Moving {direction} using joint-based movement - preserving wrist1 and wrist3 relationships")
            print(f"Shoulder: {math.degrees(shoulder_angle):.1f}° → {math.degrees(new_shoulder_angle):.1f}°")
            print(f"Elbow: {math.degrees(elbow_angle):.1f}° → {math.degrees(new_elbow_angle):.1f}°")
            print(f"Wrist1: {math.degrees(wrist1_angle):.1f}° → {math.degrees(new_wrist1_angle):.1f}°")
            print(f"Wrist3: {math.degrees(wrist3_angle):.1f}° → {math.degrees(new_wrist3_angle):.1f}°")
            
        except Exception as e:
            print(f"Error adjusting distance: {e}")
    
    def camera_loop(self):
        """Main camera processing loop."""
        print("Starting face tracking camera loop...")
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, -1)  # Changed from 1 to -1 for 180-degree flip
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process face tracking
                if len(faces) > 0 and self.tracking_active:
                    face_center = self.calculate_face_center(faces)
                    movement = self.calculate_movement(face_center)
                    
                    # Move robot along green/blue arrows for face tracking
                    self.move_robot_smooth(movement)
                    
                    # Update orientation to point red arrow at face
                    # self.update_orientation_to_face(face_center) # Temporarily commented out for debugging
                    
                    # Draw face detection with confidence scores
                    for face in faces:
                        if len(face) == 5:  # MediaPipe format with confidence
                            x, y, w, h, confidence = face
                            # Draw rectangle with confidence-based color
                            color_intensity = int(255 * confidence)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, color_intensity, 0), 2)
                            # Draw confidence score
                            cv2.putText(frame, f"{confidence:.2f}", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:  # Fallback format
                            x, y, w, h = face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw center crosshair
                    if face_center:
                        face_x, face_y, _, _ = face_center
                        cv2.circle(frame, (face_x, face_y), 5, (0, 255, 0), -1)
                
                # Draw frame center
                cv2.circle(frame, (self.frame_center_x, self.frame_center_y), 3, (0, 0, 255), -1)
                cv2.line(frame, (self.frame_center_x-20, self.frame_center_y), 
                        (self.frame_center_x+20, self.frame_center_y), (0, 0, 255), 1)
                cv2.line(frame, (self.frame_center_x, self.frame_center_y-20), 
                        (self.frame_center_x, self.frame_center_y+20), (0, 0, 255), 1)
                
                # Draw status
                status = "TRACKING" if self.tracking_active else "PAUSED"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0) if self.tracking_active else (0, 0, 255), 2)
                
                cv2.putText(frame, f"Distance: {self.current_distance:.2f}m", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, "SPACE: Toggle tracking, UP/DOWN: Distance, ESC: Exit", 
                           (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Face Tracking Camera', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("Exiting face tracking...")
                    self.running = False
                    break
                elif key == ord(' '):  # Space key
                    self.tracking_active = not self.tracking_active
                    status = "activated" if self.tracking_active else "paused"
                    print(f"Face tracking {status}")
                
                time.sleep(0.01)  # Increased frame rate from ~30 FPS to ~60 FPS for smoother tracking
                
            except Exception as e:
                print(f"Error in camera loop: {e}")
                time.sleep(0.1)
    
    def start_tracking(self):
        """Start the face tracking system."""
        try:
            print("\n=== Face Tracking Camera System ===")
            print("Features:")
            print("- Face tracking up/down controls blue arrow (up/down movements)")
            print("- Face tracking left/right controls green arrow (left/right movements)")
            print("- Arrow keys control red arrow (forward/back toward/away from user)")
            print("- SPACE: Toggle tracking, ESC: Exit")
            print("- Google MediaPipe for smooth and accurate face detection")
            
            # Initialize systems
            if not self.connect_robot():
                return False
            
            if not self.initialize_camera():
                self.disconnect_robot()
                return False
            
            # Move to optimal starting position first
            if not self.move_to_starting_position():
                self.release_camera()
                self.disconnect_robot()
                return False
            
            # Setup camera position parameters
            if not self.setup_camera_position():
                self.release_camera()
                self.disconnect_robot()
                return False
            
            print("\n✓ System ready for face tracking!")
            print("Position yourself in front of the camera and press SPACE to start tracking")
            print("Face tracking controls:")
            print("- Move your face up/down → Blue arrow movement (up/down)")
            print("- Move your face left/right → Green arrow movement (left/right)")
            print("- Arrow keys → Red arrow movement (forward/back toward/away from user)")
            
            # Start threads
            self.running = True
            self.tracking_active = False  # Start paused so user can position themselves
            
            # Start keyboard monitoring thread
            self.keyboard_thread = threading.Thread(target=self.handle_distance_control)
            self.keyboard_thread.daemon = True
            self.keyboard_thread.start()
            
            # Start camera loop (main thread)
            self.camera_loop()
            
            return True
            
        except Exception as e:
            print(f"Error starting face tracking: {e}")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up face tracking system...")
        self.running = False
        
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=1)
        
        self.release_camera()
        self.disconnect_robot()
        print("Face tracking system stopped.")

    def calculate_face_pointing_orientation(self, face_center):
        """Calculate the orientation needed to point red arrow directly at the face."""
        if face_center is None:
            return None
        
        try:
            # Get current robot position using helper method to avoid PoseVector issues
            robot_pose = self.get_robot_pose()
            robot_pos = robot_pose[:3]  # Extract [x, y, z]
            
            # Estimate face position in 3D space
            # This is a simplified estimation - in reality you'd need depth info
            face_x, face_y, face_w, face_h = face_center
            
            # Convert camera pixel coordinates to estimated 3D position
            # Assume face is at a certain distance and convert pixel offset to world offset
            estimated_face_distance = 0.8  # Assume face is 80cm away
            
            # Calculate face offset from camera center in world coordinates
            pixel_to_world_scale = 15  # Rough conversion factor ARCHIEEDIT
            face_offset_x = (face_x - self.frame_center_x) * pixel_to_world_scale
            face_offset_y = -(face_y - self.frame_center_y) * pixel_to_world_scale  # Invert Y
            
            # Estimate face position in world coordinates
            # This assumes camera is roughly aligned with world coordinates initially
            estimated_face_x = robot_pos[0] + face_offset_x
            estimated_face_y = robot_pos[1] + estimated_face_distance
            estimated_face_z = robot_pos[2] + face_offset_y
            
            # Calculate direction vector from robot to face
            direction_x = estimated_face_x - robot_pos[0]
            direction_y = estimated_face_y - robot_pos[1]
            direction_z = estimated_face_z - robot_pos[2]
            
            # Normalize direction vector
            length = math.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
            if length > 0:
                direction_x /= length
                direction_y /= length
                direction_z /= length
            
            # Calculate orientation to point red arrow (X-axis) toward face
            # Red arrow should align with direction vector
            
            # Calculate rotation around Z-axis (RZ) - yaw
            rz = math.atan2(direction_y, direction_x)
            
            # Calculate rotation around Y-axis (RY) - pitch
            ry = -math.asin(direction_z)
            
            # Keep RX (roll) at 0 for simplicity
            rx = 0.0
            
            return [rx, ry, rz]
            
        except Exception as e:
            print(f"Error calculating face pointing orientation: {e}")
            return None

    def setup_camera_position(self):
        """Set up initial camera position and orientation."""
        try:
            print("Setting up camera position...")
            
            # Get current position using helper method to avoid PoseVector issues
            self.current_pose = self.get_robot_pose()
            
            # Set base orientation - camera pointing forward (blue arrow direction)
            # Blue arrow (Z-axis) should point toward the face
            self.base_orientation = [0.0, 0.0, 0.0]  # Camera pointing forward
            
            print(f"Current position: {[f'{x:.3f}' for x in self.current_pose[:3]]}")
            print(f"Camera orientation: Blue arrow pointing forward")
            print(f"Stable tracking distance: {self.stable_distance:.2f}m")
            
            return True
            
        except Exception as e:
            print(f"Error setting up camera position: {e}")
            return False
    
    def move_to_starting_position(self):
        """Move robot to optimal starting position for face tracking using joint positions."""
        try:
            print("Moving to face tracking starting position using joint positions...")
            
            # Get current joint positions as reference
            current_joints = self.robot.getj()
            print(f"Current joint positions: {[f'{math.degrees(j):.1f}°' for j in current_joints]}")
            
            # Define optimal starting joint positions
            # Set base joint (joint 0) to 0 degrees and wrist 3 (joint 5) to -90 degrees
            
            # Convert desired degrees to radians for calculations
            shoulder_angle_deg = -60.0
            elbow_angle_deg = 80
            
            shoulder_angle_rad = math.radians(shoulder_angle_deg)
            elbow_angle_rad = math.radians(elbow_angle_deg)
            
            # Wrist 1 (joint 3) calculation using the correct formula:
            # wrist1_angle = 90 degrees + elbow_angle - abs(shoulder_angle)
            # This ensures proper orientation for the face tracking task
            wrist1_angle_deg = 90 + elbow_angle_deg - abs(shoulder_angle_deg)
            wrist1_angle_rad = math.radians(-wrist1_angle_deg)
            
            # Wrist 3 (joint 5) calculation using the formula:
            # wrist3_angle = -90 degrees + base_angle  
            base_angle_deg = 0.0  # Base joint is 0 degrees
            wrist3_angle_deg = -90 + base_angle_deg
            wrist3_angle_rad = math.radians(wrist3_angle_deg)
            
            print(f"Wrist1 calculation: 90° + {elbow_angle_deg}° - abs({shoulder_angle_deg}°) = {wrist1_angle_deg}°")
            print(f"Wrist3 calculation: -90° + {base_angle_deg}° = {wrist3_angle_deg}°")
            
            start_joints = [
                0.0,                    # Base joint (joint 0) = 0 degrees
                shoulder_angle_rad,     # Shoulder joint (joint 1) = 60 degrees
                elbow_angle_rad,        # Elbow joint (joint 2) = 120 degrees
                wrist1_angle_rad,       # Wrist 1 joint (joint 3) calculated using formula
                3*math.pi/2,            # Keep current wrist 2 joint (or set to a specific value, e.g., math.pi/2)
                wrist3_angle_rad        # Wrist 3 joint (joint 5) calculated using formula
            ]
            
            print(f"Target joint positions: {[f'{math.degrees(j):.1f}°' for j in start_joints]}")
            
            # Check if movement is needed (tolerance of 0.01 radians ≈ 0.6 degrees)
            movement_needed = False
            tolerance = 0.01
            for i, (current, target) in enumerate(zip(current_joints, start_joints)):
                if abs(current - target) > tolerance:
                    movement_needed = True
                    print(f"Joint {i} needs movement: {math.degrees(current):.1f}° → {math.degrees(target):.1f}°")
            
            if movement_needed:
                # Move to starting joint positions with safe speed
                print("Moving to starting joint positions...")
                self.robot.movej(start_joints, acc=0.3, vel=0.2)
                
                # Wait for movement to complete
                time.sleep(1)
                while self.robot.is_program_running():
                    time.sleep(0.1)
                print("✓ Joint movement completed")
            else:
                print("✓ Robot already at target joint positions - no movement needed")
            
            # Get final joint positions and Cartesian pose with better error handling
            try:
                print("Getting final joint positions...")
                final_joints = self.robot.getj()
                print("Getting final Cartesian pose...")
                final_pose = self.get_robot_pose()  # Use helper method to handle PoseVector issues
                
                print(f"✓ Final joint positions: {[f'{math.degrees(j):.1f}°' for j in final_joints]}")
                print(f"✓ Final Cartesian pose: {[f'{x:.3f}' for x in final_pose]}")
                
                # Update base orientation from the final Cartesian pose
                self.base_orientation = [final_pose[3], final_pose[4], final_pose[5]]
                self.current_distance = abs(final_pose[1])  # Estimate distance from Y position
                
                print(f"✓ Camera ready for face tracking")
                print(f"✓ Base joint: {math.degrees(final_joints[0]):.1f}°, Wrist 3 joint: {math.degrees(final_joints[5]):.1f}°")
                
                return True
                
            except Exception as pose_error:
                print(f"Error getting final pose: {pose_error}")
                print(f"Error type: {type(pose_error)}")
                raise pose_error
            
        except Exception as e:
            print(f"Error moving to starting position: {e}")
            print(f"Error type: {type(e)}")
            print("Continuing with current position...")
            
            # Even if movement fails, try to continue with current position
            try:
                print("Attempting to get current position for fallback...")
                final_joints = self.robot.getj()
                print("Got joints, now getting pose...")
                final_pose = self.get_robot_pose()  # Use helper method to handle PoseVector issues
                print("Got pose successfully")
                
                self.base_orientation = [final_pose[3], final_pose[4], final_pose[5]]
                self.current_distance = abs(final_pose[1])
                
                print(f"✓ Using current position for face tracking")
                print(f"✓ Current joint positions: {[f'{math.degrees(j):.1f}°' for j in final_joints]}")
                return True
            except Exception as e2:
                print(f"Failed to get current position: {e2}")
                print(f"Error type: {type(e2)}")
                return False
    
    def detect_faces(self, frame):
        """Detect faces in the frame using MediaPipe (Google's face detection)."""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.face_detection.process(rgb_frame)
            
            # Convert MediaPipe detections to format similar to OpenCV
            faces = []
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to pixel coordinates
                    x = int(bbox.xmin * frame.shape[1])
                    y = int(bbox.ymin * frame.shape[0])
                    w = int(bbox.width * frame.shape[1])
                    h = int(bbox.height * frame.shape[0])
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    
                    # Add confidence score for better tracking
                    confidence = detection.score[0]
                    faces.append((x, y, w, h, confidence))
            
            return faces
            
        except Exception as e:
            print(f"Error detecting faces with MediaPipe: {e}")
            return []
    
    def calculate_face_center(self, faces):
        """Calculate the center of the most confident detected face."""
        if len(faces) == 0:
            return None
        
        # Find the face with highest confidence score
        # MediaPipe faces format: (x, y, w, h, confidence)
        if len(faces[0]) == 5:  # MediaPipe format with confidence
            best_face = max(faces, key=lambda face: face[4])  # Sort by confidence
            x, y, w, h, confidence = best_face
        else:  # Fallback to OpenCV format
            best_face = max(faces, key=lambda face: face[2] * face[3])  # Sort by area
            x, y, w, h = best_face
        
        # Calculate center of the face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        return (face_center_x, face_center_y, w, h)
    
    def calculate_movement(self, face_center):
        """Calculate required robot movement based on face position with smoothing."""
        if face_center is None:
            return None
        
        face_x, face_y, face_w, face_h = face_center
        
        # Calculate offset from frame center
        offset_x = face_x - self.frame_center_x
        offset_y = face_y - self.frame_center_y
        
        # Calculate distance from center using Pythagorean theorem
        distance_from_center = math.sqrt(offset_x**2 + offset_y**2)
        
        # Check if face is within the circular dead zone around center
        if distance_from_center <= self.center_dead_zone_radius:
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter % 30 == 0:  # Print every 30th frame when in dead zone
                print(f"Face within dead zone (distance: {distance_from_center:.1f} pixels, radius: {self.center_dead_zone_radius} pixels) - no movement sent")
            return None  # No movement needed, face is well-centered
        
        # Debug output (reduce frequency to avoid spam)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 10 == 0:  # Print every 10th frame
            print(f"Face at ({face_x}, {face_y}), offset: ({offset_x}, {offset_y}), distance from center: {distance_from_center:.1f}")
        
        # Apply dead zone to prevent jittery movements
        if abs(offset_x) < self.dead_zone_pixels:
            offset_x = 0
        if abs(offset_y) < self.dead_zone_pixels:
            offset_y = 0
        
        # Convert pixel offsets to robot movements
        # SWAPPED: Negative offset_x because camera is flipped (-1)
        # Positive offset_x = face is to the right on flipped camera, robot should move left
        move_x = -offset_x * self.movement_sensitivity  # SWAPPED for flipped camera
        move_z = -offset_y * self.movement_sensitivity  # Negative because Y is inverted
        
        # Limit maximum movement per step
        move_x = max(-self.max_movement_per_step, min(self.max_movement_per_step, move_x))
        move_z = max(-self.max_movement_per_step, min(self.max_movement_per_step, move_z))
        
        # Apply movement smoothing filter
        current_movement = [move_x, 0, move_z]
        
        # Blend with last movement for smoothing
        smoothed_movement = [
            self.last_movement[0] * self.movement_filter_strength + current_movement[0] * (1 - self.movement_filter_strength),
            0,  # No Y movement
            self.last_movement[2] * self.movement_filter_strength + current_movement[2] * (1 - self.movement_filter_strength)
        ]
        
        # Apply minimum threshold to prevent micro-movements
        if abs(smoothed_movement[0]) < self.min_movement_threshold:
            smoothed_movement[0] = 0
        if abs(smoothed_movement[2]) < self.min_movement_threshold:
            smoothed_movement[2] = 0
        
        # Store for next iteration
        self.last_movement = smoothed_movement.copy()
        
        if self._debug_counter % 10 == 0:  # Print every 10th frame
            print(f"Smoothed movement: left/right={smoothed_movement[0]:.4f}, up/down={smoothed_movement[2]:.4f}")
        
        return tuple(smoothed_movement)

# Example usage
if __name__ == "__main__":
    # Create face tracking camera system
    face_tracker = FaceTrackingCamera()
    
    try:
        # Start face tracking
        face_tracker.start_tracking()
        
    except KeyboardInterrupt:
        print("\nFace tracking interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        face_tracker.cleanup() 