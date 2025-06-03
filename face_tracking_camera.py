import urx
import cv2
import numpy as np
import time
import math
import threading
import keyboard
import mediapipe as mp

class FaceTrackingCamera:
    """
    Face tracking camera system for UR10e robot using Google's MediaPipe.
    
    Features:
    - Red arrow automatically points directly at user's face (automatic orientation)
    - Face tracking up/down controls blue arrow (Z-axis) movements
    - Face tracking left/right controls green arrow (Y-axis) movements
    - Arrow keys control distance along red arrow (toward/away from user)
    - Tool-relative movements ensure consistent behavior regardless of tool orientation
    - Google MediaPipe for smooth and accurate face detection
    """
    
    def __init__(self, robot_ip="192.168.0.100", camera_index=0):
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
        
        # Movement parameters - increased for faster response
        self.movement_sensitivity = 0.002  # Increased from 0.001 for faster response
        self.max_movement_per_step = 0.05  # Increased from 0.02 for faster movement
        self.movement_smoothing = 0.7  # Increased from 0.3 for more responsive movement
        
        # Camera frame parameters
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Dead zone to prevent jittery movements
        self.dead_zone_pixels = 15  # Reduced from 20 for more responsive tracking
        
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
    
    def setup_camera_position(self):
        """Set up initial camera position and orientation."""
        try:
            print("Setting up camera position...")
            
            # Get current position
            self.current_pose = [self.robot.x, self.robot.y, self.robot.z, 
                               self.robot.rx, self.robot.ry, self.robot.rz]
            
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
            wrist1_angle_rad = math.radians(wrist1_angle_deg)
            
            print(f"Wrist1 calculation: 90° + {elbow_angle_deg}° - abs({shoulder_angle_deg}°) = {wrist1_angle_deg}°")
            
            start_joints = [
                0.0,                    # Base joint (joint 0) = 0 degrees
                shoulder_angle_rad,     # Shoulder joint (joint 1) = 60 degrees
                elbow_angle_rad,        # Elbow joint (joint 2) = 120 degrees
                wrist1_angle_rad,       # Wrist 1 joint (joint 3) calculated using formula
                3*math.pi/2,            # Keep current wrist 2 joint (or set to a specific value, e.g., math.pi/2)
                -math.pi/2              # Wrist 3 joint (joint 5) = -90 degrees
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
            
            # Get final joint positions and Cartesian pose
            final_joints = self.robot.getj()
            final_pose = [self.robot.x, self.robot.y, self.robot.z, 
                         self.robot.rx, self.robot.ry, self.robot.rz]
            
            print(f"✓ Final joint positions: {[f'{math.degrees(j):.1f}°' for j in final_joints]}")
            print(f"✓ Final Cartesian pose: {[f'{x:.3f}' for x in final_pose]}")
            
            # Update base orientation from the final Cartesian pose
            self.base_orientation = [final_pose[3], final_pose[4], final_pose[5]]
            self.current_distance = abs(final_pose[1])  # Estimate distance from Y position
            
            print(f"✓ Camera ready for face tracking")
            print(f"✓ Base joint: {math.degrees(final_joints[0]):.1f}°, Wrist 3 joint: {math.degrees(final_joints[5]):.1f}°")
            
            return True
            
        except Exception as e:
            print(f"Error moving to starting position: {e}")
            print("Continuing with current position...")
            
            # Even if movement fails, try to continue with current position
            try:
                final_joints = self.robot.getj()
                final_pose = [self.robot.x, self.robot.y, self.robot.z, 
                             self.robot.rx, self.robot.ry, self.robot.rz]
                
                self.base_orientation = [final_pose[3], final_pose[4], final_pose[5]]
                self.current_distance = abs(final_pose[1])
                
                print(f"✓ Using current position for face tracking")
                print(f"✓ Current joint positions: {[f'{math.degrees(j):.1f}°' for j in final_joints]}")
                return True
            except Exception as e2:
                print(f"Failed to get current position: {e2}")
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
        """Calculate required robot movement based on face position."""
        if face_center is None:
            return None
        
        face_x, face_y, face_w, face_h = face_center
        
        # Calculate offset from frame center
        offset_x = face_x - self.frame_center_x
        offset_y = face_y - self.frame_center_y
        
        # Temporary debug for face movement
        print(f"Face at ({face_x}, {face_y}), offset: ({offset_x}, {offset_y})")
        
        # Apply dead zone to prevent jittery movements
        if abs(offset_x) < self.dead_zone_pixels:
            offset_x = 0
        if abs(offset_y) < self.dead_zone_pixels:
            offset_y = 0
        
        # Convert pixel offsets to robot movements
        # Positive offset_x = face is to the right, robot should move right
        # Positive offset_y = face is down, robot should move down
        move_x = offset_x * self.movement_sensitivity
        move_z = -offset_y * self.movement_sensitivity  # Negative because Y is inverted
        
        # Limit maximum movement per step
        move_x = max(-self.max_movement_per_step, min(self.max_movement_per_step, move_x))
        move_z = max(-self.max_movement_per_step, min(self.max_movement_per_step, move_z))
        
        print(f"Movement: left/right={move_x:.4f}, up/down={move_z:.4f}")
        
        return (move_x, 0, move_z)  # No Y movement (depth)
    
    def move_robot_smooth(self, movement):
        """Move robot smoothly to track face using .translate() to preserve orientation and joint relationships."""
        if movement is None or not self.tracking_active:
            return
        
        try:
            move_x, move_y, move_z = movement
            
            # Get current pose for reference
            current_pose = [self.robot.x, self.robot.y, self.robot.z, 
                           self.robot.rx, self.robot.ry, self.robot.rz]
            
            print(f"Tool orientation: RX={math.degrees(current_pose[3]):.1f}°, RY={math.degrees(current_pose[4]):.1f}°, RZ={math.degrees(current_pose[5]):.1f}°")
            
            # Calculate tool-relative movements for face tracking using green and blue arrows
            # Green arrow (Y-axis) for left/right movement as seen by camera
            # Blue arrow (Z-axis) for up/down movement as seen by camera
            rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]
            
            # Calculate rotation matrix components
            cos_rx, sin_rx = math.cos(rx), math.sin(rx)
            cos_ry, sin_ry = math.cos(ry), math.sin(ry)
            cos_rz, sin_rz = math.cos(rz), math.sin(rz)
            
            # Tool coordinate system vectors
            # For UR robots with standard tool orientation:
            # X-axis (red arrow) - forward direction
            # Y-axis (green arrow) - left/right direction  
            # Z-axis (blue arrow) - up/down direction
            
            # Y-axis (green arrow) - left/right movement as seen by camera
            y_axis_x = -sin_rz
            y_axis_y = cos_rz
            y_axis_z = 0
            
            # Z-axis (blue arrow) - up/down movement as seen by camera
            z_axis_x = -cos_rz * sin_ry
            z_axis_y = -sin_rz * sin_ry
            z_axis_z = cos_ry
            
            print(f"Green arrow direction: ({y_axis_x:.3f}, {y_axis_y:.3f}, {y_axis_z:.3f})")
            print(f"Blue arrow direction: ({z_axis_x:.3f}, {z_axis_y:.3f}, {z_axis_z:.3f})")
            
            # Apply smoothing to movements
            smooth_move_x = move_x * self.movement_smoothing  # Camera left/right -> Green arrow
            smooth_move_z = move_z * self.movement_smoothing  # Camera up/down -> Blue arrow
            
            # Calculate global movements from tool-relative movements
            # Use green arrow for left/right, blue arrow for up/down
            global_move_x = y_axis_x * smooth_move_x + z_axis_x * smooth_move_z
            global_move_y = y_axis_y * smooth_move_x + z_axis_y * smooth_move_z
            global_move_z = y_axis_z * smooth_move_x + z_axis_z * smooth_move_z
            
            print(f"Global movement: X={global_move_x:.4f}, Y={global_move_y:.4f}, Z={global_move_z:.4f}")
            
            # Use .translate() to move while preserving orientation and joint relationships
            # This maintains the wrist1 formula: 90° + elbow_angle - abs(shoulder_angle)
            self.robot.translate((global_move_x, global_move_y, global_move_z), acc=0.5, vel=0.2)
            
            print(f"Using .translate() to preserve joint relationships and orientation")
            
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
            
            # Get current position
            current_pose = [self.robot.x, self.robot.y, self.robot.z, 
                           self.robot.rx, self.robot.ry, self.robot.rz]
            
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
        """Handle manual distance control with arrow keys along red arrow (X-axis)."""
        while self.running:
            try:
                if keyboard.is_pressed('up'):
                    # Move closer along red arrow (X-axis toward user)
                    self.adjust_distance(-0.03)  # Increased from 0.02 for faster movement
                    time.sleep(0.08)  # Reduced delay for more responsive control
                elif keyboard.is_pressed('down'):
                    # Move further along red arrow (X-axis away from user)
                    self.adjust_distance(0.03)  # Increased from 0.02 for faster movement
                    time.sleep(0.08)  # Reduced delay for more responsive control
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"Error in distance control: {e}")
                time.sleep(0.1)
    
    def adjust_distance(self, distance_change):
        """Adjust the distance to the face using .translate() along red arrow (X-axis) to preserve joint relationships."""
        try:
            # Get current pose for reference
            current_pose = [self.robot.x, self.robot.y, self.robot.z, 
                           self.robot.rx, self.robot.ry, self.robot.rz]
            
            # Calculate tool-relative movement along X-axis (red arrow direction)
            # This ensures movement is always along the red arrow which points toward user
            # regardless of tool orientation
            
            # Create transformation matrix for current tool orientation
            rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]
            
            # Calculate rotation matrix from tool orientation
            # For UR robots, the tool X-axis (red arrow) should point toward user
            cos_rx, sin_rx = math.cos(rx), math.sin(rx)
            cos_ry, sin_ry = math.cos(ry), math.sin(ry)
            cos_rz, sin_rz = math.cos(rz), math.sin(rz)
            
            # Tool X-axis direction vector (red arrow direction)
            # This represents the direction toward/away from the user
            x_axis_x = cos_ry * cos_rz
            x_axis_y = cos_ry * sin_rz
            x_axis_z = -sin_ry
            
            # Calculate movement in global coordinates along tool X-axis (red arrow)
            move_x = x_axis_x * distance_change
            move_y = x_axis_y * distance_change
            move_z = x_axis_z * distance_change
            
            # Use .translate() to move while preserving orientation and joint relationships
            # This maintains the wrist1 formula: 90° + elbow_angle - abs(shoulder_angle)
            self.robot.translate((move_x, move_y, move_z), acc=0.3, vel=0.15)
            
            # Update current distance (calculate actual distance from starting position)
            self.current_distance = math.sqrt(move_x**2 + move_y**2 + move_z**2)
            
            direction = "closer" if distance_change < 0 else "further"
            print(f"Moving {direction} along red arrow using .translate() - Distance change: {abs(distance_change):.3f}m")
            print(f"Preserving wrist1 joint relationship: 90° + elbow_angle - abs(shoulder_angle)")
            
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
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in camera loop: {e}")
                time.sleep(0.1)
    
    def start_tracking(self):
        """Start the face tracking system."""
        try:
            print("\n=== Face Tracking Camera System ===")
            print("Features:")
            print("- Red arrow automatically points directly at user's face (automatic orientation)")
            print("- Face tracking up/down controls blue arrow (Z-axis) movements")
            print("- Face tracking left/right controls green arrow (Y-axis) movements")
            print("- Arrow keys control distance along red arrow (toward/away from user)")
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
            print("Red arrow will automatically point at your face")
            print("Move your face up/down → Blue arrow movement")
            print("Move your face left/right → Green arrow movement")
            print("Arrow keys → Move closer/further along red arrow")
            
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
            # Get current robot position
            robot_pos = [self.robot.x, self.robot.y, self.robot.z]
            
            # Estimate face position in 3D space
            # This is a simplified estimation - in reality you'd need depth info
            face_x, face_y, face_w, face_h = face_center
            
            # Convert camera pixel coordinates to estimated 3D position
            # Assume face is at a certain distance and convert pixel offset to world offset
            estimated_face_distance = 0.8  # Assume face is 80cm away
            
            # Calculate face offset from camera center in world coordinates
            pixel_to_world_scale = 0.001  # Rough conversion factor
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