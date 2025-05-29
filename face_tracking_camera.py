import urx
import cv2
import numpy as np
import time
import math
import threading
import keyboard

class FaceTrackingCamera:
    """
    Face tracking camera system for UR10e robot.
    
    Features:
    - Tracks face left/right and up/down movements
    - Maintains stable distance (no forward/backward following)
    - Manual distance control with up/down arrow keys
    - Camera mounted on tool pointing along blue arrow (Z-axis)
    """
    
    def __init__(self, robot_ip="192.168.10.152", camera_index=0):
        """Initialize the face tracking camera system."""
        self.robot = None
        self.robot_ip = robot_ip
        self.camera_index = camera_index
        self.cap = None
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
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
        """Move robot to optimal starting position for face tracking."""
        try:
            print("Moving to face tracking starting position...")
            
            # Get current position as reference
            current_pose = [self.robot.x, self.robot.y, self.robot.z, 
                           self.robot.rx, self.robot.ry, self.robot.rz]
            
            # Define optimal starting position
            # Position camera at a good height and distance for face tracking
            start_x = current_pose[0]  # Keep current X
            start_y = current_pose[1]   # Move 50cm forward (toward user)
            start_z = current_pose[2] + 0.2  # Move 20cm up for face level
            start_rx = 0.0  # Camera pointing forward
            start_ry = 0.0
            start_rz = 0.0
            
            print(f"Moving from: {[f'{x:.3f}' for x in current_pose[:3]]}")
            print(f"Moving to:   [{start_x:.3f}, {start_y:.3f}, {start_z:.3f}]")
            
            # Move to starting position with faster speed
            urscript_cmd = f"movel(p[{start_x}, {start_y}, {start_z}, {start_rx}, {start_ry}, {start_rz}], a=0.3, v=0.2)"
            self.robot.send_program(urscript_cmd)
            
            # Wait for movement to complete
            print("Moving to starting position...")
            time.sleep(1)
            while self.robot.is_program_running():
                time.sleep(0.1)
            
            # Update base orientation and current distance
            self.base_orientation = [start_rx, start_ry, start_rz]
            self.current_distance = abs(start_y)
            
            # Verify position
            final_pose = [self.robot.x, self.robot.y, self.robot.z, 
                         self.robot.rx, self.robot.ry, self.robot.rz]
            print(f"✓ Starting position reached: {[f'{x:.3f}' for x in final_pose[:3]]}")
            print(f"✓ Camera ready for face tracking at {self.current_distance:.2f}m distance")
            
            return True
            
        except Exception as e:
            print(f"Error moving to starting position: {e}")
            return False
    
    def detect_faces(self, frame):
        """Detect faces in the frame."""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50)
            )
            
            return faces
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def calculate_face_center(self, faces):
        """Calculate the center of the largest detected face."""
        if len(faces) == 0:
            return None
        
        # Find the largest face (closest to camera)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
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
        
        return (move_x, 0, move_z)  # No Y movement (depth)
    
    def move_robot_smooth(self, movement):
        """Move robot smoothly to track face."""
        if movement is None or not self.tracking_active:
            return
        
        try:
            move_x, move_y, move_z = movement
            
            # Get current position
            current_x = self.robot.x
            current_y = self.robot.y
            current_z = self.robot.z
            
            # Calculate new position with smoothing
            new_x = current_x + move_x * self.movement_smoothing
            new_z = current_z + move_z * self.movement_smoothing
            
            # Keep Y at stable distance (no depth following)
            new_y = current_y
            
            # Use current orientation (camera always pointing forward)
            new_rx, new_ry, new_rz = self.base_orientation
            
            # Move robot using URScript with faster speeds for tracking
            urscript_cmd = f"movel(p[{new_x}, {new_y}, {new_z}, {new_rx}, {new_ry}, {new_rz}], a=0.5, v=0.3)"
            self.robot.send_program(urscript_cmd)
            
        except Exception as e:
            print(f"Error moving robot: {e}")
    
    def handle_distance_control(self):
        """Handle manual distance control with arrow keys."""
        while self.running:
            try:
                if keyboard.is_pressed('up'):
                    # Move closer (decrease Y - move toward face)
                    self.adjust_distance(-0.03)  # Increased from 0.02 for faster movement
                    time.sleep(0.08)  # Reduced delay for more responsive control
                elif keyboard.is_pressed('down'):
                    # Move further (increase Y - move away from face)
                    self.adjust_distance(0.03)  # Increased from 0.02 for faster movement
                    time.sleep(0.08)  # Reduced delay for more responsive control
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"Error in distance control: {e}")
                time.sleep(0.1)
    
    def adjust_distance(self, distance_change):
        """Adjust the distance to the face using tool-relative movement."""
        try:
            # Get current pose
            current_pose = [self.robot.x, self.robot.y, self.robot.z, 
                           self.robot.rx, self.robot.ry, self.robot.rz]
            
            # Calculate tool-relative movement along Z-axis (blue arrow direction)
            # This ensures movement is always along the camera's viewing direction
            # regardless of tool orientation
            
            # Create transformation matrix for current tool orientation
            rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]
            
            # Calculate rotation matrix from tool orientation
            # For UR robots, the tool Z-axis (blue arrow) points forward
            cos_rx, sin_rx = math.cos(rx), math.sin(rx)
            cos_ry, sin_ry = math.cos(ry), math.sin(ry)
            cos_rz, sin_rz = math.cos(rz), math.sin(rz)
            
            # Tool Z-axis direction vector (blue arrow direction)
            # This represents the direction the camera is pointing
            z_axis_x = -sin_ry
            z_axis_y = sin_rx * cos_ry
            z_axis_z = cos_rx * cos_ry
            
            # Calculate movement in global coordinates along tool Z-axis
            move_x = z_axis_x * distance_change
            move_y = z_axis_y * distance_change
            move_z = z_axis_z * distance_change
            
            # Calculate new position
            new_x = current_pose[0] + move_x
            new_y = current_pose[1] + move_y
            new_z = current_pose[2] + move_z
            
            # Keep the same orientation
            new_rx, new_ry, new_rz = current_pose[3], current_pose[4], current_pose[5]
            
            # Move robot with tool-relative movement
            urscript_cmd = f"movel(p[{new_x}, {new_y}, {new_z}, {new_rx}, {new_ry}, {new_rz}], a=0.3, v=0.15)"
            self.robot.send_program(urscript_cmd)
            
            # Update current distance (calculate actual distance from starting position)
            start_pos = [current_pose[0] - move_x, current_pose[1] - move_y, current_pose[2] - move_z]
            self.current_distance = math.sqrt(move_x**2 + move_y**2 + move_z**2)
            
            direction = "closer" if distance_change < 0 else "further"
            print(f"Moving {direction} along camera axis - Distance change: {abs(distance_change):.3f}m")
            
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
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process face tracking
                if len(faces) > 0 and self.tracking_active:
                    face_center = self.calculate_face_center(faces)
                    movement = self.calculate_movement(face_center)
                    self.move_robot_smooth(movement)
                    
                    # Draw face detection
                    for (x, y, w, h) in faces:
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
            print("- Tracks face left/right and up/down movements")
            print("- Maintains stable distance (no depth following)")
            print("- UP arrow: Move closer, DOWN arrow: Move further")
            print("- SPACE: Toggle tracking, ESC: Exit")
            print("- Faster movement speeds for better responsiveness")
            
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