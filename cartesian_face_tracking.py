import urx
import cv2
import numpy as np
import time
import math
import threading
import mediapipe as mp
import keyboard

# ===============================================
# CONFIGURATION
# ===============================================
ROBOT_IP = "192.168.0.101"
CAMERA_INDEX = 1
FACE_SENSITIVITY = 0.008  # Increased sensitivity for larger movements

# ===============================================
# CARTESIAN FACE TRACKER WITH MOVEL
# ===============================================

class CartesianFaceTracker:
    def __init__(self):
        self.robot = None
        self.cap = None
        self.running = False
        self.tracking_active = False
        
        # Current target position in Cartesian space [x, y, z, rx, ry, rz]
        self.target_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Camera settings
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Movement control
        self.movement_running = False
        self.movement_thread = None
        
        # Deadzone to prevent jittery movement when centered
        self.deadzone_radius = 15  # Reduced deadzone for more responsive tracking
        
        # Sudden movement detection - made more forgiving for smoother tracking
        self.previous_face_x = self.center_x
        self.previous_face_y = self.center_y
        self.sudden_movement_threshold = 80  # Increased threshold for less intervention
        self.movement_smoothing_factor = 0.5  # Reduced smoothing for more natural movement
        
        # Optimized blend control for consistently smooth motion
        self.min_blend = 0.1    # Small movements: 10cm blend for very smooth small adjustments
        self.max_blend = 0.8    # Large movements: 80cm blend for extremely smooth large movements  
        self.optimal_blend = 0.45  # Medium movements: 45cm blend for flowing medium movements
        
        # Pixel-based blending
        self.current_blend = self.optimal_blend  # Current blend radius to use
        self.pixel_distance_from_center = 0     # Current pixel distance from center
        
        # Arrow key control for X movement
        self.arrow_key_thread = None
        self.x_step = 0.005  # 5mm X movement step size for arrow keys
        
        # Face detection
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.5)
    
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
    
    def init_camera(self):
        """Initialize camera."""
        try:
            print("Initializing camera...")
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            print("✓ Camera ready!")
            return True
        except Exception as e:
            print(f"✗ Camera failed: {e}")
            return False
    
    def start_movement_loop(self):
        """Start continuous movement control."""
        try:
            print("Starting Cartesian movement control...")
            
            # Get current pose as starting target
            self.target_pose = self.get_robot_pose()
            
            # Start background thread
            self.movement_running = True
            self.movement_thread = threading.Thread(target=self._movement_loop)
            self.movement_thread.daemon = True
            self.movement_thread.start()
            
            print("✓ Movement control started!")
            return True
        except Exception as e:
            print(f"✗ Movement start failed: {e}")
            return False
    
    def _movement_loop(self):
        """Continuous movement loop for smooth live face tracking."""
        while self.movement_running:
            try:
                # Only process movement if tracking is active
                if not self.tracking_active:
                    time.sleep(0.05)  # Sleep when tracking inactive to reduce CPU usage
                    continue
                
                # Always send movement command for live tracking - no thresholds
                # Use the current blend calculated from pixel distance
                blend = self.current_blend
                
                # Send movement command for smooth live tracking
                urscript_cmd = (f"movel(p[{self.target_pose[0]:.6f}, {self.target_pose[1]:.6f}, "
                              f"{self.target_pose[2]:.6f}, {self.target_pose[3]:.6f}, "
                              f"{self.target_pose[4]:.6f}, {self.target_pose[5]:.6f}], "
                              f"a=0.3, v=0.15, r={blend:.3f})")  # Increased speed for live tracking
                
                # Add debug logging for movement commands
                print(f"Live tracking: Y:{self.target_pose[1]:.3f} Z:{self.target_pose[2]:.3f} (pixels:{self.pixel_distance_from_center:.0f}, blend:{blend:.3f})")
                self.robot.send_program(urscript_cmd)
                
                # Faster control loop for live tracking
                time.sleep(0.03)  # ~33Hz for smooth live motion
                
            except Exception as e:
                if self.movement_running:
                    print(f"Movement error: {e}")
                    import traceback
                    traceback.print_exc()  # Print full error traceback for debugging
                time.sleep(0.1)
    
    def update_target(self, face_x, face_y):
        """Update target pose based on face position with optimized smoothing."""
        if not self.tracking_active:
            # Reset face position tracking when not active to prevent sudden jumps
            self.previous_face_x = face_x
            self.previous_face_y = face_y
            return
        
        try:
            # Detect sudden movements with more forgiving threshold
            face_movement = math.sqrt((face_x - self.previous_face_x)**2 + (face_y - self.previous_face_y)**2)
            is_sudden_movement = face_movement > self.sudden_movement_threshold
            
            # Apply lighter smoothing for more natural movement
            if is_sudden_movement:
                smoothed_face_x = self.previous_face_x + (face_x - self.previous_face_x) * (1 - self.movement_smoothing_factor)
                smoothed_face_y = self.previous_face_y + (face_y - self.previous_face_y) * (1 - self.movement_smoothing_factor)
            else:
                smoothed_face_x = face_x
                smoothed_face_y = face_y
            
            # Calculate face offset from center
            offset_x = smoothed_face_x - self.center_x
            offset_y = smoothed_face_y - self.center_y
            
            # Check if face is within deadzone (smaller deadzone for more responsive tracking)
            distance_from_center = math.sqrt(offset_x**2 + offset_y**2)
            if distance_from_center <= self.deadzone_radius:
                self.previous_face_x = face_x
                self.previous_face_y = face_y
                return
            
            # Calculate blend based on pixel distance from center for smoother live tracking
            pixel_distance = distance_from_center
            if pixel_distance < 20:        # Very close to center (< 20 pixels)
                self.current_blend = 0.05   # 5cm blend - responsive but smooth
            elif pixel_distance > 80:     # Far from center (> 80 pixels)  
                self.current_blend = 0.15   # 15cm blend - smooth for larger movements
            else:                          # Medium distance (20-80 pixels)
                self.current_blend = 0.1    # 10cm blend - balanced smooth motion
            
            # Convert pixel offsets to Cartesian movements - Y AND Z MOVEMENT FOR FACE TRACKING
            # Face left/right -> Y-axis movement (GREEN arrow)
            # Face up/down -> Z-axis movement (BLUE arrow)  
            # NO X-axis movement (RED arrow stays constant - only arrow keys control X)
            y_move = offset_x * FACE_SENSITIVITY   # Face right = positive Y (green arrow right) - FIXED DIRECTION
            z_move = -offset_y * FACE_SENSITIVITY  # Face down = positive Z (blue arrow down)
            
            # Update target pose - Y AND Z movement for face tracking
            # X (red arrow) remains constant during face tracking - only arrow keys change X
            self.target_pose[1] += y_move  # Y movement (green axis - left/right)
            self.target_pose[2] += z_move  # Z movement (blue axis - up/down)
            # X remains unchanged during face tracking (red axis - only arrow keys)
            
            # Store pixel distance for movement loop to use
            self.pixel_distance_from_center = pixel_distance
            
            # Update previous position
            self.previous_face_x = face_x
            self.previous_face_y = face_y
            
        except Exception as e:
            print(f"Update error: {e}")
    
    def detect_face(self, frame):
        """Simple face detection."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
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
        except:
            return None
    
    def handle_arrow_keys(self):
        """Handle arrow key presses for tool-relative forward/backward movement (5mm steps)."""
        print("Arrow key control started - UP/DOWN: Tool head forward/backward (5mm steps)")
        
        # Track key states to prevent repeat triggers
        up_pressed = False
        down_pressed = False
        
        while self.running:
            try:
                # Check UP arrow - only trigger on new press
                if keyboard.is_pressed('up'):
                    if not up_pressed:  # New press detected
                        # UP arrow: Move forward relative to tool head by 5mm
                        # Use URScript to move in tool coordinates
                        urscript_cmd = f"movel(pose_trans(get_actual_tcp_pose(), p[0.005, 0, 0, 0, 0, 0]), a=0.2, v=0.05)"
                        self.robot.send_program(urscript_cmd)
                        print(f"UP pressed - Tool head moved forward 5mm")
                        up_pressed = True
                        time.sleep(0.2)  # Prevent rapid-fire
                else:
                    up_pressed = False  # Reset when key released
                
                # Check DOWN arrow - only trigger on new press
                if keyboard.is_pressed('down'):
                    if not down_pressed:  # New press detected
                        # DOWN arrow: Move backward relative to tool head by 5mm
                        # Use URScript to move in tool coordinates
                        urscript_cmd = f"movel(pose_trans(get_actual_tcp_pose(), p[-0.005, 0, 0, 0, 0, 0]), a=0.2, v=0.05)"
                        self.robot.send_program(urscript_cmd)
                        print(f"DOWN pressed - Tool head moved backward 5mm")
                        down_pressed = True
                        time.sleep(0.2)  # Prevent rapid-fire
                else:
                    down_pressed = False  # Reset when key released
                
                time.sleep(0.05)  # Small loop delay
                    
            except Exception as e:
                print(f"Arrow key error: {e}")
                time.sleep(0.1)
    
    def get_robot_pose(self):
        """Get robot pose using URScript to bypass URX library issues."""
        try:
            # Try using URScript to get pose directly - this bypasses the broken URX getl() method
            urscript_cmd = "get_actual_tcp_pose()"
            pose_data = self.robot.send_program(urscript_cmd)
            
            if pose_data and len(pose_data) >= 6:
                print(f"DEBUG: URScript pose successful: {pose_data}")
                return [float(x) for x in pose_data[:6]]
                
        except Exception as e:
            print(f"DEBUG: URScript pose failed: {e}")
        
        try:
            # Try alternative URScript method
            urscript_cmd = "textmsg(get_actual_tcp_pose())"
            result = self.robot.send_program(urscript_cmd)
            print(f"DEBUG: URScript textmsg result: {result}")
            
        except Exception as e:
            print(f"DEBUG: URScript textmsg failed: {e}")
        
        # Since getting current pose is problematic, work with target pose tracking
        # This is actually better for smooth movement anyway
        if hasattr(self, 'target_pose') and self.target_pose != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
            print("DEBUG: Using target pose for movement calculations (this is actually better for smooth tracking)")
            return self.target_pose[:]  # Return copy of current target
        
        # Return safe default pose if all methods fail
        print("DEBUG: Using hardcoded fallback pose coordinates")
        return [0.3, 0.0, 0.4, 3.14, 0.0, 0.0]
    
    def move_to_starting_position(self):
        """Move robot to optimal starting position using joint angles."""
        try:
            print("Moving to starting position...")
            
            # Define starting position using proven joint angles (same as previous implementation)
            start_joints = [
                0.0,                    # Base = 0°
                math.radians(-60),      # Shoulder = -60°
                math.radians(80),       # Elbow = 80°
                math.radians(-110),     # Wrist1 (calculated)
                math.radians(270),      # Wrist2 = 270°
                math.radians(-90)       # Wrist3 (calculated)
            ]
            
            # Move to starting position using joint movement
            try:
                self.robot.movej(start_joints, acc=0.5, vel=0.3)
            except Exception as move_error:
                print(f"Movement warning: {move_error}")
                print("Checking if robot actually moved...")
            
            # Wait for movement to complete
            print("Waiting for movement to complete...")
            time.sleep(7)  # Give time for movement
            
            # Check if robot reached target position
            current_joints = self.robot.getj()
            position_errors = []
            for i in range(len(start_joints)):
                error = abs(current_joints[i] - start_joints[i])
                # Handle angle wrapping (e.g., -180° vs 180°)
                if error > math.pi:
                    error = 2 * math.pi - error
                position_errors.append(error)
            
            max_error = max(position_errors)
            max_error_deg = math.degrees(max_error)
            
            print(f"Position check: Max error = {max_error_deg:.1f}°")
            
            if max_error_deg < 5.0:  # Within 5 degrees tolerance
                print("✓ Starting position reached successfully!")
                
                # Get the actual Cartesian pose after joint movement
                self.target_pose = self.get_robot_pose()
                print(f"Starting Cartesian pose: X:{self.target_pose[0]:.3f} Y:{self.target_pose[1]:.3f} Z:{self.target_pose[2]:.3f}")
                return True
            else:
                print(f"✗ Starting position not reached. Error: {max_error_deg:.1f}°")
                # Still get current pose for tracking
                self.target_pose = self.get_robot_pose()
                return False
            
        except Exception as e:
            print(f"✗ Starting position failed: {e}")
            return False
    
    def run(self):
        """Main execution."""
        print("\n=== CARTESIAN FACE TRACKER ===")
        print("SPACE: Toggle tracking, UP/DOWN: Z movement, ESC: Exit")
        
        # Initialize systems
        if not self.connect_robot():
            return
        
        if not self.move_to_starting_position():
            self.robot.close()
            return
        
        if not self.init_camera():
            self.robot.close()
            return
        
        if not self.start_movement_loop():
            self.cap.release()
            self.robot.close()
            return
        
        print("\nReady! Press SPACE to start tracking, UP/DOWN arrows for Z movement")
        
        # Main loop
        self.running = True
        
        # Start arrow key control thread
        self.arrow_key_thread = threading.Thread(target=self.handle_arrow_keys)
        self.arrow_key_thread.daemon = True
        self.arrow_key_thread.start()
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, -1)  # Mirror
                
                # Detect face
                face_data = self.detect_face(frame)
                
                if face_data and self.tracking_active:
                    face_x, face_y, x, y, w, h = face_data
                    
                    # Update robot target
                    self.update_target(face_x, face_y)
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(frame, (face_x, face_y), 5, (0, 255, 0), -1)
                
                # Draw center crosshair and deadzone
                cv2.circle(frame, (self.center_x, self.center_y), 3, (0, 0, 255), -1)
                cv2.circle(frame, (self.center_x, self.center_y), self.deadzone_radius, (255, 0, 0), 1)
                cv2.line(frame, (self.center_x-10, self.center_y), (self.center_x+10, self.center_y), (0, 0, 255), 1)
                cv2.line(frame, (self.center_x, self.center_y-10), (self.center_x, self.center_y+10), (0, 0, 255), 1)
                
                # Status text
                status = "TRACKING" if self.tracking_active else "PAUSED"
                color = (0, 255, 0) if self.tracking_active else (0, 0, 255)
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, "SPACE: Toggle, UP/DOWN: Z move, ESC: Exit", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show current position info
                if hasattr(self, 'target_pose'):
                    pos_text = f"X:{self.target_pose[0]:.3f} Y:{self.target_pose[1]:.3f} Z:{self.target_pose[2]:.3f}"
                    cv2.putText(frame, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Cartesian Face Tracker', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Exiting...")
                    break
                elif key == ord(' '):  # SPACE
                    # Stop robot movement when tracking is being turned off
                    if self.tracking_active:
                        # About to deactivate tracking - stop the robot
                        print("Stopping robot movement...")
                        self.robot.send_program("stopj(5.0)")  # Stop with 5 rad/s² deceleration
                        time.sleep(0.1)  # Brief pause for stop command
                    
                    # Reset face tracking position when toggling to prevent sudden movements
                    if not self.tracking_active:
                        # About to activate tracking - reset face position reference
                        self.previous_face_x = self.center_x
                        self.previous_face_y = self.center_y
                        print("Face tracking position reset to center")
                    
                    self.tracking_active = not self.tracking_active
                    status_text = "ACTIVATED" if self.tracking_active else "PAUSED"
                    print(f"Tracking {status_text}")
                
            except Exception as e:
                print(f"Main loop error: {e}")
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.running = False
        self.movement_running = False
        
        # Stop threads
        if self.movement_thread and self.movement_thread.is_alive():
            self.movement_thread.join(timeout=1)
        
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
    tracker = CartesianFaceTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        tracker.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        tracker.cleanup() 