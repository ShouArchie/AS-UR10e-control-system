import urx
import cv2
import numpy as np
import time
import math
import threading
import mediapipe as mp
import keyboard

# ===============================================
# SIMPLE CONFIGURATION
# ===============================================
ROBOT_IP = "192.168.0.101"
CAMERA_INDEX = 1
FACE_SENSITIVITY = 0.01  # Movement sensitivity

# ===============================================
# SIMPLE FACE TRACKER
# ===============================================

class SimpleFaceTracker:
    def __init__(self):
        self.robot = None
        self.cap = None
        self.running = False
        self.tracking_active = False
        
        # Current target joints (start with safe position)
        self.target_joints = [0.0, -1.047, 1.396, -1.919, 4.712, -1.571]  # radians
        
        # Camera settings
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # ServJ control
        self.servoj_running = False
        self.servoj_thread = None
        
        # Dynamic control parameters
        self.current_face_x = self.center_x
        self.current_face_y = self.center_y
        self.current_t = 1.5
        self.current_a = 0.35
        self.current_v = 0.45
        self.current_lookahead = 1.35
        self.current_gain = 300
        
        # Sudden movement detection
        self.previous_face_x = self.center_x
        self.previous_face_y = self.center_y
        self.sudden_movement_threshold = 40  # pixels - made more sensitive
        self.movement_smoothing_factor = 0.85  # Increased from 0.7 for more smoothing
        
        # Deadzone to prevent jittery movement when centered
        self.deadzone_radius = 20  # pixels - robot stops moving when face is within this radius
        
        # PID Controller parameters
        self.pid_kp = 0.8  # Proportional gain
        self.pid_ki = 0.1  # Integral gain  
        self.pid_kd = 0.3  # Derivative gain
        
        # PID state variables
        self.pid_error_integral = [0.0] * 6  # For each joint
        self.pid_previous_error = [0.0] * 6  # For each joint
        self.joint_velocities = [0.0] * 6    # Current joint velocities
        self.max_joint_velocity = 0.5       # Max velocity (rad/s)
        self.velocity_smoothing = 0.7        # Smooth velocity transitions
        
        # Arrow key control for Z movement
        self.arrow_key_thread = None
        self.z_step = 0.05  # Z movement step size
        
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
    
    def start_servoj_loop(self):
        """Start continuous servoj at 125Hz."""
        try:
            print("Starting continuous ServoJ at 125Hz...")
            
            # Get current position as starting target
            current = self.robot.getj()
            self.target_joints = [current[i] for i in range(6)]
            
            # Start background thread
            self.servoj_running = True
            self.servoj_thread = threading.Thread(target=self._servoj_loop)
            self.servoj_thread.daemon = True
            self.servoj_thread.start()
            
            print("✓ ServoJ loop started!")
            return True
        except Exception as e:
            print(f"✗ ServoJ start failed: {e}")
            return False
    
    def _servoj_loop(self):
        """Continuous servoj loop - sends commands every 0.016 seconds."""
        while self.servoj_running:
            try:
                # Create servoj command with dynamic parameters
                cmd = (f"servoj([{self.target_joints[0]:.6f}, {self.target_joints[1]:.6f}, "
                       f"{self.target_joints[2]:.6f}, {self.target_joints[3]:.6f}, "
                       f"{self.target_joints[4]:.6f}, {self.target_joints[5]:.6f}], "
                       f"a={self.current_a:.2f}, v={self.current_v:.2f}, t={self.current_t:.3f}, "
                       f"lookahead_time={self.current_lookahead:.2f}, gain={self.current_gain})")
                
                # Send command
                self.robot.send_program(cmd)
                
                # Match camera fps (~60Hz)
                time.sleep(0.016)
                
            except Exception as e:
                if self.servoj_running:
                    print(f"ServoJ error: {e}")
                break
    
    def calculate_pid_velocities(self):
        """Calculate joint velocities using PID control."""
        try:
            # Get current joint positions
            current_joints = self.robot.getj()
            target_velocities = [0.0] * 6
            
            # Calculate PID for each joint
            for i in range(6):
                # Error calculation
                error = self.target_joints[i] - current_joints[i]
                
                # Handle angle wrapping for continuous joints
                if abs(error) > math.pi:
                    error = error - 2 * math.pi * (1 if error > 0 else -1)
                
                # PID calculation
                self.pid_error_integral[i] += error
                error_derivative = error - self.pid_previous_error[i]
                
                # Integral windup protection
                if abs(self.pid_error_integral[i]) > 0.5:
                    self.pid_error_integral[i] = 0.5 * (1 if self.pid_error_integral[i] > 0 else -1)
                
                # PID output (desired velocity)
                velocity = (self.pid_kp * error + 
                           self.pid_ki * self.pid_error_integral[i] + 
                           self.pid_kd * error_derivative)
                
                # Limit velocity
                velocity = max(-self.max_joint_velocity, min(self.max_joint_velocity, velocity))
                target_velocities[i] = velocity
                
                # Update previous error
                self.pid_previous_error[i] = error
            
            return target_velocities
            
        except Exception as e:
            print(f"PID calculation error: {e}")
            return [0.0] * 6
    
    def update_target(self, face_x, face_y):
        """Update target joints based on face position."""
        if not self.tracking_active:
            return
        
        try:
            # Detect sudden movements
            face_movement = math.sqrt((face_x - self.previous_face_x)**2 + (face_y - self.previous_face_y)**2)
            is_sudden_movement = face_movement > self.sudden_movement_threshold
            
            # Apply movement smoothing for sudden movements
            if is_sudden_movement:
                # Smooth the face position to reduce choppiness
                smoothed_face_x = self.previous_face_x + (face_x - self.previous_face_x) * (1 - self.movement_smoothing_factor)
                smoothed_face_y = self.previous_face_y + (face_y - self.previous_face_y) * (1 - self.movement_smoothing_factor)
            else:
                smoothed_face_x = face_x
                smoothed_face_y = face_y
            
            # Calculate face offset from center using smoothed position
            offset_x = smoothed_face_x - self.center_x
            offset_y = smoothed_face_y - self.center_y
            
            # Check if face is within deadzone - if so, don't move robot
            distance_from_center = math.sqrt(offset_x**2 + offset_y**2)
            if distance_from_center <= self.deadzone_radius:
                # Face is centered - don't move robot, just update previous position
                self.previous_face_x = face_x
                self.previous_face_y = face_y
                return
            
            # Convert to movement increments - CONSTRAINED TO X-Y ONLY
            base_move = -offset_x * FACE_SENSITIVITY / 100  # Left/right movement (X-axis)
            shoulder_move = offset_y * FACE_SENSITIVITY / 100   # Up/down movement (Y-axis) - FIXED SIGN
            # NO elbow movement during face tracking to prevent Z-axis drift
            
            # Update target joints - FACE TRACKING ONLY AFFECTS BASE AND SHOULDER
            self.target_joints[0] += base_move      # Base (left/right)
            self.target_joints[1] += shoulder_move  # Shoulder (up/down)
            # Elbow stays fixed during face tracking - only arrow keys can change Z-depth
            
            # Calculate wrist relationships
            shoulder_deg = math.degrees(self.target_joints[1])
            elbow_deg = math.degrees(self.target_joints[2])
            base_deg = math.degrees(self.target_joints[0])
            
            # Wrist formulas (keep end-effector orientation consistent)
            wrist1_deg = 90 + elbow_deg - abs(shoulder_deg)
            wrist3_deg = -90 + base_deg
            
            self.target_joints[3] = math.radians(-wrist1_deg)  # Wrist1
            self.target_joints[5] = math.radians(wrist3_deg)   # Wrist3
            
            # Update dynamic control parameters (focus on distance-based lookahead)
            self.current_face_x = smoothed_face_x
            self.current_face_y = smoothed_face_y
            self.current_t, self.current_a, self.current_v, self.current_lookahead, self.current_gain = self.calculate_dynamic_t(smoothed_face_x, smoothed_face_y, is_sudden_movement)
            
            # Update previous position for next iteration
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
        except:
            return None
    
    def handle_arrow_keys(self):
        """Handle arrow key presses for Z-direction movement."""
        while self.running:
            try:
                if keyboard.is_pressed('up'):
                    # UP arrow: Move backward in Z (away from camera)
                    self.adjust_z_position(-self.z_step)
                    time.sleep(0.1)  # Prevent too rapid movement
                elif keyboard.is_pressed('down'):
                    # DOWN arrow: Move forward in Z (toward camera)
                    self.adjust_z_position(self.z_step)
                    time.sleep(0.1)
                else:
                    time.sleep(0.05)  # Check keys regularly
            except Exception as e:
                print(f"Arrow key error: {e}")
                time.sleep(0.1)
    
    def adjust_z_position(self, z_change):
        """Adjust robot position in Z direction (forward/backward)."""
        try:
            # Z movement primarily affects shoulder and elbow joints
            # Positive z_change = move forward (toward camera)
            # Negative z_change = move backward (away from camera)
            
            shoulder_adjustment = z_change * 0.5  # Shoulder moves more
            elbow_adjustment = -z_change * 0.8    # Elbow compensates
            
            # Update target joints for Z movement
            self.target_joints[1] += shoulder_adjustment  # Shoulder
            self.target_joints[2] += elbow_adjustment     # Elbow
            
            # Recalculate wrist relationships
            shoulder_deg = math.degrees(self.target_joints[1])
            elbow_deg = math.degrees(self.target_joints[2])
            base_deg = math.degrees(self.target_joints[0])
            
            # Apply wrist formulas
            wrist1_deg = 90 + elbow_deg - abs(shoulder_deg)
            wrist3_deg = -90 + base_deg
            
            self.target_joints[3] = math.radians(-wrist1_deg)  # Wrist1
            self.target_joints[5] = math.radians(wrist3_deg)   # Wrist3
            
        except Exception as e:
            print(f"Z adjustment error: {e}")
    
    def calculate_dynamic_t(self, face_x, face_y, is_sudden_movement=False):
        """Calculate dynamic parameters based on distance from center."""
        try:
            # Calculate distance from center
            offset_x = face_x - self.center_x
            offset_y = face_y - self.center_y
            distance = math.sqrt(offset_x**2 + offset_y**2)
            
            # Normalize distance (0-1, where 1 is edge of frame)
            max_distance = math.sqrt(self.center_x**2 + self.center_y**2)
            normalized_distance = min(distance / max_distance, 1.0)
            
            # Dynamic scaling based on distance from center
            # Close to center = low values (very responsive, PID-like)
            # Far from center = high values (smooth approach)
            
            # Less aggressive decay to reduce overshooting (square instead of cube)
            decay_factor = normalized_distance ** 2  # Reduced from ** 3 to ** 2
            
            # Adjust parameters for sudden movements
            if is_sudden_movement:
                # For sudden movements: moderate parameters
                min_t = 0.8
                max_t = 1.2
                min_a = 0.2
                max_a = 0.5
                min_v = 0.3
                max_v = 0.6
                # Higher lookahead for much smoother motion when far from center
                min_lookahead = 1.0
                max_lookahead = 3.0  # Very high lookahead when suddenly far out
                min_gain = 300
                max_gain = 500
            else:
                # Normal parameters for regular tracking
                min_t = 0.5
                max_t = 0.9
                min_a = 0.2
                max_a = 0.8
                min_v = 0.3
                max_v = 0.8
                # Higher lookahead when far from center for smooth motion
                min_lookahead = 0.2
                max_lookahead = 1.5  # Increased from 0.3 to 1.5
                min_gain = 200
                max_gain = 400
            
            dynamic_t = min_t + (max_t - min_t) * decay_factor
            dynamic_a = min_a + (max_a - min_a) * decay_factor
            dynamic_v = min_v + (max_v - min_v) * decay_factor
            dynamic_lookahead = min_lookahead + (max_lookahead - min_lookahead) * decay_factor
            dynamic_gain = int(min_gain + (max_gain - min_gain) * decay_factor)
            
            return dynamic_t, dynamic_a, dynamic_v, dynamic_lookahead, dynamic_gain
            
        except:
            return 0.8, 0.4, 0.5, 0.8, 300  # Default fallback
    
    def run(self):
        """Main execution."""
        print("\n=== SIMPLE FACE TRACKER ===")
        print("SPACE: Toggle tracking, UP/DOWN: Z movement, ESC: Exit")
        
        # Initialize systems
        if not self.connect_robot():
            return
        
        # Move to starting position
        if not self.move_to_starting_position():
            self.robot.close()
            return
        
        if not self.init_camera():
            self.robot.close()
            return
        
        if not self.start_servoj_loop():
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
                
                # Draw center crosshair
                cv2.circle(frame, (self.center_x, self.center_y), 3, (0, 0, 255), -1)
                cv2.line(frame, (self.center_x-10, self.center_y), (self.center_x+10, self.center_y), (0, 0, 255), 1)
                cv2.line(frame, (self.center_x, self.center_y-10), (self.center_x, self.center_y+10), (0, 0, 255), 1)
                
                # Status text
                status = "TRACKING" if self.tracking_active else "PAUSED"
                color = (0, 255, 0) if self.tracking_active else (0, 0, 255)
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, "SPACE: Toggle, UP/DOWN: Z move, ESC: Exit", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Simple Face Tracker', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Exiting...")
                    break
                elif key == ord(' '):  # SPACE
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
        self.servoj_running = False
        
        # Stop threads
        if self.servoj_thread and self.servoj_thread.is_alive():
            self.servoj_thread.join(timeout=1)
        
        if self.arrow_key_thread and self.arrow_key_thread.is_alive():
            self.arrow_key_thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.robot:
            self.robot.close()
        
        print("✓ Cleanup complete")
    
    def move_to_starting_position(self):
        """Move robot to optimal starting position."""
        try:
            print("Moving to starting position...")
            
            # Define starting position (safe position for face tracking)
            start_joints = [
                0.0,                    # Base = 0°
                math.radians(-60),      # Shoulder = -60°
                math.radians(80),       # Elbow = 80°
                math.radians(-110),     # Wrist1 (calculated)
                math.radians(270),      # Wrist2 = 270°
                math.radians(-90)       # Wrist3 (calculated)
            ]
            
            # Send movement command (may show "Robot stopped" warning but still moves)
            try:
                self.robot.movej(start_joints, acc=0.5, vel=0.3)
            except Exception as move_error:
                print(f"Movement warning: {move_error}")
                print("Checking if robot actually moved...")
            
            # Wait for movement to complete
            print("Waiting for movement to complete...")
            time.sleep(4)  # Give more time for movement
            
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
                
                # Update target joints to current position for servoj
                self.target_joints = [current_joints[i] for i in range(6)]
                return True
            else:
                print(f"✗ Starting position not reached. Error: {max_error_deg:.1f}°")
                return False
            
        except Exception as e:
            print(f"✗ Starting position failed: {e}")
            return False

# ===============================================
# MAIN EXECUTION
# ===============================================

if __name__ == "__main__":
    tracker = SimpleFaceTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        tracker.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        tracker.cleanup() 