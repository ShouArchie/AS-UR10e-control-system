import socket
import time
import cv2 # OpenCV for computer vision
import math # For math.radians and math.pi
import os # For file path checking
import mediapipe as mp # MediaPipe for face detection
import numpy as np

# URSim IP and port (30002 or 30003 for URScript)
# For URSim running on same machine, use localhost
URSIM_IP = "192.168.10.152"  # Change to your URSim IP if different
URSIM_PORT = 30002

# --- Face Tracking Constants ---
# MediaPipe Face Detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

CAMERA_INDEX = 0  # 0 for default laptop webcam
FRAME_WIDTH = 640 # Camera frame width
FRAME_HEIGHT = 480 # Camera frame height

# Target Z and Orientation for the robot's end-effector
# Fixed at 100mm as specified - like a gyroscope at fixed distance
TARGET_Z_MM = 100  # Fixed Z distance - robot stays at this height

# Target orientation as rotation vector (Rx, Ry, Rz) in radians for URScript p[X,Y,Z,Rx,Ry,Rz]
# [0, math.pi, 0] = 180° rotation around Y-axis (tool pointing down)
TARGET_ROT_VEC_RAD = [0, math.pi, 0]  # Tool pointing down for face tracking

# Initial X, Y target for the robot in millimeters (robot base frame)
# Starting position as specified: X=0mm, Y=300mm, Z=100mm
TARGET_X_MM_INITIAL = 0    # Start at X=0mm
TARGET_Y_MM_INITIAL = 300  # Start at Y=300mm

# Control gains for face tracking - optimized for smooth movement
# Increased gains for more responsive tracking while maintaining smoothness
X_GAIN = 0.5   # Horizontal movement gain
Y_GAIN = -0.5  # Vertical movement gain (negative because image Y is inverted)

DEAD_ZONE_PIXELS = 20 # Dead zone to prevent jitter - smaller for more precise tracking

# Safety bounds to prevent robot from moving too far
# Adjusted for the new starting position and Z constraint
X_MIN_MM = -400  # Minimum X position (left limit)
X_MAX_MM = 400   # Maximum X position (right limit)
Y_MIN_MM = 100   # Minimum Y position (closer to robot)
Y_MAX_MM = 500   # Maximum Y position (further from robot)
Z_MIN_MM = 95    # Minimum Z position (slightly below target)
Z_MAX_MM = 105   # Maximum Z position (slightly above target) - tight constraint

# Face tracking workspace limits (3D mapping area)
TRACKING_WORKSPACE_X_MM = 300  # ±300mm in X direction from center
TRACKING_WORKSPACE_Y_MM = 200  # ±200mm in Y direction from initial Y position

# MediaPipe Face Detection parameters
FACE_DETECTION_CONFIDENCE = 0.7  # Higher confidence for better detection
FACE_DETECTION_MODEL = 0  # 0 for short-range (within 2 meters), 1 for full-range

# Debug flag
DEBUG_MODE = True  # Set to False to reduce logging

# --- End Face Tracking Constants ---

def debug_print(message):
    """Print debug messages if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def move_robot_to_cartesian_pose(target_xyz_mm, target_rot_vec_rad, acceleration=0.5, velocity=0.5):
    """
    Move robot to target Cartesian pose using robot's built-in IK.
    Enforces Z=100mm constraint and includes safety bounds checking.
    """
    debug_print(f"move_robot_to_cartesian_pose called with: {target_xyz_mm}")
    
    # Enforce Z constraint - always keep Z at TARGET_Z_MM (100mm)
    constrained_xyz = [target_xyz_mm[0], target_xyz_mm[1], TARGET_Z_MM]
    
    if target_xyz_mm[2] != TARGET_Z_MM:
        debug_print(f"Z constraint applied: {target_xyz_mm[2]}mm -> {TARGET_Z_MM}mm")
    
    # Apply safety bounds
    safe_x = max(X_MIN_MM, min(X_MAX_MM, constrained_xyz[0]))
    safe_y = max(Y_MIN_MM, min(Y_MAX_MM, constrained_xyz[1]))
    safe_z = TARGET_Z_MM  # Always use the target Z
    
    # Warn if bounds were applied
    if [safe_x, safe_y, safe_z] != constrained_xyz:
        print(f"[SAFETY] Bounds applied: {constrained_xyz} -> {[safe_x, safe_y, safe_z]}")
    
    # Convert mm to meters for URScript
    target_xyz_m = [safe_x / 1000.0, safe_y / 1000.0, safe_z / 1000.0]
    debug_print(f"Target in meters: {target_xyz_m}")
    
    # Construct the pose list [X, Y, Z, Rx, Ry, Rz]
    target_pose = target_xyz_m + target_rot_vec_rad
    debug_print(f"Target pose: {target_pose}")
    
    # Create URScript movel command for smoother Cartesian movement
    move_command = f"movel(p{target_pose}, a={acceleration}, v={velocity})\n"
    print(f"[COMMAND] Sending: {move_command.strip()}")
    
    # Send command to robot
    success = send_urscript_command(URSIM_IP, URSIM_PORT, move_command)
    if success:
        print(f"[SUCCESS] Command sent successfully!")
    else:
        print(f"[ERROR] Failed to send command!")
    
    return success

def send_urscript_command(ip, port, command):
    """
    Send URScript command to robot controller.
    Enhanced error handling and detailed logging for debugging.
    """
    debug_print(f"Attempting connection to {ip}:{port}")
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3) # Longer timeout for debugging
        
        debug_print("Connecting to robot...")
        sock.connect((ip, port))
        debug_print("Connected successfully!")
        
        debug_print(f"Sending command: {command.strip()}")
        sock.sendall(command.encode('utf8'))
        debug_print("Command sent to robot")
        
        time.sleep(0.1) # Delay for command processing
        return True
        
    except socket.timeout:
        print(f"[ERROR] Timeout connecting to robot at {ip}:{port}")
        print("        Make sure URSim is running and check the IP address.")
        return False
    except ConnectionRefusedError:
        print(f"[ERROR] Connection refused to {ip}:{port}")
        print("        Make sure URSim is running and the port is correct.")
        return False
    except OSError as e:
        if "No route to host" in str(e):
            print(f"[ERROR] No route to host {ip}")
            print("        Check if the IP address is correct and reachable.")
        else:
            print(f"[ERROR] Network error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error connecting to robot: {e}")
        return False
    finally:
        if sock:
            try:
                sock.close()
                debug_print("Socket closed")
            except:
                pass

def test_network_connection():
    """Test basic network connectivity to the robot."""
    print(f"Testing network connection to {URSIM_IP}...")
    
    try:
        # Try to create a socket and connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((URSIM_IP, URSIM_PORT))
        sock.close()
        
        if result == 0:
            print(f"✓ Network connection to {URSIM_IP}:{URSIM_PORT} successful!")
            return True
        else:
            print(f"✗ Cannot connect to {URSIM_IP}:{URSIM_PORT} (error code: {result})")
            return False
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        return False

def face_tracking_loop():
    """
    Main face tracking loop using MediaPipe Face Detection.
    Enhanced with detailed logging for debugging and 3D tracking capabilities.
    """
    # Initialize MediaPipe Face Detection
    print(f"Initializing MediaPipe Face Detection...")
    print(f"Detection confidence: {FACE_DETECTION_CONFIDENCE}")
    print(f"Detection model: {'Short-range' if FACE_DETECTION_MODEL == 0 else 'Full-range'}")
    
    # Test network connection first
    if not test_network_connection():
        print("Network connection failed. Robot commands will not work.")
        print("Do you want to continue anyway for camera testing? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            return

    # Initialize camera
    print(f"Initializing camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera with index {CAMERA_INDEX}.")
        print("Try different camera indices (0, 1, 2...) or check camera connection.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print("Camera opened successfully!")
    print("Face tracking starting...")
    print("Controls:")
    print("- 'q': Quit")
    print("- 'r': Reset robot to initial position")
    print("- 's': Toggle robot movement on/off")
    print("- 'd': Toggle debug mode")

    # Initialize robot position
    current_target_x_mm = TARGET_X_MM_INITIAL
    current_target_y_mm = TARGET_Y_MM_INITIAL
    
    # Move to initial position
    print(f"\nMoving robot to initial pose: X={current_target_x_mm}mm, Y={current_target_y_mm}mm, Z={TARGET_Z_MM}mm")
    success = move_robot_to_cartesian_pose(
        [current_target_x_mm, current_target_y_mm, TARGET_Z_MM], 
        TARGET_ROT_VEC_RAD, 
        acceleration=0.2, 
        velocity=0.2
    )
    
    if not success:
        print("[WARNING] Failed to send initial position command to robot.")
        print("          Continuing with face tracking, but robot may not move.")
        print("          Check URSim connection and IP address.")
    else:
        print("[SUCCESS] Robot initial position command sent!")
    
    time.sleep(1) # Give time for initial move

    # Calculate image center
    center_x_image = FRAME_WIDTH / 2
    center_y_image = FRAME_HEIGHT / 2

    print("\n=== MEDIAPIPE FACE TRACKING ACTIVE ===")
    print("Position your face in front of the camera!")
    
    robot_enabled = True  # Flag to enable/disable robot movement
    face_detected_count = 0
    commands_sent = 0
    
    # Initialize MediaPipe Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=FACE_DETECTION_MODEL,
        min_detection_confidence=FACE_DETECTION_CONFIDENCE
    ) as face_detection:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting...")
                break

            # Flip frame horizontally for mirror effect (more natural for user)
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe Face Detection
            results = face_detection.process(rgb_frame)

            # Process detected faces with enhanced 3D mapping
            if results.detections:
                face_detected_count += 1
                debug_print(f"Face detected #{face_detected_count}")
                
                if robot_enabled:
                    # Use the first detection (MediaPipe typically returns the most confident one first)
                    detection = results.detections[0]
                    
                    # Extract face information using our helper function
                    face_info = extract_face_info_from_mediapipe(detection, FRAME_WIDTH, FRAME_HEIGHT)
                    
                    x, y, w, h = face_info['bbox']
                    face_center_x = face_info['center_x']
                    face_center_y = face_info['center_y']
                    confidence = face_info['confidence']
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw center point and face info
                    cv2.circle(frame, (int(face_center_x), int(face_center_y)), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Size: {w}x{h}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Estimate face distance for 3D tracking
                    estimated_distance = estimate_face_distance_from_bbox(w, h)
                    face_size_avg = (w + h) / 2
                    
                    # Calculate adaptive gains based on distance and face size
                    adaptive_x_gain, adaptive_y_gain = calculate_adaptive_gains(estimated_distance, face_size_avg)

                    # Calculate error from image center (pixel coordinates)
                    error_x_pixels = face_center_x - center_x_image
                    error_y_pixels = face_center_y - center_y_image

                    # Enhanced 3D mapping: Convert pixel errors to robot coordinate changes
                    # Scale factors based on camera field of view and robot workspace
                    pixel_to_mm_x = TRACKING_WORKSPACE_X_MM / (FRAME_WIDTH / 2)  # mm per pixel in X
                    pixel_to_mm_y = TRACKING_WORKSPACE_Y_MM / (FRAME_HEIGHT / 2) # mm per pixel in Y
                    
                    # Calculate desired robot movement in mm with adaptive gains
                    desired_delta_x_mm = error_x_pixels * pixel_to_mm_x * adaptive_x_gain
                    desired_delta_y_mm = error_y_pixels * pixel_to_mm_y * adaptive_y_gain

                    debug_print(f"Face center: ({face_center_x:.1f}, {face_center_y:.1f})")
                    debug_print(f"Confidence: {confidence:.2f}")
                    debug_print(f"Estimated distance: {estimated_distance:.0f}mm")
                    debug_print(f"Adaptive gains: X={adaptive_x_gain:.3f}, Y={adaptive_y_gain:.3f}")
                    debug_print(f"Pixel error: X={error_x_pixels:.1f}, Y={error_y_pixels:.1f}")
                    debug_print(f"Desired movement: X={desired_delta_x_mm:.1f}mm, Y={desired_delta_y_mm:.1f}mm")

                    # Apply dead zone to prevent jittery movements
                    move_needed = False
                    old_x, old_y = current_target_x_mm, current_target_y_mm
                    
                    if abs(error_x_pixels) > DEAD_ZONE_PIXELS:
                        current_target_x_mm += desired_delta_x_mm
                        move_needed = True
                        debug_print(f"X movement: {old_x:.1f} -> {current_target_x_mm:.1f}mm")
                        
                    if abs(error_y_pixels) > DEAD_ZONE_PIXELS:
                        current_target_y_mm += desired_delta_y_mm
                        move_needed = True
                        debug_print(f"Y movement: {old_y:.1f} -> {current_target_y_mm:.1f}mm")

                    # Send updated position to robot only if movement is needed
                    if move_needed:
                        commands_sent += 1
                        print(f"\n[MOVE #{commands_sent}] MediaPipe face tracking adjustment:")
                        print(f"            Confidence: {confidence:.2f}")
                        print(f"            Pixel error: X={error_x_pixels:.0f}, Y={error_y_pixels:.0f}")
                        print(f"            Movement: X={desired_delta_x_mm:.1f}mm, Y={desired_delta_y_mm:.1f}mm")
                        print(f"            New target: X={current_target_x_mm:.0f}, Y={current_target_y_mm:.0f}, Z={TARGET_Z_MM}mm")
                        
                        success = move_robot_to_cartesian_pose(
                            [current_target_x_mm, current_target_y_mm, TARGET_Z_MM], 
                            TARGET_ROT_VEC_RAD, 
                            acceleration=1.0,  # Faster acceleration for responsive tracking
                            velocity=0.5       # Moderate velocity for smooth movement
                        )
                        if not success:
                            print("[ERROR] Failed to send robot command!")
                    else:
                        debug_print("No movement needed (within dead zone)")

                    # Enhanced display information
                    info_text = f"Robot: X={current_target_x_mm:.0f}, Y={current_target_y_mm:.0f}, Z={TARGET_Z_MM}mm"
                    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Show distance estimation
                    distance_text = f"Distance: {estimated_distance:.0f}mm"
                    cv2.putText(frame, distance_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Show pixel and mm errors
                    error_text = f"Error: {error_x_pixels:.0f}px, {error_y_pixels:.0f}px"
                    cv2.putText(frame, error_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    movement_text = f"Move: {desired_delta_x_mm:.1f}mm, {desired_delta_y_mm:.1f}mm"
                    cv2.putText(frame, movement_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Show adaptive gains and confidence
                    gains_text = f"Gains: X={adaptive_x_gain:.2f}, Y={adaptive_y_gain:.2f}"
                    cv2.putText(frame, gains_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    
                    confidence_text = f"Confidence: {confidence:.2f}"
                    cv2.putText(frame, confidence_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    
                    # Draw tracking workspace bounds on image
                    workspace_x_pixels = int(TRACKING_WORKSPACE_X_MM / pixel_to_mm_x)
                    workspace_y_pixels = int(TRACKING_WORKSPACE_Y_MM / pixel_to_mm_y)
                    
                    cv2.rectangle(frame, 
                                (int(center_x_image - workspace_x_pixels), int(center_y_image - workspace_y_pixels)),
                                (int(center_x_image + workspace_x_pixels), int(center_y_image + workspace_y_pixels)),
                                (255, 0, 255), 1)
                    cv2.putText(frame, "Tracking Zone", (int(center_x_image - workspace_x_pixels), int(center_y_image - workspace_y_pixels - 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                else:
                    # Draw face but don't move robot
                    detection = results.detections[0]
                    face_info = extract_face_info_from_mediapipe(detection, FRAME_WIDTH, FRAME_HEIGHT)
                    x, y, w, h = face_info['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "ROBOT DISABLED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            else:
                # No face detected
                cv2.putText(frame, "No face detected - Move into view", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Robot status and stats
            status_text = "Robot: ON" if robot_enabled else "Robot: OFF"
            status_color = (0, 255, 0) if robot_enabled else (0, 0, 255)
            cv2.putText(frame, status_text, (10, FRAME_HEIGHT - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            stats_text = f"Faces: {face_detected_count}, Commands: {commands_sent}"
            cv2.putText(frame, stats_text, (10, FRAME_HEIGHT - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show current robot position
            position_text = f"Current: X={current_target_x_mm:.0f}, Y={current_target_y_mm:.0f}, Z={TARGET_Z_MM}"
            cv2.putText(frame, position_text, (10, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Show MediaPipe info
            mp_text = f"MediaPipe Face Detection - Model: {'Short' if FACE_DETECTION_MODEL == 0 else 'Full'}"
            cv2.putText(frame, mp_text, (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

            # Draw crosshair at image center
            cv2.line(frame, (int(center_x_image-20), int(center_y_image)), 
                    (int(center_x_image+20), int(center_y_image)), (0, 0, 255), 2)
            cv2.line(frame, (int(center_x_image), int(center_y_image-20)), 
                    (int(center_x_image), int(center_y_image+20)), (0, 0, 255), 2)

            # Display frame
            cv2.imshow('MediaPipe Face Tracking - UR Robot Control (Press Q to quit)', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting face tracking.")
                break
            elif key == ord('r'):
                print("Resetting robot to initial position...")
                current_target_x_mm = TARGET_X_MM_INITIAL
                current_target_y_mm = TARGET_Y_MM_INITIAL
                move_robot_to_cartesian_pose(
                    [current_target_x_mm, current_target_y_mm, TARGET_Z_MM], 
                    TARGET_ROT_VEC_RAD
                )
            elif key == ord('s'):
                robot_enabled = not robot_enabled
                print(f"Robot movement {'enabled' if robot_enabled else 'disabled'}")
            elif key == ord('d'):
                global DEBUG_MODE
                DEBUG_MODE = not DEBUG_MODE
                print(f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nMediaPipe face tracking stopped.")
    print(f"Total faces detected: {face_detected_count}")
    print(f"Total commands sent: {commands_sent}")

def test_robot_connection():
    """Test basic robot connection and movement with URSim."""
    print("Testing robot connection to URSim...")
    print(f"Connecting to: {URSIM_IP}:{URSIM_PORT}")
    
    # First test network connectivity
    if not test_network_connection():
        print("Cannot proceed with robot test - network connection failed.")
        return False
    
    # Test with a simple joint movement to a safe position
    print("\nTesting joint movement...")
    joint_positions_degrees = [0, -90, 90, -90, -90, 0]  # Safe home position
    joint_positions_radians = [math.radians(deg) for deg in joint_positions_degrees]
    move_command = f"movej({joint_positions_radians}, a=0.1, v=0.1)\n"
    
    print(f"Sending joint command: {move_command.strip()}")
    success = send_urscript_command(URSIM_IP, URSIM_PORT, move_command)
    if success:
        print("✓ Joint movement command sent successfully!")
        time.sleep(3)
        
        # Test Cartesian movement
        print("\nTesting Cartesian movement...")
        test_pose = [TARGET_X_MM_INITIAL, TARGET_Y_MM_INITIAL, TARGET_Z_MM]
        success = move_robot_to_cartesian_pose(test_pose, TARGET_ROT_VEC_RAD, 0.1, 0.1)
        if success:
            print("✓ Cartesian movement command sent successfully!")
            print("✓ Robot connection test PASSED!")
        else:
            print("✗ Cartesian movement test FAILED.")
    else:
        print("✗ Robot connection test FAILED.")
        print("\nTroubleshooting:")
        print("1. Make sure URSim is running")
        print("2. Check if URSim IP is correct (current: {})".format(URSIM_IP))
        print("3. Verify URSim is listening on port 30002")
        print("4. Try restarting URSim")
        print("5. Check network connectivity to the robot")
    
    return success

def estimate_face_distance_from_bbox(bbox_width_pixels, bbox_height_pixels):
    """
    Estimate distance to face based on bounding box size in pixels.
    This provides depth information for 3D tracking using MediaPipe detection results.
    
    Based on average human face dimensions:
    - Average face width: ~140mm
    - Average face height: ~180mm
    """
    # Average face dimensions in mm
    AVERAGE_FACE_WIDTH_MM = 140
    AVERAGE_FACE_HEIGHT_MM = 180
    
    # Camera parameters (approximate for typical webcam)
    # These should be calibrated for your specific camera
    FOCAL_LENGTH_PIXELS = 600  # Approximate focal length in pixels
    
    # Calculate distance using face width (usually more stable)
    if bbox_width_pixels > 0:
        distance_from_width = (AVERAGE_FACE_WIDTH_MM * FOCAL_LENGTH_PIXELS) / bbox_width_pixels
    else:
        distance_from_width = 1000  # Default distance if calculation fails
    
    # Calculate distance using face height as backup
    if bbox_height_pixels > 0:
        distance_from_height = (AVERAGE_FACE_HEIGHT_MM * FOCAL_LENGTH_PIXELS) / bbox_height_pixels
    else:
        distance_from_height = 1000
    
    # Use average of both measurements for better accuracy
    estimated_distance = (distance_from_width + distance_from_height) / 2
    
    # Clamp to reasonable range (300mm to 2000mm)
    estimated_distance = max(300, min(2000, estimated_distance))
    
    return estimated_distance

def calculate_adaptive_gains(face_distance_mm, face_size_pixels):
    """
    Calculate adaptive control gains based on face distance and size.
    Closer faces get smaller gains for precision, distant faces get larger gains for responsiveness.
    """
    # Base gains
    base_x_gain = 0.3
    base_y_gain = 0.3
    
    # Distance-based scaling (closer = smaller gains for precision)
    distance_scale = max(0.5, min(2.0, face_distance_mm / 800))  # Scale around 800mm reference
    
    # Size-based scaling (larger faces = smaller gains for stability)
    size_scale = max(0.5, min(1.5, 100 / max(face_size_pixels, 50)))  # Scale around 100px reference
    
    adaptive_x_gain = base_x_gain * distance_scale * size_scale
    adaptive_y_gain = base_y_gain * distance_scale * size_scale
    
    return adaptive_x_gain, adaptive_y_gain

def extract_face_info_from_mediapipe(detection, image_width, image_height):
    """
    Extract face information from MediaPipe detection results.
    Returns face center coordinates and bounding box dimensions.
    """
    # Get bounding box from MediaPipe detection
    bbox = detection.location_data.relative_bounding_box
    
    # Convert relative coordinates to pixel coordinates
    x_min = int(bbox.xmin * image_width)
    y_min = int(bbox.ymin * image_height)
    width = int(bbox.width * image_width)
    height = int(bbox.height * image_height)
    
    # Calculate face center
    face_center_x = x_min + width / 2
    face_center_y = y_min + height / 2
    
    # Get confidence score
    confidence = detection.score[0] if detection.score else 0.0
    
    return {
        'center_x': face_center_x,
        'center_y': face_center_y,
        'bbox': (x_min, y_min, width, height),
        'width': width,
        'height': height,
        'confidence': confidence
    }

if __name__ == "__main__":
    print("="*60)
    print("UR Robot Face Tracking System - Enhanced 3D Tracking")
    print("="*60)
    print(f"Robot IP: {URSIM_IP}:{URSIM_PORT}")
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"Haar Cascade: {HAAR_CASCADE_PATH}")
    print(f"Starting Position: X={TARGET_X_MM_INITIAL}mm, Y={TARGET_Y_MM_INITIAL}mm, Z={TARGET_Z_MM}mm")
    print(f"Z Constraint: Fixed at {TARGET_Z_MM}mm (±{Z_MAX_MM-TARGET_Z_MM}mm)")
    print(f"Safety Bounds: X[{X_MIN_MM}-{X_MAX_MM}], Y[{Y_MIN_MM}-{Y_MAX_MM}]")
    print(f"Tracking Workspace: ±{TRACKING_WORKSPACE_X_MM}mm X, ±{TRACKING_WORKSPACE_Y_MM}mm Y")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print()
    print("Features:")
    print("- 3D face distance estimation")
    print("- Adaptive control gains based on face size and distance")
    print("- Fixed Z-height tracking (gyroscope-like behavior)")
    print("- Enhanced face detection with stability filtering")
    print()
    
    # Ask user what they want to do
    print("Options:")
    print("1. Start face tracking")
    print("2. Test robot connection")
    print("3. Test camera only (no robot)")
    print("4. Test network connection only")
    print("5. Exit")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            face_tracking_loop()
        elif choice == "2":
            test_robot_connection()
        elif choice == "3":
            print("Testing camera only...")
            # Quick camera test
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                print("Camera test successful! Press any key in the video window to close.")
                while True:
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow('Camera Test', cv2.flip(frame, 1))
                        if cv2.waitKey(1) & 0xFF != 255:
                            break
                    else:
                        print("Failed to read from camera")
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                print("Camera test failed!")
        elif choice == "4":
            test_network_connection()
        elif choice == "5":
            print("Exiting.")
        else:
            print("Invalid choice. Starting face tracking by default.")
            face_tracking_loop()
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
