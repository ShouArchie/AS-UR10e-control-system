#!/usr/bin/env python3
"""
Test script for MediaPipe face tracking functionality without robot connection.
This allows testing the computer vision components independently.
"""

import cv2
import time
import mediapipe as mp
import numpy as np

# MediaPipe Face Detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe parameters
FACE_DETECTION_CONFIDENCE = 0.7
FACE_DETECTION_MODEL = 0  # 0 for short-range, 1 for full-range

def estimate_face_distance_from_bbox(bbox_width_pixels, bbox_height_pixels):
    """Estimate distance to face based on bounding box size in pixels."""
    AVERAGE_FACE_WIDTH_MM = 140
    AVERAGE_FACE_HEIGHT_MM = 180
    FOCAL_LENGTH_PIXELS = 600
    
    if bbox_width_pixels > 0:
        distance_from_width = (AVERAGE_FACE_WIDTH_MM * FOCAL_LENGTH_PIXELS) / bbox_width_pixels
    else:
        distance_from_width = 1000
    
    if bbox_height_pixels > 0:
        distance_from_height = (AVERAGE_FACE_HEIGHT_MM * FOCAL_LENGTH_PIXELS) / bbox_height_pixels
    else:
        distance_from_height = 1000
    
    estimated_distance = (distance_from_width + distance_from_height) / 2
    estimated_distance = max(300, min(2000, estimated_distance))
    
    return estimated_distance

def extract_face_info_from_mediapipe(detection, image_width, image_height):
    """Extract face information from MediaPipe detection results."""
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

def test_face_tracking():
    """Test MediaPipe face tracking without robot connection."""
    print("="*60)
    print("MediaPipe Face Tracking Test - No Robot Required")
    print("="*60)
    print("Press 'q' to quit")
    print("This test will show MediaPipe face detection and distance estimation")
    print(f"Detection confidence: {FACE_DETECTION_CONFIDENCE}")
    print(f"Detection model: {'Short-range' if FACE_DETECTION_MODEL == 0 else 'Full-range'}")
    print()
    
    # Initialize camera
    print(f"Initializing camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera with index {CAMERA_INDEX}")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print("Camera opened successfully!")
    
    # Calculate image center
    center_x = FRAME_WIDTH / 2
    center_y = FRAME_HEIGHT / 2
    
    face_count = 0
    start_time = time.time()
    
    # Initialize MediaPipe Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=FACE_DETECTION_MODEL,
        min_detection_confidence=FACE_DETECTION_CONFIDENCE
    ) as face_detection:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = face_detection.process(rgb_frame)
            
            # Process detected faces
            if results.detections:
                face_count += 1
                
                # Use first detection
                detection = results.detections[0]
                face_info = extract_face_info_from_mediapipe(detection, FRAME_WIDTH, FRAME_HEIGHT)
                
                x, y, w, h = face_info['bbox']
                face_center_x = face_info['center_x']
                face_center_y = face_info['center_y']
                confidence = face_info['confidence']
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, (int(face_center_x), int(face_center_y)), 5, (0, 255, 0), -1)
                
                # Estimate distance
                estimated_distance = estimate_face_distance_from_bbox(w, h)
                
                # Calculate errors from center
                error_x = face_center_x - center_x
                error_y = face_center_y - center_y
                
                # Display information
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Size: {w}x{h}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Distance: {estimated_distance:.0f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Error: X={error_x:.0f}, Y={error_y:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Center: ({face_center_x:.0f}, {face_center_y:.0f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show what robot movement would be
                robot_move_x = error_x * 0.5  # Simulated movement
                robot_move_y = error_y * 0.5
                cv2.putText(frame, f"Robot Move: X={robot_move_x:.1f}, Y={robot_move_y:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                # Draw MediaPipe landmarks if available
                mp_drawing.draw_detection(frame, detection)
                
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw crosshair at center
            cv2.line(frame, (int(center_x-20), int(center_y)), (int(center_x+20), int(center_y)), (0, 0, 255), 2)
            cv2.line(frame, (int(center_x), int(center_y-20)), (int(center_x), int(center_y+20)), (0, 0, 255), 2)
            
            # Show stats
            elapsed_time = time.time() - start_time
            fps = face_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"Faces detected: {face_count}", (10, FRAME_HEIGHT - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Detection rate: {fps:.1f}/s", (10, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show MediaPipe info
            mp_text = f"MediaPipe Face Detection - Model: {'Short' if FACE_DETECTION_MODEL == 0 else 'Full'}"
            cv2.putText(frame, mp_text, (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            
            # Display frame
            cv2.imshow('MediaPipe Face Tracking Test (Press Q to quit)', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nMediaPipe test completed!")
    print(f"Total faces detected: {face_count}")
    print(f"Average detection rate: {fps:.1f} faces/second")
    
    return True

if __name__ == "__main__":
    test_face_tracking() 