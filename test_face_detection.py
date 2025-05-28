import cv2
import numpy as np
import time

def test_face_detection():
    """Test face detection without robot connection."""
    print("Testing face detection system...")
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Try camera 0 first
    
    if not cap.isOpened():
        print("✗ Could not open camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print("✗ Could not open any camera")
        return False
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✓ Camera initialized successfully!")
    print("Controls:")
    print("- ESC: Exit")
    print("- SPACE: Toggle face detection")
    
    face_detection_active = True
    frame_center_x = 320
    frame_center_y = 240
    dead_zone_pixels = 20
    
    while True:
        try:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            if face_detection_active:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(50, 50)
                )
                
                # Process faces
                if len(faces) > 0:
                    # Find largest face
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h = largest_face
                    
                    # Calculate face center
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    
                    # Calculate offset from frame center
                    offset_x = face_center_x - frame_center_x
                    offset_y = face_center_y - frame_center_y
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw face center
                    cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)
                    
                    # Draw offset information
                    cv2.putText(frame, f"Offset X: {offset_x:+d}px", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Offset Y: {offset_y:+d}px", (10, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show movement commands
                    if abs(offset_x) > dead_zone_pixels:
                        direction_x = "RIGHT" if offset_x > 0 else "LEFT"
                        cv2.putText(frame, f"Move {direction_x}", (10, 170), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    if abs(offset_y) > dead_zone_pixels:
                        direction_y = "DOWN" if offset_y > 0 else "UP"
                        cv2.putText(frame, f"Move {direction_y}", (10, 200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Draw dead zone
                    cv2.rectangle(frame, 
                                 (frame_center_x - dead_zone_pixels, frame_center_y - dead_zone_pixels),
                                 (frame_center_x + dead_zone_pixels, frame_center_y + dead_zone_pixels),
                                 (255, 0, 0), 1)
            
            # Draw frame center crosshair
            cv2.circle(frame, (frame_center_x, frame_center_y), 3, (0, 0, 255), -1)
            cv2.line(frame, (frame_center_x-20, frame_center_y), 
                    (frame_center_x+20, frame_center_y), (0, 0, 255), 1)
            cv2.line(frame, (frame_center_x, frame_center_y-20), 
                    (frame_center_x, frame_center_y+20), (0, 0, 255), 1)
            
            # Draw status
            status = "DETECTING" if face_detection_active else "PAUSED"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0) if face_detection_active else (0, 0, 255), 2)
            
            cv2.putText(frame, "SPACE: Toggle detection, ESC: Exit", 
                       (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Face Detection Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Exiting face detection test...")
                break
            elif key == ord(' '):  # Space key
                face_detection_active = not face_detection_active
                status = "activated" if face_detection_active else "paused"
                print(f"Face detection {status}")
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Face detection test completed")
    return True

if __name__ == "__main__":
    test_face_detection() 