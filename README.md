# UR Robot Face Tracking System - Enhanced 3D Tracking with MediaPipe

An advanced face tracking system for Universal Robots (UR) that implements real-time 3D face tracking with fixed Z-height constraint, using Google's MediaPipe Face Detection for superior accuracy and performance.

## Features

- **MediaPipe Face Detection**: State-of-the-art face detection from Google (sub-millisecond performance)
- **3D Face Distance Estimation**: Estimates face distance based on face size in pixels
- **Adaptive Control Gains**: Automatically adjusts tracking sensitivity based on face distance and size
- **Fixed Z-Height Tracking**: Maintains robot at constant Z=100mm height (gyroscope-like behavior)
- **Real-time Visual Feedback**: Comprehensive on-screen information display with confidence scores
- **Safety Constraints**: Built-in safety bounds to prevent dangerous robot movements
- **High Accuracy**: MediaPipe provides much better detection than traditional Haar cascades

## System Configuration

### Starting Position
- **X**: 0mm (robot base frame)
- **Y**: 300mm (robot base frame) 
- **Z**: 100mm (fixed height - never changes)

### Tracking Workspace
- **X Range**: ±300mm from center
- **Y Range**: ±200mm from initial Y position
- **Z Constraint**: Fixed at 100mm (±5mm safety margin)

### Safety Bounds
- **X**: -400mm to +400mm
- **Y**: 100mm to 500mm
- **Z**: 95mm to 105mm (tight constraint)

## Requirements

### Hardware
- Universal Robot (UR3, UR5, UR10) or URSim simulator
- USB webcam or laptop camera
- Network connection to robot controller

### Software
- Python 3.7+
- OpenCV 4.x
- MediaPipe
- NumPy

### Python Dependencies
```bash
pip install opencv-python mediapipe numpy
```

## Installation

1. Clone or download the repository
2. Install Python dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
3. Update the robot IP address in `sim.py`:
   ```python
   URSIM_IP = "192.168.10.152"  # Change to your robot's IP
   ```

## Usage

### Quick Start

1. **Test Camera Only** (no robot required):
   ```bash
   python test_face_tracking.py
   ```

2. **Full Face Tracking with Robot**:
   ```bash
   python sim.py
   ```

### Menu Options

When running `sim.py`, you'll see a menu:
1. Start face tracking
2. Test robot connection
3. Test camera only (no robot)
4. Test network connection only
5. Exit

### Controls During Face Tracking

- **'q'**: Quit face tracking
- **'r'**: Reset robot to initial position
- **'s'**: Toggle robot movement on/off
- **'d'**: Toggle debug mode

## How It Works

### MediaPipe Face Detection
- Uses Google's MediaPipe Face Detection model
- Two model options:
  - **Short-range** (default): Optimized for faces within 2 meters
  - **Full-range**: Better for longer distances
- Provides confidence scores for each detection
- Sub-millisecond inference time on modern hardware
- Much more accurate and stable than traditional methods

### Detection Parameters
```python
FACE_DETECTION_CONFIDENCE = 0.7  # Higher confidence for better detection
FACE_DETECTION_MODEL = 0  # 0 for short-range, 1 for full-range
```

### Distance Estimation
The system estimates face distance using the relationship:
```
distance = (average_face_size_mm * focal_length_pixels) / face_size_pixels
```

Where:
- Average face width: 140mm
- Average face height: 180mm
- Focal length: ~600 pixels (typical webcam)

### Adaptive Control
Control gains are automatically adjusted based on:
- **Face distance**: Closer faces get smaller gains for precision
- **Face size**: Larger faces get smaller gains for stability
- **Detection confidence**: Higher confidence allows more responsive tracking

### 3D Mapping
Pixel errors are converted to robot movements using:
```
robot_movement_mm = pixel_error * workspace_scale * adaptive_gain
```

## Configuration

### Key Parameters (in `sim.py`)

```python
# Starting position
TARGET_X_MM_INITIAL = 0      # X starting position
TARGET_Y_MM_INITIAL = 300    # Y starting position  
TARGET_Z_MM = 100           # Fixed Z height

# Tracking workspace
TRACKING_WORKSPACE_X_MM = 300  # ±300mm in X
TRACKING_WORKSPACE_Y_MM = 200  # ±200mm in Y

# MediaPipe parameters
FACE_DETECTION_CONFIDENCE = 0.7  # Detection confidence threshold
FACE_DETECTION_MODEL = 0         # 0=short-range, 1=full-range

# Control parameters
DEAD_ZONE_PIXELS = 20       # Dead zone to prevent jitter
```

### Robot IP Configuration
Update the robot IP address:
```python
URSIM_IP = "192.168.10.152"  # Change to your robot's IP
URSIM_PORT = 30002          # URScript port
```

## Visual Feedback

The system displays real-time information:
- **Robot position**: Current X, Y, Z coordinates
- **Face distance**: Estimated distance in mm
- **Detection confidence**: MediaPipe confidence score
- **Pixel errors**: X, Y pixel offsets from center
- **Movement commands**: Actual robot movement in mm
- **Adaptive gains**: Current control gain values
- **Tracking zone**: Visual workspace boundaries
- **Detection statistics**: Face count and command count
- **MediaPipe landmarks**: Visual face detection overlay

## Performance Comparison

| Method | Detection Speed | Accuracy | Stability | False Positives |
|--------|----------------|----------|-----------|-----------------|
| **MediaPipe** | **Sub-ms** | **Excellent** | **Very High** | **Very Low** |
| Haar Cascades | ~10-50ms | Good | Medium | Medium |
| Traditional CV | ~100ms+ | Fair | Low | High |

## Safety Features

1. **Position Bounds**: Prevents robot from moving outside safe workspace
2. **Z Constraint**: Enforces fixed Z=100mm height
3. **Dead Zone**: Prevents jittery movements from small face movements
4. **Adaptive Gains**: Reduces movement sensitivity for large/close faces
5. **Confidence Filtering**: Only tracks high-confidence detections
6. **Connection Monitoring**: Handles network failures gracefully

## Troubleshooting

### Camera Issues
- Try different camera indices (0, 1, 2...)
- Check camera permissions
- Ensure camera is not used by other applications

### Robot Connection Issues
- Verify robot IP address
- Check network connectivity
- Ensure URSim is running (if using simulator)
- Verify robot is not in protective stop

### MediaPipe Issues
- Ensure good lighting conditions
- Position face clearly in camera view
- Try adjusting `FACE_DETECTION_CONFIDENCE` (lower for easier detection)
- Switch between short-range and full-range models

### Performance Issues
- Close other applications using the camera
- Ensure adequate CPU/GPU resources
- Try reducing camera resolution if needed

## Technical Details

### Coordinate Systems
- **Image coordinates**: (0,0) at top-left, Y increases downward
- **Robot coordinates**: Robot base frame, Y inverted for natural tracking

### Movement Commands
- Uses `movel()` for smooth Cartesian movements
- Acceleration: 1.0 rad/s² (responsive)
- Velocity: 0.5 rad/s (smooth)

### Communication Protocol
- URScript over TCP socket (port 30002)
- Real-time command streaming
- Error handling for connection failures

### MediaPipe Models
- **BlazeFace Short-range**: Optimized for selfie-like images (default)
- **BlazeFace Full-range**: Better for longer distances and varied angles
- Both models are lightweight and optimized for real-time performance

## Based On

This implementation is inspired by and builds upon:
- [Google MediaPipe Face Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)
- [UR_Facetracking by robin-gdwl](https://github.com/robin-gdwl/UR_Facetracking)
- Universal Robots URScript programming interface
- Google's BlazeFace research

## License

This project follows the same licensing as the original UR_Facetracking repository (GPLv3).

MediaPipe is licensed under the Apache License 2.0.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system. 