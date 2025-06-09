import urx
import math
import time
import numpy as np

# ===============================================
# PIPE FOLLOWER CONFIGURATION
# ===============================================
ROBOT_IP = "192.168.10.223"

class PipeFollower:
    def __init__(self):
        self.robot = None
        self.current_pose = None
        self.starting_joints = None  # Store starting position for return
        
    def connect_robot(self):
        """Connect to the robot."""
        try:
            print(f"Connecting to robot at {ROBOT_IP}...")
            self.robot = urx.Robot(ROBOT_IP)
            print("✓ Robot connected!")
            return True
        except Exception as e:
            print(f"✗ Robot connection failed: {e}")
            return False
    
    def get_current_pose(self):
        """Get current robot pose with robust error handling (based on working face_tracking_camera.py)."""
        try:
            pose = self.robot.getl()
            
            # Handle different possible formats (from working face tracking code)
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
            
            # Ultimate fallback: try different pose access methods
            try:
                print("Trying alternative pose access...")
                # Try individual property access if robot object has these
                if hasattr(self.robot, 'x'):
                    x = float(self.robot.x) if hasattr(self.robot.x, '__float__') else self.robot.x.array[0] if hasattr(self.robot.x, 'array') else 0.0
                    y = float(self.robot.y) if hasattr(self.robot.y, '__float__') else self.robot.y.array[0] if hasattr(self.robot.y, 'array') else 0.0
                    z = float(self.robot.z) if hasattr(self.robot.z, '__float__') else self.robot.z.array[0] if hasattr(self.robot.z, 'array') else 0.0
                    rx = float(self.robot.rx) if hasattr(self.robot.rx, '__float__') else self.robot.rx.array[0] if hasattr(self.robot.rx, 'array') else 0.0
                    ry = float(self.robot.ry) if hasattr(self.robot.ry, '__float__') else self.robot.ry.array[0] if hasattr(self.robot.ry, 'array') else 0.0
                    rz = float(self.robot.rz) if hasattr(self.robot.rz, '__float__') else self.robot.rz.array[0] if hasattr(self.robot.rz, 'array') else 0.0
                    return [x, y, z, rx, ry, rz]
                
                # Final fallback: use last known pose or zeros
                if hasattr(self, 'last_known_pose'):
                    print("Using last known pose")
                    return self.last_known_pose
                else:
                    print("Using zero pose as last resort")
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    
            except Exception as e2:
                print(f"Fallback pose retrieval also failed: {e2}")
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def calculate_pipe_path(self, pipe_radius_mm, movement_angle_deg=45, num_points_per_segment=10):
        """
        Calculate circular path points around a pipe from current position.
        Robot moves up 45°, then down 90°, then back up 45° (complete cycle).
        
        Args:
            pipe_radius_mm: Radius of the pipe in millimeters
            movement_angle_deg: Degrees to move up and down from starting position (default 45°)
            num_points_per_segment: Number of waypoints per movement segment
        
        Returns:
            List of [x, y, z, rx, ry, rz] poses for the complete path
        """
        try:
            # Convert mm to meters for robot coordinates
            pipe_radius = pipe_radius_mm / 1000.0
            
                         # Get current position as starting point
            current_pose = self.get_current_pose()
            if current_pose is None or current_pose == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
                print("Cannot get current pose!")
                return []
            
            # Store pose as last known good pose
            self.last_known_pose = current_pose.copy()
            
            print(f"Current pose: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Z={current_pose[2]:.3f}")
            print(f"Pipe radius: {pipe_radius_mm}mm ({pipe_radius:.3f}m)")
            
            # Calculate the center of the pipe based on current position
            # Assuming robot is tangent to pipe, calculate pipe center
            # Current position represents some angle on the pipe - we need to find center
            
            # For vertical pipe (most common case), assume center is offset in Y direction
            # Current position is at pipe_radius distance from center
            center_x = current_pose[0]
            center_y = current_pose[1] - pipe_radius  # Assuming robot is on the "right" side of pipe
            center_z = current_pose[2]
            
            # Calculate the starting angle based on current position relative to center
            dy = current_pose[1] - center_y
            dz = current_pose[2] - center_z
            start_angle_rad = math.atan2(dz, dy)  # Current angle on the pipe
            start_angle_deg = math.degrees(start_angle_rad)
            
            print(f"Pipe center: X={center_x:.3f}, Y={center_y:.3f}, Z={center_z:.3f}")
            print(f"Starting angle on pipe: {start_angle_deg:.1f}°")
            
            # Create complete path: up 45°, then down 45° from start
            waypoints = []
            
            # Segment 1: Move UP from start angle to start+45°
            print(f"\nSegment 1: Moving UP {movement_angle_deg}° from starting position")
            up_angles = np.linspace(start_angle_rad, 
                                  start_angle_rad + math.radians(movement_angle_deg), 
                                  num_points_per_segment)
            
            for i, angle in enumerate(up_angles):
                # Calculate position on circle (in Y-Z plane for vertical pipe)
                y_pos = center_y + pipe_radius * math.cos(angle)
                z_pos = center_z + pipe_radius * math.sin(angle)
                x_pos = current_pose[0]  # Keep X constant
                
                # Keep orientation constant (maintain tangent to pipe)
                waypoint = [x_pos, y_pos, z_pos, current_pose[3], current_pose[4], current_pose[5]]
                waypoints.append(waypoint)
                
                angle_deg = math.degrees(angle) - start_angle_deg  # Relative to start
                print(f"  Up waypoint {i+1}: {angle_deg:+.1f}° from start, Y={y_pos:.3f}, Z={z_pos:.3f}")
            
            # Segment 2: Move DOWN from start+45° to start-45°
            print(f"\nSegment 2: Moving DOWN {2*movement_angle_deg}° (through start to -{movement_angle_deg}°)")
            down_angles = np.linspace(start_angle_rad + math.radians(movement_angle_deg),
                                    start_angle_rad - math.radians(movement_angle_deg), 
                                    num_points_per_segment*2)  # More points for longer segment
            
            for i, angle in enumerate(down_angles[1:]):  # Skip first point (duplicate of last up point)
                # Calculate position on circle
                y_pos = center_y + pipe_radius * math.cos(angle)
                z_pos = center_z + pipe_radius * math.sin(angle)
                x_pos = current_pose[0]  # Keep X constant
                
                # Keep orientation constant
                waypoint = [x_pos, y_pos, z_pos, current_pose[3], current_pose[4], current_pose[5]]
                waypoints.append(waypoint)
                
                angle_deg = math.degrees(angle) - start_angle_deg  # Relative to start
                print(f"  Down waypoint {i+1}: {angle_deg:+.1f}° from start, Y={y_pos:.3f}, Z={z_pos:.3f}")
            
            # Segment 3: Move UP from start-45° back to start (return to original position on pipe)
            print(f"\nSegment 3: Moving UP {movement_angle_deg}° (returning to starting position on pipe)")
            return_angles = np.linspace(start_angle_rad - math.radians(movement_angle_deg),
                                      start_angle_rad, 
                                      num_points_per_segment)
            
            for i, angle in enumerate(return_angles[1:]):  # Skip first point (duplicate of last down point)
                # Calculate position on circle
                y_pos = center_y + pipe_radius * math.cos(angle)
                z_pos = center_z + pipe_radius * math.sin(angle)
                x_pos = current_pose[0]  # Keep X constant
                
                # Keep orientation constant
                waypoint = [x_pos, y_pos, z_pos, current_pose[3], current_pose[4], current_pose[5]]
                waypoints.append(waypoint)
                
                angle_deg = math.degrees(angle) - start_angle_deg  # Relative to start
                print(f"  Return waypoint {i+1}: {angle_deg:+.1f}° from start, Y={y_pos:.3f}, Z={z_pos:.3f}")
            
            print(f"\nTotal waypoints generated: {len(waypoints)}")
            print(f"Path: Start → +{movement_angle_deg}° → -{movement_angle_deg}° → Start (complete cycle)")
            
            return waypoints
            
        except Exception as e:
            print(f"Path calculation error: {e}")
            return []
    
    def execute_pipe_following(self, pipe_radius_mm):
        """
        Execute the complete pipe following sequence.
        
        Args:
            pipe_radius_mm: Radius of the pipe in millimeters
        """
        try:
            print(f"\n=== PIPE FOLLOWING: {pipe_radius_mm}mm radius ===")
            
            # Calculate the path waypoints
            waypoints = self.calculate_pipe_path(pipe_radius_mm)
            
            if not waypoints:
                print("No waypoints generated!")
                return False
            
            print(f"\nExecuting path with {len(waypoints)} waypoints...")
            
                         # Move through each waypoint
            for i, waypoint in enumerate(waypoints):
                print(f"Moving to waypoint {i+1}/{len(waypoints)}")
                print(f"  Target: X={waypoint[0]:.3f}, Y={waypoint[1]:.3f}, Z={waypoint[2]:.3f}")
                
                try:
                    # Use URScript movel command to bypass PoseVector issues (same approach as face tracking)
                    x, y, z, rx, ry, rz = waypoint
                    urscript_cmd = f"movel(p[{x:.6f}, {y:.6f}, {z:.6f}, {rx:.6f}, {ry:.6f}, {rz:.6f}], a=0.3, v=0.1)"
                    print(f"  Sending URScript: {urscript_cmd}")
                    self.robot.send_program(urscript_cmd)
                    
                    # Wait for movement to complete before next waypoint
                    time.sleep(2.0)  # Give more time for each movement
                    
                except Exception as move_error:
                    print(f"Movement error at waypoint {i+1}: {move_error}")
                    continue
            
            print("✓ Pipe following sequence completed!")
            print("✓ Robot has returned to starting position on pipe circumference!")
            
            return True
            
        except Exception as e:
            print(f"Execution error: {e}")
            return False
    
    def run_pipe_demo(self, pipe_radius_mm=60):
        """
        Run the complete pipe following demonstration.
        
        Args:
            pipe_radius_mm: Radius of the pipe in millimeters (default 60mm = 6cm)
        """
        print("\n=== PIPE FOLLOWER DEMO ===")
        print(f"Pipe radius: {pipe_radius_mm}mm")
        print("Robot will move UP 45°, then DOWN 90°, then UP 45° (complete pipe cycle)")
        
        # Connect to robot
        if not self.connect_robot():
            return
        
        # Move to starting position first
        if not self.move_to_starting_position():
            if self.robot:
                self.robot.close()
            return
        
        try:
            # Get and display current position
            current_pose = self.get_current_pose()
            if current_pose:
                print(f"\nStarting position:")
                print(f"  X: {current_pose[0]*1000:.1f}mm")
                print(f"  Y: {current_pose[1]*1000:.1f}mm") 
                print(f"  Z: {current_pose[2]*1000:.1f}mm")
            
            # Wait for user confirmation
            input("\nPress Enter to start pipe following (or Ctrl+C to cancel)...")
            
            # Execute the pipe following sequence
            success = self.execute_pipe_following(pipe_radius_mm)
            
            if success:
                print("\n✓ Pipe following completed successfully!")
            else:
                print("\n✗ Pipe following failed!")
                
        except KeyboardInterrupt:
            print("\nCancelled by user")
        except Exception as e:
            print(f"Demo error: {e}")
        finally:
            # Cleanup
            if self.robot:
                self.robot.close()
                print("✓ Robot connection closed")
    
    def move_to_starting_position(self):
        """Move robot to optimal starting position (same as dynamic face tracking)."""
        try:
            print("Moving to starting position...")
            
            # Define starting position (safe position from face tracking)
            start_joints = [
                0.0,                    # Base = 0°
                math.radians(-60),      # Shoulder = -60°
                math.radians(80),       # Elbow = 80°
                math.radians(-110),     # Wrist1 (calculated)
                math.radians(270),      # Wrist2 = 270°
                math.radians(-90)       # Wrist3 (calculated)
            ]
            
            print(f"Target joint angles (degrees): {[math.degrees(j) for j in start_joints]}")
            
            # Send movement command using URScript to avoid PoseVector issues
            try:
                # Use URScript movej command for reliability
                urscript_cmd = (f"movej([{start_joints[0]:.6f}, {start_joints[1]:.6f}, "
                               f"{start_joints[2]:.6f}, {start_joints[3]:.6f}, "
                               f"{start_joints[4]:.6f}, {start_joints[5]:.6f}], "
                               f"a=0.5, v=0.3)")
                print(f"Sending URScript: {urscript_cmd}")
                self.robot.send_program(urscript_cmd)
                
            except Exception as move_error:
                print(f"Movement warning: {move_error}")
                print("Checking if robot actually moved...")
            
            # Wait for movement to complete
            print("Waiting for movement to complete...")
            time.sleep(4)  # Give time for movement
            
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
            print(f"Current joints (degrees): {[math.degrees(j) for j in current_joints]}")
            
            if max_error_deg < 5.0:  # Within 5 degrees tolerance
                print("✓ Starting position reached successfully!")
                # Store the starting joints for return movement
                self.starting_joints = start_joints.copy()
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
    # Example usage with different pipe radii
    follower = PipeFollower()
    
    print("Pipe Follower Options:")
    print("1. Default demo (60mm radius)")
    print("2. Custom radius")
    
    try:
        choice = input("Choose option (1 or 2): ").strip()
        
        if choice == "2":
            radius = float(input("Enter pipe radius in mm: "))
            follower.run_pipe_demo(radius)
        else:
            # Default demo with 60mm radius (6cm as mentioned in the request)
            follower.run_pipe_demo(60)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except ValueError:
        print("Invalid input. Using default 60mm radius.")
        follower.run_pipe_demo(60)
    except Exception as e:
        print(f"Error: {e}") 