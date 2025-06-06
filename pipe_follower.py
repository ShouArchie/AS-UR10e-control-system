import urx
import math
import time
import numpy as np

# ===============================================
# PIPE FOLLOWER CONFIGURATION
# ===============================================
ROBOT_IP = "192.168.0.101"

class PipeFollower:
    def __init__(self):
        self.robot = None
        self.current_pose = None
        
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
        """Get current robot pose with error handling."""
        try:
            pose = self.robot.getl()
            
            # Handle different PoseVector object types (same as existing code)
            if hasattr(pose, 'array') and pose.array is not None:
                try:
                    import numpy as np
                    if isinstance(pose.array, np.ndarray):
                        return pose.array.tolist()
                except:
                    pass
            
            # Try direct array access
            if hasattr(pose, 'array') and hasattr(pose.array, 'tolist'):
                try:
                    return pose.array.tolist()
                except:
                    pass
            
            # Try list conversion
            if hasattr(pose, 'array'):
                try:
                    return list(pose.array)
                except:
                    pass
                    
            # Try individual property access
            try:
                return [float(pose.x), float(pose.y), float(pose.z), 
                       float(pose.rx), float(pose.ry), float(pose.rz)]
            except:
                pass
            
            # Try index access
            try:
                return [float(pose[i]) for i in range(6)]
            except:
                pass
                
            print(f"Unable to parse pose object: {type(pose)}")
            return None
            
        except Exception as e:
            print(f"Get pose error: {e}")
            return None
    
    def calculate_pipe_path(self, pipe_radius_mm, movement_angle_deg=45, num_points_per_segment=10):
        """
        Calculate circular path points around a pipe from current position.
        Robot moves up 45° from start, then down 45° from start position.
        
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
            if current_pose is None:
                print("Cannot get current pose!")
                return []
            
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
            
            print(f"\nTotal waypoints generated: {len(waypoints)}")
            print(f"Path: Start → +{movement_angle_deg}° → -{movement_angle_deg}° (total {2*movement_angle_deg}° travel)")
            
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
                    # Use movel for smooth linear movement between points
                    self.robot.movel(waypoint, acc=0.3, vel=0.1)  # Slow and smooth movement
                    
                    # Small pause between movements
                    time.sleep(0.5)
                    
                except Exception as move_error:
                    print(f"Movement error at waypoint {i+1}: {move_error}")
                    continue
            
            print("✓ Pipe following sequence completed!")
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
        print("Robot will move UP 45° from current position, then DOWN 45° from starting position")
        
        # Connect to robot
        if not self.connect_robot():
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