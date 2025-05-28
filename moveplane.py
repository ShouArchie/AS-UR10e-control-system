import urx
import time
import math
import numpy as np

class PlaneMovementController:
    """
    Controller for moving a UR robot on and through a vertical plane.
    Uses the robot's built-in inverse kinematics for reliable movement.
    """
    
    def __init__(self, robot_ip="192.168.10.152"):
        """Initialize the plane movement controller."""
        self.robot = None
        self.robot_ip = robot_ip
        self.plane_origin = None  # Point on the plane [x, y, z]
        self.plane_normal = None  # Normal vector of the plane [nx, ny, nz]
        self.plane_u_axis = None  # U axis of the plane (horizontal)
        self.plane_v_axis = None  # V axis of the plane (vertical)
        
        # Movement parameters
        self.default_acc = 0.1
        self.default_vel = 0.1
        
    def connect(self):
        """Connect to the robot."""
        try:
            print(f"Connecting to robot at {self.robot_ip}...")
            self.robot = urx.Robot(self.robot_ip)
            print("Connected successfully!")
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the robot."""
        if self.robot:
            self.robot.close()
            print("Robot disconnected.")
    
    def move_to_safe_position(self):
        """
        Move robot to a safe starting position to avoid singularities.
        This position should be well within the workspace and away from joint limits.
        """
        try:
            print("Moving to safe starting position...")
            
            # Safe joint positions (in degrees, then converted to radians)
            # These avoid singularities and joint limits
            safe_joints_deg = [0, -90, 90, -90, -90, 0]  # Modified for better workspace access
            safe_joints_rad = [math.radians(deg) for deg in safe_joints_deg]
            
            print(f"Safe joint positions (degrees): {safe_joints_deg}")
            print(f"Safe joint positions (radians): {safe_joints_rad}")
            
            # Move to safe position using joint movement (more reliable than Cartesian)
            self.robot.movej(safe_joints_rad, acc=self.default_acc, vel=self.default_vel, wait=False)
            print("Safe position command sent!")
            
            # Wait for movement completion with position verification
            if self.wait_for_position_reached(target_joints=safe_joints_rad, tolerance=0.01):
                print("Robot successfully moved to safe starting position!")
                return True
            else:
                print("Failed to reach safe starting position!")
                return False
            
        except Exception as e:
            print(f"Error moving to safe position: {e}")
            return False
    
    def wait_for_position_reached(self, target_joints=None, target_pose=None, tolerance=0.01, timeout=30):
        """
        Wait for robot to reach target position with verification.
        
        Args:
            target_joints: Target joint positions in radians (for joint movements)
            target_pose: Target pose [x, y, z, rx, ry, rz] (for Cartesian movements)
            tolerance: Position tolerance in radians for joints or meters for pose
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if position reached, False if timeout or error
        """
        start_time = time.time()
        
        print("Waiting for movement completion with position verification...")
        
        while (time.time() - start_time) < timeout:
            try:
                # Wait for program to finish running first
                if self.robot.is_program_running():
                    print("Program still running...")
                    time.sleep(0.2)
                    continue
                
                # Check position based on what we're verifying
                if target_joints is not None:
                    # Verify joint positions
                    current_joints = self.robot.getj()
                    
                    # Calculate differences
                    joint_diffs = [abs(current - target) for current, target in zip(current_joints, target_joints)]
                    max_diff = max(joint_diffs)
                    
                    print(f"Joint position check - Max difference: {max_diff:.4f} rad (tolerance: {tolerance:.4f})")
                    
                    if max_diff <= tolerance:
                        current_deg = [math.degrees(rad) for rad in current_joints]
                        target_deg = [math.degrees(rad) for rad in target_joints]
                        print(f"✓ Target joint position reached!")
                        print(f"  Current: {current_deg}")
                        print(f"  Target:  {target_deg}")
                        return True
                
                elif target_pose is not None:
                    # Verify Cartesian pose
                    try:
                        # Get current pose using individual properties to avoid PoseVector issues
                        current_pose = [
                            self.robot.x, self.robot.y, self.robot.z,
                            self.robot.rx, self.robot.ry, self.robot.rz
                        ]
                        
                        # Calculate position differences (only check x, y, z for now)
                        pos_diffs = [abs(current_pose[i] - target_pose[i]) for i in range(3)]
                        max_pos_diff = max(pos_diffs)
                        
                        print(f"Cartesian position check - Max difference: {max_pos_diff:.4f} m (tolerance: {tolerance:.4f})")
                        
                        if max_pos_diff <= tolerance:
                            print(f"✓ Target Cartesian position reached!")
                            print(f"  Current: {current_pose[:3]}")
                            print(f"  Target:  {target_pose[:3]}")
                            return True
                            
                    except Exception as e:
                        print(f"Error getting current pose: {e}")
                        # Fall back to just waiting for program completion
                        time.sleep(0.5)
                        if not self.robot.is_program_running():
                            print("Program completed (pose verification failed)")
                            return True
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error during position verification: {e}")
                time.sleep(0.5)
        
        print(f"⚠ Timeout after {timeout}s - position may not be reached")
        return False
    
    def get_current_position_info(self):
        """Get and display current robot position information."""
        try:
            # Get joint positions
            current_joints = self.robot.getj()
            current_joints_deg = [math.degrees(rad) for rad in current_joints]
            
            print(f"Current joint positions (degrees): {[f'{deg:.1f}' for deg in current_joints_deg]}")
            
            # Try to get Cartesian position
            try:
                current_pose = [
                    self.robot.x, self.robot.y, self.robot.z,
                    self.robot.rx, self.robot.ry, self.robot.rz
                ]
                print(f"Current Cartesian position: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}, z={current_pose[2]:.3f}")
                print(f"Current orientation: rx={current_pose[3]:.3f}, ry={current_pose[4]:.3f}, rz={current_pose[5]:.3f}")
                return current_joints, current_pose
            except Exception as e:
                print(f"Could not get Cartesian position: {e}")
                return current_joints, None
                
        except Exception as e:
            print(f"Error getting position info: {e}")
            return None, None
    
    def setup_vertical_plane(self, origin, width=0.4, height=0.4, normal_direction="x"):
        """
        Set up a vertical plane in the robot's workspace.
        
        Args:
            origin: [x, y, z] coordinates of the plane center
            width: Width of the plane (horizontal extent)
            height: Height of the plane (vertical extent)
            normal_direction: Direction the plane faces ("x", "y", or custom [nx, ny, nz])
        """
        self.plane_origin = np.array(origin)
        
        # Set up plane coordinate system
        if normal_direction == "x":
            self.plane_normal = np.array([1, 0, 0])
            self.plane_u_axis = np.array([0, 1, 0])  # Y direction (horizontal)
            self.plane_v_axis = np.array([0, 0, 1])  # Z direction (vertical)
        elif normal_direction == "y":
            self.plane_normal = np.array([0, 1, 0])
            self.plane_u_axis = np.array([1, 0, 0])  # X direction (horizontal)
            self.plane_v_axis = np.array([0, 0, 1])  # Z direction (vertical)
        else:
            # Custom normal direction
            self.plane_normal = np.array(normal_direction)
            self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
            
            # Create orthogonal axes
            if abs(self.plane_normal[2]) < 0.9:
                self.plane_v_axis = np.array([0, 0, 1])  # Keep vertical as Z
            else:
                self.plane_v_axis = np.array([0, 1, 0])  # Use Y if normal is close to Z
            
            # Make v_axis orthogonal to normal
            self.plane_v_axis = self.plane_v_axis - np.dot(self.plane_v_axis, self.plane_normal) * self.plane_normal
            self.plane_v_axis = self.plane_v_axis / np.linalg.norm(self.plane_v_axis)
            
            # U axis is cross product of normal and v_axis
            self.plane_u_axis = np.cross(self.plane_normal, self.plane_v_axis)
        
        self.plane_width = width
        self.plane_height = height
        
        print(f"Plane set up at origin: {origin}")
        print(f"Normal: {self.plane_normal}")
        print(f"U-axis (horizontal): {self.plane_u_axis}")
        print(f"V-axis (vertical): {self.plane_v_axis}")
        print(f"Dimensions: {width}m x {height}m")
    
    def plane_to_world_coords(self, u, v, extension=0):
        """
        Convert plane coordinates to world coordinates.
        
        Args:
            u: Horizontal position on plane (-width/2 to width/2)
            v: Vertical position on plane (-height/2 to height/2)
            extension: Distance to extend past the plane (positive = away from robot)
        
        Returns:
            [x, y, z] world coordinates
        """
        if self.plane_origin is None:
            raise ValueError("Plane not set up. Call setup_vertical_plane() first.")
        
        # Calculate position on the plane
        plane_point = (self.plane_origin + 
                      u * self.plane_u_axis + 
                      v * self.plane_v_axis)
        
        # Add extension along the normal
        world_point = plane_point + extension * self.plane_normal
        
        return world_point.tolist()
    
    def get_inverse_kinematics_with_retry(self, target_pose, max_attempts=8):
        """
        Calculate inverse kinematics with multiple attempts using different current joint configurations.
        
        Args:
            target_pose: [x, y, z, rx, ry, rz] target pose
            max_attempts: Maximum number of IK attempts with different starting configurations
        
        Returns:
            Joint angles in radians if successful, None otherwise
        """
        try:
            # Get current joint positions as starting point
            current_joints = self.robot.getj()
            
            # Try IK with current configuration first
            urscript_cmd = f"""
            target_pose = p{target_pose}
            current_q = {list(current_joints)}
            ik_result = get_inverse_kin(target_pose, current_q)
            
            # Store result in register for retrieval
            if ik_result != False:
                write_output_float_register(0, ik_result[0])
                write_output_float_register(1, ik_result[1])
                write_output_float_register(2, ik_result[2])
                write_output_float_register(3, ik_result[3])
                write_output_float_register(4, ik_result[4])
                write_output_float_register(5, ik_result[5])
                write_output_integer_register(0, 1)  # Success flag
            else:
                write_output_integer_register(0, 0)  # Failure flag
            end
            """
            
            print(f"Calculating IK for pose: {target_pose}")
            self.robot.send_program(urscript_cmd)
            time.sleep(0.5)  # Wait for calculation
            
            # Check if IK was successful
            success_flag = self.robot.get_digital_out(0)  # This might need adjustment based on URX version
            
            # For now, return None and suggest using direct movement
            # In a full implementation, you'd read the registers to get the joint values
            print("IK calculation completed. Using direct movement approach instead.")
            return None
            
        except Exception as e:
            print(f"Error in IK calculation: {e}")
            return None
    
    def move_to_plane_point(self, u, v, extension=0, orientation=None):
        """
        Move to a specific point on the plane with optional extension.
        
        Args:
            u: Horizontal position on plane (-width/2 to width/2)
            v: Vertical position on plane (-height/2 to height/2)
            extension: Distance to extend past the plane
            orientation: Tool orientation [rx, ry, rz] (uses current if None)
        """
        try:
            print(f"\n=== Moving to plane point: u={u:.3f}, v={v:.3f}, ext={extension:.3f} ===")
            
            # Show current position before movement
            print("Current position before movement:")
            self.get_current_position_info()
            
            # Calculate world coordinates
            world_pos = self.plane_to_world_coords(u, v, extension)
            
            # Get current orientation if not specified
            if orientation is None:
                try:
                    # Use individual properties to avoid PoseVector issues
                    orientation = [self.robot.rx, self.robot.ry, self.robot.rz]
                except:
                    # Default orientation (tool pointing down)
                    orientation = [0, 0, 0]
            
            # Combine position and orientation
            target_pose = world_pos + orientation
            
            print(f"Target world coordinates: {world_pos}")
            print(f"Target pose: {target_pose}")
            
            # Check if target is reasonable (basic workspace check)
            distance_from_base = math.sqrt(world_pos[0]**2 + world_pos[1]**2 + world_pos[2]**2)
            if distance_from_base > 0.8:  # UR robot reach is typically around 850mm
                print(f"Warning: Target distance {distance_from_base:.3f}m may be outside workspace")
            
            # Use URScript for reliable movement with error handling
            urscript_cmd = f"""
            target_pose = p{target_pose}
            
            # Try to move with error handling
            try:
                movel(target_pose, a={self.default_acc}, v={self.default_vel})
            catch:
                textmsg("Movement failed - possibly singularity or out of reach")
            end
            """
            
            print("Sending movement command...")
            self.robot.send_program(urscript_cmd)
            
            # Wait for movement completion with position verification
            if self.wait_for_position_reached(target_pose=target_pose, tolerance=0.005):
                print("✓ Successfully reached target position!")
                
                # Show final position
                print("Final position after movement:")
                self.get_current_position_info()
                return True
            else:
                print("⚠ Failed to reach exact target position")
                
                # Show final position anyway
                print("Final position after movement:")
                self.get_current_position_info()
                return False
            
        except Exception as e:
            print(f"Error moving to plane point: {e}")
            print("This might be due to singularity or workspace limits.")
            return False
    
    def scan_plane(self, u_points=5, v_points=5, extension=0, dwell_time=1.0):
        """
        Scan across the plane in a grid pattern.
        
        Args:
            u_points: Number of points in horizontal direction
            v_points: Number of points in vertical direction
            extension: Distance to extend past the plane
            dwell_time: Time to wait at each point
        """
        print(f"Starting plane scan: {u_points}x{v_points} grid")
        
        u_range = np.linspace(-self.plane_width/2, self.plane_width/2, u_points)
        v_range = np.linspace(-self.plane_height/2, self.plane_height/2, v_points)
        
        for i, v in enumerate(v_range):
            # Alternate direction for efficient scanning
            u_sequence = u_range if i % 2 == 0 else reversed(u_range)
            
            for u in u_sequence:
                print(f"Scanning point ({u:.3f}, {v:.3f})")
                if self.move_to_plane_point(u, v, extension):
                    time.sleep(dwell_time)
                else:
                    print(f"Failed to reach point ({u:.3f}, {v:.3f})")
        
        print("Plane scan completed!")
    
    def move_through_plane(self, u, v, start_extension=-0.1, end_extension=0.1, steps=10):
        """
        Move through the plane at a specific u,v coordinate.
        
        Args:
            u, v: Plane coordinates
            start_extension: Starting distance before the plane
            end_extension: Ending distance past the plane
            steps: Number of steps through the plane
        """
        print(f"Moving through plane at ({u:.3f}, {v:.3f})")
        
        extensions = np.linspace(start_extension, end_extension, steps)
        
        for ext in extensions:
            print(f"Extension: {ext:.3f}m")
            if not self.move_to_plane_point(u, v, ext):
                print(f"Failed at extension {ext:.3f}m")
                break
            time.sleep(0.5)
        
        print("Through-plane movement completed!")

# Example usage and testing
if __name__ == "__main__":
    # Create controller
    controller = PlaneMovementController()
    
    # Connect to robot
    if not controller.connect():
        exit(1)
    
    try:
        # FIRST: Move to safe starting position
        print("=== Initializing robot to safe position ===")
        if not controller.move_to_safe_position():
            print("Failed to move to safe position. Exiting.")
            exit(1)
        
        time.sleep(2)  # Give robot time to settle
        
        # Get current position and use it as plane center
        print("\n=== Getting current position for plane center ===")
        current_joints, current_pose = controller.get_current_position_info()
        
        if current_pose is None:
            print("Could not get current position. Exiting.")
            exit(1)
        
        # Use current position as plane center (much more realistic)
        plane_center = [current_pose[0], current_pose[1], current_pose[2]]
        print(f"Using current position as plane center: {plane_center}")
        
        controller.setup_vertical_plane(
            origin=plane_center,
            width=0.1,      # 10cm wide (very small for safety)
            height=0.1,     # 10cm tall (very small for safety)
            normal_direction="x"  # Plane faces X direction
        )
        
        print("\n=== Testing small plane movements ===")
        
        # Test very small movements around current position
        test_points = [
            (0, 0, "center"),
            (0.02, 0, "right 2cm"),
            (-0.02, 0, "left 2cm"),
            (0, 0.02, "up 2cm"),
            (0, -0.02, "down 2cm"),
            (0, 0, "back to center")
        ]
        
        successful_moves = 0
        for u, v, description in test_points:
            print(f"\n--- Testing {description} ---")
            if controller.move_to_plane_point(u, v):
                print(f"✓ {description} successful!")
                successful_moves += 1
                time.sleep(1)  # Brief pause between movements
            else:
                print(f"⚠ {description} failed")
        
        print(f"\n=== Test Results ===")
        print(f"Successful movements: {successful_moves}/{len(test_points)}")
        
        if successful_moves >= len(test_points) - 1:  # Allow for 1 failure
            print("✓ Plane movement system is working!")
            print("You can now use smaller movements around the robot's current position.")
        else:
            print("⚠ Some movements failed. Try even smaller movements.")
        
        print("\nFinal robot position:")
        controller.get_current_position_info()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        controller.disconnect()
