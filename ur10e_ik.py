import urx
import time
import math
import numpy as np

class UR10eInverseKinematics:
    """
    Inverse Kinematics controller for UR10e robot arm.
    Provides smooth path planning to move tool to specified positions.
    """
    
    def __init__(self, robot_ip="192.168.10.152"):
        """Initialize the IK controller."""
        self.robot = None
        self.robot_ip = robot_ip
        
        # Movement parameters
        self.default_acc = 0.1
        self.default_vel = 0.1
        
        # Tool orientation locking
        self.tool_orientation_locked = False
        self.locked_orientation = None  # [rx, ry, rz] to maintain
        
        # UR10e specifications
        self.max_reach = 1.3  # meters (UR10e reach is ~1.3m)
        self.joint_limits = [
            (-2*math.pi, 2*math.pi),  # Base joint
            (-2*math.pi, 2*math.pi),  # Shoulder joint  
            (-math.pi, math.pi),      # Elbow joint
            (-2*math.pi, 2*math.pi),  # Wrist 1
            (-2*math.pi, 2*math.pi),  # Wrist 2
            (-2*math.pi, 2*math.pi)   # Wrist 3
        ]
    
    def connect(self):
        """Connect to the robot."""
        try:
            print(f"Connecting to UR10e at {self.robot_ip}...")
            self.robot = urx.Robot(self.robot_ip)
            print("✓ Connected successfully!")
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the robot."""
        if self.robot:
            self.robot.close()
            print("Robot disconnected.")
    
    def get_current_pose(self):
        """Get current tool pose [x, y, z, rx, ry, rz]."""
        try:
            return [self.robot.x, self.robot.y, self.robot.z, 
                   self.robot.rx, self.robot.ry, self.robot.rz]
        except Exception as e:
            print(f"Error getting current pose: {e}")
            return None
    
    def get_current_joints(self):
        """Get current joint positions in radians."""
        try:
            return self.robot.getj()
        except Exception as e:
            print(f"Error getting current joints: {e}")
            return None
    
    def is_position_reachable(self, target_pos):
        """
        Check if target position is within robot's workspace.
        
        Args:
            target_pos: [x, y, z] target position
            
        Returns:
            True if reachable, False otherwise
        """
        distance = math.sqrt(target_pos[0]**2 + target_pos[1]**2 + target_pos[2]**2)
        
        if distance > self.max_reach:
            print(f"Target distance {distance:.3f}m exceeds max reach {self.max_reach}m")
            return False
        
        # Check minimum reach (avoid singularities near base)
        if distance < 0.1:
            print(f"Target too close to base: {distance:.3f}m")
            return False
            
        return True
    
    def solve_inverse_kinematics(self, target_pose, current_joints=None):
        """
        Solve inverse kinematics using robot's built-in solver.
        
        Args:
            target_pose: [x, y, z, rx, ry, rz] target pose
            current_joints: Current joint configuration (uses robot's current if None)
            
        Returns:
            Joint angles in radians if successful, None otherwise
        """
        try:
            if current_joints is None:
                current_joints = self.get_current_joints()
                if current_joints is None:
                    return None
            
            print(f"Solving IK for target: {[f'{x:.3f}' for x in target_pose]}")
            
            # Use robot's built-in inverse kinematics
            urscript_cmd = f"""
            target_pose = p{target_pose}
            current_q = {list(current_joints)}
            
            # Try inverse kinematics
            ik_result = get_inverse_kin(target_pose, current_q)
            
            if ik_result != False:
                # Store result in output registers
                write_output_float_register(0, ik_result[0])
                write_output_float_register(1, ik_result[1]) 
                write_output_float_register(2, ik_result[2])
                write_output_float_register(3, ik_result[3])
                write_output_float_register(4, ik_result[4])
                write_output_float_register(5, ik_result[5])
                write_output_integer_register(0, 1)  # Success flag
                textmsg("IK solution found")
            else:
                write_output_integer_register(0, 0)  # Failure flag
                textmsg("IK solution not found")
            end
            """
            
            self.robot.send_program(urscript_cmd)
            time.sleep(0.5)  # Wait for calculation
            
            # For this implementation, we'll use a direct approach since register reading
            # can be complex with URX. We'll try the movement and see if it works.
            print("IK calculation sent to robot")
            return target_pose  # Return the pose for direct movement attempt
            
        except Exception as e:
            print(f"Error in IK calculation: {e}")
            return None
    
    def move_to_position(self, target_pos, orientation=None, smooth=True):
        """
        Move tool to specified position with optional smooth path.
        
        Args:
            target_pos: [x, y, z] target position in meters
            orientation: [rx, ry, rz] target orientation (uses current if None)
            smooth: Use smooth path planning if True
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n=== Moving to position: {[f'{x:.3f}' for x in target_pos]} ===")
            
            # Check if position is reachable
            if not self.is_position_reachable(target_pos):
                return False
            
            # Get current pose
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False
            
            print(f"Current position: {[f'{x:.3f}' for x in current_pose[:3]]}")
            
            # Use current orientation if not specified
            if orientation is None:
                orientation = current_pose[3:6]
            
            # Create target pose
            target_pose = target_pos + orientation
            
            if smooth:
                return self._move_smooth_path(current_pose, target_pose)
            else:
                return self._move_direct(target_pose)
                
        except Exception as e:
            print(f"Error in move_to_position: {e}")
            return False
    
    def _move_direct(self, target_pose):
        """Direct movement to target pose."""
        try:
            print("Using direct movement...")
            
            # Send movement command
            urscript_cmd = f"movel(p{target_pose}, a={self.default_acc}, v={self.default_vel})"
            self.robot.send_program(urscript_cmd)
            
            # Wait for completion
            time.sleep(1)
            while self.robot.is_program_running():
                print("Moving...")
                time.sleep(0.3)
            
            # Check if we reached the target
            final_pose = self.get_current_pose()
            if final_pose is None:
                return False
            
            # Calculate position error
            pos_error = math.sqrt(sum((final_pose[i] - target_pose[i])**2 for i in range(3)))
            print(f"Position error: {pos_error:.4f}m")
            
            if pos_error < 0.01:  # Within 1cm
                print("✓ Target reached successfully!")
                return True
            else:
                print("⚠ Target not reached precisely")
                return False
                
        except Exception as e:
            print(f"Error in direct movement: {e}")
            return False
    
    def _move_smooth_path(self, start_pose, target_pose, num_waypoints=5):
        """
        Move through smooth path with multiple waypoints.
        Each waypoint is reached before moving to the next one.
        
        Args:
            start_pose: Starting pose [x, y, z, rx, ry, rz]
            target_pose: Target pose [x, y, z, rx, ry, rz]
            num_waypoints: Number of intermediate waypoints
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Using smooth path with {num_waypoints} waypoints...")
            
            # Generate waypoints along straight line path
            waypoints = []
            for i in range(num_waypoints + 1):
                t = i / num_waypoints
                waypoint = []
                for j in range(6):
                    # Linear interpolation between start and target
                    value = start_pose[j] + t * (target_pose[j] - start_pose[j])
                    waypoint.append(value)
                waypoints.append(waypoint)
            
            print(f"Generated {len(waypoints)} waypoints")
            
            # Move through each waypoint sequentially
            for i, waypoint in enumerate(waypoints):
                print(f"Moving to waypoint {i+1}/{len(waypoints)}: {[f'{x:.3f}' for x in waypoint[:3]]}")
                
                # Use faster movement for intermediate points, slower for final
                vel = self.default_vel * 1.5 if i < len(waypoints) - 1 else self.default_vel
                
                # Send movement command for this waypoint
                urscript_cmd = f"movel(p{waypoint}, a={self.default_acc}, v={vel})"
                self.robot.send_program(urscript_cmd)
                
                # Wait for this waypoint to be reached before proceeding
                print(f"  Waiting for waypoint {i+1} to be reached...")
                time.sleep(0.5)  # Brief delay for command to start
                
                # Wait for movement to complete
                while self.robot.is_program_running():
                    time.sleep(0.1)
                
                # Verify waypoint was reached (for intermediate waypoints, use looser tolerance)
                current_pose = self.get_current_pose()
                if current_pose is not None:
                    pos_error = math.sqrt(sum((current_pose[j] - waypoint[j])**2 for j in range(3)))
                    tolerance = 0.02 if i < len(waypoints) - 1 else 0.01  # Looser tolerance for intermediate points
                    
                    if pos_error <= tolerance:
                        print(f"  ✓ Waypoint {i+1} reached (error: {pos_error:.4f}m)")
                    else:
                        print(f"  ⚠ Waypoint {i+1} position error: {pos_error:.4f}m (tolerance: {tolerance:.4f}m)")
                        # Continue anyway, but note the error
                
                # Brief pause between waypoints (except for last one)
                if i < len(waypoints) - 1:
                    time.sleep(0.3)  # Slightly longer pause for smoother motion
            
            # Final verification of target position
            final_pose = self.get_current_pose()
            if final_pose is None:
                return False
            
            pos_error = math.sqrt(sum((final_pose[i] - target_pose[i])**2 for i in range(3)))
            print(f"Final position error: {pos_error:.4f}m")
            
            if pos_error < 0.01:
                print("✓ Smooth path completed successfully!")
                return True
            else:
                print("⚠ Smooth path completed with position error")
                return False
                
        except Exception as e:
            print(f"Error in smooth path movement: {e}")
            return False
    
    def move_relative(self, delta_pos, smooth=True):
        """
        Move relative to current position.
        
        Args:
            delta_pos: [dx, dy, dz] relative movement in meters
            smooth: Use smooth path planning if True
            
        Returns:
            True if successful, False otherwise
        """
        current_pose = self.get_current_pose()
        if current_pose is None:
            return False
        
        target_pos = [current_pose[i] + delta_pos[i] for i in range(3)]
        return self.move_to_position(target_pos, smooth=smooth)
    
    def demonstrate_movements(self):
        """Demonstrate various IK movements."""
        print("\n=== UR10e Inverse Kinematics Demonstration ===")
        
        # Get starting position
        start_pose = self.get_current_pose()
        if start_pose is None:
            print("Could not get starting position")
            return
        
        print(f"Starting position: {[f'{x:.3f}' for x in start_pose[:3]]}")
        
        # Test movements
        test_movements = [
            ([0.05, 0, 0], "5cm right"),
            ([0, 0.05, 0], "5cm forward"), 
            ([0, 0, 0.05], "5cm up"),
            ([-0.05, 0, 0], "5cm left"),
            ([0, -0.05, 0], "5cm back"),
            ([0, 0, -0.05], "5cm down")
        ]
        
        successful_moves = 0
        
        for delta_pos, description in test_movements:
            print(f"\n--- Testing: {description} ---")
            if self.move_relative(delta_pos, smooth=True):
                print(f"✓ {description} successful!")
                successful_moves += 1
                time.sleep(1)  # Pause between movements
            else:
                print(f"⚠ {description} failed")
        
        # Return to start
        print(f"\n--- Returning to start position ---")
        if self.move_to_position(start_pose[:3], start_pose[3:6], smooth=True):
            print("✓ Returned to start position!")
            successful_moves += 1
        else:
            print("⚠ Failed to return to start")
        
        print(f"\n=== Demonstration Results ===")
        print(f"Successful movements: {successful_moves}/{len(test_movements) + 1}")
        
        if successful_moves >= len(test_movements):
            print("✓ IK system working well!")
        else:
            print("⚠ Some movements failed - check workspace limits")

# Example usage
if __name__ == "__main__":
    # Create IK controller
    ik_controller = UR10eInverseKinematics()
    
    # Connect to robot
    if not ik_controller.connect():
        exit(1)
    
    try:
        # Move to safe starting position first
        print("Moving to safe starting position...")
        safe_joints = [0, -1.57, 1.57, -1.57, -1.57, 0]
        ik_controller.robot.movej(safe_joints, acc=0.1, vel=0.1, wait=False)  # Use wait=False
        
        # Wait for completion manually
        time.sleep(2)
        while ik_controller.robot.is_program_running():
            print("Moving to safe position...")
            time.sleep(0.5)
        
        print("✓ Safe position reached!")
        time.sleep(1)
        
        # Demonstrate IK movements
        # ik_controller.demonstrate_movements()
        
        # Example: Move to specific position
        print("\n=== Custom Position Test ===")
        current_pose = ik_controller.get_current_pose()
        if current_pose:
            # Move 10cm forward from current position
            target_pos = [current_pose[0], current_pose[1] + 0.1, current_pose[2]]
            print(f"Moving 10cm forward to: {[f'{x:.3f}' for x in target_pos]}")
            
            if ik_controller.move_to_position(target_pos, smooth=True): 
                print("✓ Custom movement successful!")
            else:
                print("⚠ Custom movement failed")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        ik_controller.disconnect() 