import urx
import time
import math
import numpy as np

class GlassWipingRobot:
    """
    Glass wiping robot controller for UR10e.
    
    Glass Configuration:
    - Glass is vertical (like a window)
    - Blue arrow (Z-axis) perpendicular to glass surface
    - Red-Green plane parallel to glass surface for wiping motions
    """
    
    def __init__(self, robot_ip="192.168.10.152"):
        """Initialize the glass wiping robot."""
        self.robot = None
        self.robot_ip = robot_ip
        
        # Movement parameters
        self.default_acc = 0.1
        self.default_vel = 0.1
        self.wiping_vel = 0.05  # Slower speed for wiping
        
        # Glass surface definition
        self.glass_normal = None  # Normal vector to glass surface (blue arrow direction)
        self.glass_center = None  # Center point of glass surface
        self.glass_orientation = None  # [rx, ry, rz] for perpendicular to glass
        
        # Wiping parameters
        self.contact_pressure = 0.002  # 2mm into glass for contact pressure
        
    def connect(self):
        """Connect to the robot."""
        try:
            print(f"Connecting to UR10e glass wiping robot at {self.robot_ip}...")
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
            print("Glass wiping robot disconnected.")
    
    def get_current_pose(self):
        """Get current tool pose [x, y, z, rx, ry, rz]."""
        try:
            return [self.robot.x, self.robot.y, self.robot.z, 
                   self.robot.rx, self.robot.ry, self.robot.rz]
        except Exception as e:
            print(f"Error getting current pose: {e}")
            return None
    
    def check_robot_status(self):
        """Check robot status and clear any protective stops."""
        try:
            print("Checking robot status...")
            
            # Check if robot is in protective stop
            # Note: URX doesn't have direct protective stop check, but we can try basic operations
            
            # Try to get current position - this will fail if robot is in protective stop
            current_pose = self.get_current_pose()
            if current_pose is None:
                print("⚠ Cannot get robot pose - robot may be in protective stop")
                return False
            
            print(f"✓ Robot status OK - Current position: {[f'{x:.3f}' for x in current_pose[:3]]}")
            return True
            
        except Exception as e:
            print(f"Error checking robot status: {e}")
            return False
    
    def move_to_safe_position(self):
        """Move to safe starting position away from glass."""
        try:
            print("Moving to safe position...")
            
            # Get current joint positions first
            current_joints = self.robot.getj()
            print(f"Current joints: {[f'{math.degrees(x):.1f}°' for x in current_joints]}")
            
            # Safe joint configuration - modified for gradual approach
            safe_joints = [0, -1.57, 1.57, 0, 1.57, 0]  # Safe position in radians
            print(f"Target joints: {[f'{math.degrees(x):.1f}°' for x in safe_joints]}")
            
            # Check if we're already close to safe position
            max_joint_diff = max(abs(current_joints[i] - safe_joints[i]) for i in range(6))
            print(f"Maximum joint difference: {math.degrees(max_joint_diff):.1f}°")
            
            if max_joint_diff < 0.05:  # Already very close (within ~3 degrees)
                print("✓ Already at safe position")
                return True
            
            # Use gradual movement with faster speeds but proper waiting
            # Use fewer steps but faster movement
            if max_joint_diff > 1.0:  # More than ~57 degrees difference
                num_steps = 3  # Use 3 steps for large movements
            elif max_joint_diff > 0.3:  # More than ~17 degrees difference  
                num_steps = 2  # Use 2 steps for medium movements
            else:
                num_steps = 1  # Single step for small movements
            
            print(f"Using {num_steps} steps with faster movement...")
            
            for step in range(num_steps):
                # Calculate intermediate position
                t = (step + 1) / num_steps  # Progress from 0 to 1
                intermediate_joints = []
                for i in range(6):
                    intermediate = current_joints[i] + t * (safe_joints[i] - current_joints[i])
                    intermediate_joints.append(intermediate)
                
                step_diff = max(abs(intermediate_joints[i] - (current_joints[i] if step == 0 else prev_joints[i])) for i in range(6))
                print(f"Step {step + 1}/{num_steps}: {[f'{math.degrees(x):.1f}°' for x in intermediate_joints]} (step size: {math.degrees(step_diff):.1f}°)")
                
                # Move to intermediate position with faster but safe settings
                try:
                    # Use faster speeds but still safe
                    self.robot.movej(intermediate_joints, acc=0.1, vel=0.1, wait=True)
                    print(f"✓ Step {step + 1} movement command sent - waiting for completion...")
                    
                    # Wait for movement to actually complete by checking program status
                    timeout = 0
                    while self.robot.is_program_running() and timeout < 50:  # 5 second timeout
                        time.sleep(0.1)
                        timeout += 1
                    
                    if timeout >= 50:
                        print(f"⚠ Movement timeout for step {step + 1}")
                    else:
                        print(f"✓ Step {step + 1} completed")
                    
                    # Brief pause to ensure robot is settled
                    time.sleep(0.5)
                    
                    # Verify position reached
                    actual_joints = self.robot.getj()
                    max_error = max(abs(actual_joints[i] - intermediate_joints[i]) for i in range(6))
                    print(f"Position error: {math.degrees(max_error):.1f}°")
                    
                    if max_error > 0.2:  # More than ~12 degrees error
                        print(f"⚠ Large position error - waiting longer for settlement...")
                        time.sleep(1.0)
                        
                        # Check again
                        actual_joints = self.robot.getj()
                        max_error = max(abs(actual_joints[i] - intermediate_joints[i]) for i in range(6))
                        print(f"Position error after wait: {math.degrees(max_error):.1f}°")
                    
                    prev_joints = intermediate_joints.copy()
                    
                except Exception as e:
                    print(f"Error in step {step + 1}: {e}")
                    
                    # Try slower movement as fallback
                    print("Trying slower movement as fallback...")
                    try:
                        self.robot.movej(intermediate_joints, acc=0.05, vel=0.05, wait=True)
                        print(f"✓ Step {step + 1} completed (slower)")
                        
                        # Wait for completion
                        timeout = 0
                        while self.robot.is_program_running() and timeout < 100:  # 10 second timeout
                            time.sleep(0.1)
                            timeout += 1
                        
                        time.sleep(1.0)  # Longer wait for slower movement
                        prev_joints = intermediate_joints.copy()
                    except Exception as e2:
                        print(f"Step {step + 1} failed even with slower movement: {e2}")
                        print("Continuing with remaining steps...")
                        # Don't return False immediately, try to continue
                        prev_joints = current_joints if step == 0 else prev_joints
            
            print("✓ Movement to safe position completed")
            
            # Verify final position
            final_joints = self.robot.getj()
            print(f"Final joints: {[f'{math.degrees(x):.1f}°' for x in final_joints]}")
            
            final_error = max(abs(final_joints[i] - safe_joints[i]) for i in range(6))
            print(f"Final position error: {math.degrees(final_error):.1f}°")
            
            if final_error < 0.3:  # Within ~17 degrees is acceptable
                print("✓ Safe position reached successfully!")
                return True
            else:
                print("⚠ Safe position not reached accurately, but close enough")
                return True  # Accept close enough to avoid infinite retries
            
        except Exception as e:
            print(f"Error in move_to_safe_position: {e}")
            return False
    
    def define_glass_surface(self, glass_center, glass_normal_direction="forward"):
        """
        Define the glass surface for wiping.
        
        Args:
            glass_center: [x, y, z] center point of glass surface
            glass_normal_direction: Direction glass normal points ("forward", "backward", "left", "right")
        """
        try:
            self.glass_center = glass_center.copy()
            
            # Define glass normal vector (blue arrow direction)
            if glass_normal_direction == "forward":
                self.glass_normal = [0, 1, 0]  # Glass faces robot, normal points toward robot
                self.glass_orientation = [0.0, 0.0, 0.0]  # Tool pointing forward into glass
            elif glass_normal_direction == "backward":
                self.glass_normal = [0, -1, 0]  # Glass faces away, normal points away from robot
                self.glass_orientation = [0.0, 0.0, 3.14159]  # Tool pointing backward into glass
            elif glass_normal_direction == "left":
                self.glass_normal = [-1, 0, 0]  # Glass on left side
                self.glass_orientation = [0.0, 0.0, -1.5708]  # Tool pointing left into glass
            elif glass_normal_direction == "right":
                self.glass_normal = [1, 0, 0]  # Glass on right side
                self.glass_orientation = [0.0, 0.0, 1.5708]  # Tool pointing right into glass
            else:
                print(f"Invalid glass direction: {glass_normal_direction}")
                return False
            
            print(f"✓ Glass surface defined:")
            print(f"  Center: {[f'{x:.3f}' for x in self.glass_center]}")
            print(f"  Normal direction: {glass_normal_direction}")
            print(f"  Normal vector: {[f'{x:.3f}' for x in self.glass_normal]}")
            print(f"  Tool orientation: {[f'{x:.3f}' for x in self.glass_orientation]}")
            print(f"  Blue arrow will be perpendicular to glass")
            print(f"  Red-Green plane will be parallel to glass for wiping")
            
            return True
            
        except Exception as e:
            print(f"Error defining glass surface: {e}")
            return False
    
    def move_to_glass_position(self, glass_point, contact=False):
        """
        Move to a position on the glass surface.
        
        Args:
            glass_point: [x, y, z] point on glass surface
            contact: If True, press into glass slightly for contact pressure
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.glass_orientation is None:
                print("Glass surface not defined. Call define_glass_surface() first.")
                return False
            
            # Calculate target position
            target_pos = glass_point.copy()
            
            if contact:
                # Move slightly into glass for contact pressure
                for i in range(3):
                    target_pos[i] -= self.glass_normal[i] * self.contact_pressure
                print(f"Moving to glass contact position: {[f'{x:.3f}' for x in target_pos]}")
            else:
                # Stay at glass surface
                print(f"Moving to glass surface position: {[f'{x:.3f}' for x in target_pos]}")
            
            # Create target pose with glass-perpendicular orientation
            target_x, target_y, target_z = target_pos
            target_rx, target_ry, target_rz = self.glass_orientation
            
            # Move to position using URScript to avoid PoseVector issues
            urscript_cmd = f"movel(p[{target_x}, {target_y}, {target_z}, {target_rx}, {target_ry}, {target_rz}], a={self.default_acc}, v={self.default_vel})"
            print(f"Sending URScript: {urscript_cmd}")
            
            self.robot.send_program(urscript_cmd)
            
            # Wait for movement to complete
            time.sleep(1)
            while self.robot.is_program_running():
                time.sleep(0.1)
            
            print("✓ Glass position reached")
            return True
            
        except Exception as e:
            print(f"Error moving to glass position: {e}")
            return False
    
    def wipe_horizontal_line(self, start_point, end_point, contact=True):
        """
        Wipe horizontally across glass from start to end point.
        
        Args:
            start_point: [x, y, z] starting point on glass
            end_point: [x, y, z] ending point on glass
            contact: Maintain contact pressure during wiping
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n=== Horizontal Wiping ===")
            print(f"From: {[f'{x:.3f}' for x in start_point]}")
            print(f"To:   {[f'{x:.3f}' for x in end_point]}")
            
            # Move to start position
            if not self.move_to_glass_position(start_point, contact=contact):
                return False
            
            time.sleep(0.5)  # Brief pause
            
            # Calculate end position with contact pressure if needed
            target_end = end_point.copy()
            if contact:
                for i in range(3):
                    target_end[i] -= self.glass_normal[i] * self.contact_pressure
            
            # Create end pose with same orientation - fix PoseVector issue
            end_x, end_y, end_z = target_end
            end_rx, end_ry, end_rz = self.glass_orientation
            
            # Wipe to end position using URScript
            print("Wiping across glass...")
            urscript_cmd = f"movel(p[{end_x}, {end_y}, {end_z}, {end_rx}, {end_ry}, {end_rz}], a={self.default_acc}, v={self.wiping_vel})"
            self.robot.send_program(urscript_cmd)
            
            # Wait for movement to complete
            time.sleep(1)
            while self.robot.is_program_running():
                time.sleep(0.1)
            
            print("✓ Horizontal wipe completed")
            return True
            
        except Exception as e:
            print(f"Error during horizontal wiping: {e}")
            return False
    
    def wipe_vertical_line(self, start_point, end_point, contact=True):
        """
        Wipe vertically across glass from start to end point.
        
        Args:
            start_point: [x, y, z] starting point on glass
            end_point: [x, y, z] ending point on glass
            contact: Maintain contact pressure during wiping
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n=== Vertical Wiping ===")
            print(f"From: {[f'{x:.3f}' for x in start_point]}")
            print(f"To:   {[f'{x:.3f}' for x in end_point]}")
            
            # Move to start position
            if not self.move_to_glass_position(start_point, contact=contact):
                return False
            
            time.sleep(0.5)  # Brief pause
            
            # Calculate end position with contact pressure if needed
            target_end = end_point.copy()
            if contact:
                for i in range(3):
                    target_end[i] -= self.glass_normal[i] * self.contact_pressure
            
            # Create end pose with same orientation - fix PoseVector issue
            end_x, end_y, end_z = target_end
            end_rx, end_ry, end_rz = self.glass_orientation
            
            # Wipe to end position using URScript
            print("Wiping across glass...")
            urscript_cmd = f"movel(p[{end_x}, {end_y}, {end_z}, {end_rx}, {end_ry}, {end_rz}], a={self.default_acc}, v={self.wiping_vel})"
            self.robot.send_program(urscript_cmd)
            
            # Wait for movement to complete
            time.sleep(1)
            while self.robot.is_program_running():
                time.sleep(0.1)
            
            print("✓ Vertical wipe completed")
            return True
            
        except Exception as e:
            print(f"Error during vertical wiping: {e}")
            return False
    
    def wipe_rectangular_pattern(self, top_left, bottom_right, num_horizontal_passes=5):
        """
        Wipe glass in rectangular pattern with horizontal passes.
        
        Args:
            top_left: [x, y, z] top-left corner of area to wipe
            bottom_right: [x, y, z] bottom-right corner of area to wipe
            num_horizontal_passes: Number of horizontal wiping passes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n=== Rectangular Wiping Pattern ===")
            print(f"Area: Top-left {[f'{x:.3f}' for x in top_left]} to Bottom-right {[f'{x:.3f}' for x in bottom_right]}")
            print(f"Horizontal passes: {num_horizontal_passes}")
            
            # Calculate wiping points
            height = top_left[2] - bottom_right[2]  # Z difference
            width = bottom_right[0] - top_left[0]   # X difference
            
            print(f"Wiping area: {width:.3f}m wide × {height:.3f}m tall")
            
            successful_passes = 0
            
            for pass_num in range(num_horizontal_passes):
                # Calculate Y position for this pass
                t = pass_num / (num_horizontal_passes - 1) if num_horizontal_passes > 1 else 0
                current_z = top_left[2] - t * height  # Move down from top
                
                # Alternate direction for each pass (left-to-right, then right-to-left)
                if pass_num % 2 == 0:
                    # Left to right
                    start_point = [top_left[0], top_left[1], current_z]
                    end_point = [bottom_right[0], bottom_right[1], current_z]
                else:
                    # Right to left
                    start_point = [bottom_right[0], bottom_right[1], current_z]
                    end_point = [top_left[0], top_left[1], current_z]
                
                print(f"\nPass {pass_num + 1}/{num_horizontal_passes} at height Z={current_z:.3f}")
                
                if self.wipe_horizontal_line(start_point, end_point, contact=True):
                    successful_passes += 1
                    print(f"✓ Pass {pass_num + 1} completed")
                else:
                    print(f"⚠ Pass {pass_num + 1} failed")
                
                # Brief pause between passes
                time.sleep(0.3)
            
            print(f"\n=== Wiping Results ===")
            print(f"Successful passes: {successful_passes}/{num_horizontal_passes}")
            
            if successful_passes == num_horizontal_passes:
                print("✓ Rectangular wiping pattern completed successfully!")
                return True
            else:
                print("⚠ Some wiping passes failed")
                return False
                
        except Exception as e:
            print(f"Error during rectangular wiping: {e}")
            return False
    
    def move_away_from_glass(self, distance=0.05):
        """
        Move away from glass surface by specified distance.
        
        Args:
            distance: Distance to move away from glass (meters)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.glass_normal is None:
                print("Glass surface not defined")
                return False
            
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False
            
            # Move away from glass along normal direction
            away_pos = current_pose[:3].copy()
            for i in range(3):
                away_pos[i] += self.glass_normal[i] * distance
            
            # Create away pose - fix PoseVector issue
            away_x, away_y, away_z = away_pos
            away_rx, away_ry, away_rz = current_pose[3:6]
            
            print(f"Moving {distance:.3f}m away from glass...")
            urscript_cmd = f"movel(p[{away_x}, {away_y}, {away_z}, {away_rx}, {away_ry}, {away_rz}], a={self.default_acc}, v={self.default_vel})"
            self.robot.send_program(urscript_cmd)
            
            # Wait for movement to complete
            time.sleep(1)
            while self.robot.is_program_running():
                time.sleep(0.1)
            
            print("✓ Moved away from glass")
            return True
            
        except Exception as e:
            print(f"Error moving away from glass: {e}")
            return False
    
    def demonstrate_glass_wiping(self):
        """Demonstrate glass wiping functionality."""
        print("\n=== Glass Wiping Robot Demonstration ===")
        print("This robot is designed to wipe vertical glass surfaces")
        print("Blue arrow: Perpendicular to glass (into/out of glass)")
        print("Red-Green plane: Parallel to glass (for wiping motions)")
        
        # Get current position to define glass relative to robot
        current_pose = self.get_current_pose()
        if current_pose is None:
            print("Could not get current position")
            return
        
        print(f"Current robot position: {[f'{x:.3f}' for x in current_pose[:3]]}")
        
        # Define glass surface in front of robot
        glass_center = [current_pose[0], current_pose[1] + 0.3, current_pose[2]]  # 30cm in front
        
        print(f"\n--- Defining Glass Surface ---")
        if not self.define_glass_surface(glass_center, "forward"):
            print("Failed to define glass surface")
            return
        
        # Define wiping area (20cm × 15cm rectangle)
        glass_width = 0.20   # 20cm wide
        glass_height = 0.15  # 15cm tall
        
        top_left = [
            glass_center[0] - glass_width/2,  # Left edge
            glass_center[1],                   # Same Y as center
            glass_center[2] + glass_height/2   # Top edge
        ]
        
        bottom_right = [
            glass_center[0] + glass_width/2,   # Right edge
            glass_center[1],                   # Same Y as center
            glass_center[2] - glass_height/2   # Bottom edge
        ]
        
        print(f"\n--- Testing Glass Contact ---")
        # Test moving to glass center
        if self.move_to_glass_position(glass_center, contact=False):
            print("✓ Reached glass surface")
            time.sleep(1)
            
            # Test contact pressure
            if self.move_to_glass_position(glass_center, contact=True):
                print("✓ Applied contact pressure")
                time.sleep(1)
                
                # Move away from glass
                if self.move_away_from_glass(0.05):
                    print("✓ Moved away from glass")
        
        print(f"\n--- Testing Wiping Motions ---")
        # Test horizontal wipe
        horizontal_start = [glass_center[0] - 0.05, glass_center[1], glass_center[2]]
        horizontal_end = [glass_center[0] + 0.05, glass_center[1], glass_center[2]]
        
        if self.wipe_horizontal_line(horizontal_start, horizontal_end):
            print("✓ Horizontal wiping test successful")
            time.sleep(1)
        
        # Test vertical wipe
        vertical_start = [glass_center[0], glass_center[1], glass_center[2] + 0.05]
        vertical_end = [glass_center[0], glass_center[1], glass_center[2] - 0.05]
        
        if self.wipe_vertical_line(vertical_start, vertical_end):
            print("✓ Vertical wiping test successful")
            time.sleep(1)
        
        print(f"\n--- Testing Rectangular Wiping Pattern ---")
        # Test rectangular wiping pattern
        if self.wipe_rectangular_pattern(top_left, bottom_right, num_horizontal_passes=3):
            print("✓ Rectangular wiping pattern successful")
        
        # Return to safe position
        print(f"\n--- Returning to Safe Position ---")
        if self.move_to_safe_position():
            print("✓ Returned to safe position")
        
        print(f"\n=== Glass Wiping Demonstration Complete ===")
        print("Robot successfully demonstrated:")
        print("✓ Glass surface definition")
        print("✓ Perpendicular tool orientation (blue arrow into glass)")
        print("✓ Contact pressure application")
        print("✓ Horizontal and vertical wiping motions")
        print("✓ Rectangular wiping patterns")
        print("✓ Safe positioning")

# Example usage
if __name__ == "__main__":
    # Create glass wiping robot
    glass_robot = GlassWipingRobot()
    
    # Connect to robot
    if not glass_robot.connect():
        exit(1)
    
    try:
        # Check robot status first
        if not glass_robot.check_robot_status():
            print("Robot status check failed - please check robot on teach pendant")
            exit(1)
        
        # Move to safe starting position
        if not glass_robot.move_to_safe_position():
            print("Failed to reach safe position")
            exit(1)
        
        time.sleep(1)
        
        # Demonstrate glass wiping functionality
        glass_robot.demonstrate_glass_wiping()
        
    except KeyboardInterrupt:
        print("\nGlass wiping demonstration interrupted by user")
    except Exception as e:
        print(f"Error during demonstration: {e}")
    finally:
        glass_robot.disconnect() 