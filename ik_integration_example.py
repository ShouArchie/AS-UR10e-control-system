"""
Example integration of custom IK solver with UR robot face tracking system.
This shows how to use constrained IK for specific wrist configurations.
"""

import urx
import math
import time
from ur_ik_solver import URIKSolver

class ConstrainedFaceTracking:
    """
    Face tracking system with constrained wrist joints using custom IK solver.
    """
    
    def __init__(self, robot_ip="192.168.0.100", robot_model="ur10e"):
        self.robot_ip = robot_ip
        self.robot = None
        
        # Initialize custom IK solver
        self.ik_solver = URIKSolver(robot_model)
        
        # Wrist constraints (fixed angles in radians)
        self.wrist1_fixed_angle = math.radians(-90)  # -90 degrees
        self.wrist3_fixed_angle = math.radians(0)    # 0 degrees
        
        print(f"✓ Constrained face tracking initialized for {robot_model.upper()}")
        print(f"✓ Wrist1 fixed at: {math.degrees(self.wrist1_fixed_angle):.1f}°")
        print(f"✓ Wrist3 fixed at: {math.degrees(self.wrist3_fixed_angle):.1f}°")
    
    def connect_robot(self):
        """Connect to the robot."""
        try:
            print(f"Connecting to robot at {self.robot_ip}...")
            self.robot = urx.Robot(self.robot_ip)
            print("✓ Robot connected successfully!")
            return True
        except Exception as e:
            print(f"✗ Robot connection failed: {e}")
            return False
    
    def disconnect_robot(self):
        """Disconnect from the robot."""
        if self.robot:
            self.robot.close()
            print("Robot disconnected.")
    
    def get_current_pose(self):
        """Get current robot pose."""
        return [self.robot.x, self.robot.y, self.robot.z, 
                self.robot.rx, self.robot.ry, self.robot.rz]
    
    def get_current_joints(self):
        """Get current joint angles."""
        return self.robot.getj()
    
    def move_to_pose_with_constrained_wrists(self, target_position, target_orientation):
        """
        Move to target pose while keeping wrist1 and wrist3 at fixed angles.
        
        Args:
            target_position: [x, y, z] in meters
            target_orientation: [rx, ry, rz] in radians
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current joint positions for initial guess
            current_joints = self.get_current_joints()
            
            # Solve IK with wrist constraints
            solution = self.ik_solver.solve_constrained_wrist_ik(
                target_position=target_position,
                target_orientation=target_orientation,
                wrist1_angle=self.wrist1_fixed_angle,
                wrist3_angle=self.wrist3_fixed_angle,
                current_joints=current_joints
            )
            
            if solution is None:
                print("✗ No IK solution found for target pose with wrist constraints")
                return False
            
            print(f"✓ IK solution found:")
            print(f"  Joint angles: {[f'{math.degrees(j):.1f}°' for j in solution]}")
            print(f"  Wrist1: {math.degrees(solution[3]):.1f}° (fixed)")
            print(f"  Wrist3: {math.degrees(solution[5]):.1f}° (fixed)")
            
            # Execute movement
            print("Moving robot to target pose...")
            self.robot.movej(solution, acc=0.3, vel=0.2)
            
            # Wait for movement completion
            time.sleep(1)
            while self.robot.is_program_running():
                time.sleep(0.1)
            
            print("✓ Movement completed")
            return True
            
        except Exception as e:
            print(f"✗ Error in constrained movement: {e}")
            return False
    
    def demonstrate_constrained_movements(self):
        """Demonstrate various movements with constrained wrists."""
        print("\n=== Demonstrating Constrained Wrist Movements ===")
        
        # Get starting pose
        start_pose = self.get_current_pose()
        start_pos = start_pose[:3]
        start_orient = start_pose[3:]
        
        print(f"Starting position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        
        # Define some target poses to test
        test_poses = [
            # Move forward 10cm
            ([start_pos[0] + 0.1, start_pos[1], start_pos[2]], start_orient),
            
            # Move right 10cm
            ([start_pos[0], start_pos[1] + 0.1, start_pos[2]], start_orient),
            
            # Move up 10cm  
            ([start_pos[0], start_pos[1], start_pos[2] + 0.1], start_orient),
            
            # Move to a different orientation (30 degree rotation around Z)
            (start_pos, [start_orient[0], start_orient[1], start_orient[2] + math.radians(30)]),
            
            # Return to start
            (start_pos, start_orient)
        ]
        
        for i, (target_pos, target_orient) in enumerate(test_poses):
            print(f"\n--- Test Movement {i+1} ---")
            print(f"Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"Target orientation: [{math.degrees(target_orient[0]):.1f}°, {math.degrees(target_orient[1]):.1f}°, {math.degrees(target_orient[2]):.1f}°]")
            
            success = self.move_to_pose_with_constrained_wrists(target_pos, target_orient)
            
            if success:
                # Verify final pose
                final_pose = self.get_current_pose()
                final_joints = self.get_current_joints()
                
                print(f"✓ Final position: [{final_pose[0]:.3f}, {final_pose[1]:.3f}, {final_pose[2]:.3f}]")
                print(f"✓ Wrist1 actual: {math.degrees(final_joints[3]):.1f}°")
                print(f"✓ Wrist3 actual: {math.degrees(final_joints[5]):.1f}°")
                
                time.sleep(2)  # Pause between movements
            else:
                print("✗ Movement failed, skipping to next")
    
    def update_wrist_constraints(self, wrist1_angle_deg, wrist3_angle_deg):
        """
        Update the fixed wrist angles.
        
        Args:
            wrist1_angle_deg: New wrist1 angle in degrees
            wrist3_angle_deg: New wrist3 angle in degrees
        """
        self.wrist1_fixed_angle = math.radians(wrist1_angle_deg)
        self.wrist3_fixed_angle = math.radians(wrist3_angle_deg)
        
        print(f"✓ Wrist constraints updated:")
        print(f"  Wrist1: {wrist1_angle_deg:.1f}°")
        print(f"  Wrist3: {wrist3_angle_deg:.1f}°")


# Example usage
if __name__ == "__main__":
    # Create constrained face tracking system
    tracker = ConstrainedFaceTracking(robot_model="ur10e")  # Change to "ur30" if needed
    
    try:
        # Connect to robot
        if not tracker.connect_robot():
            exit(1)
        
        # Test the IK solver first
        print("\n=== Testing IK Solver ===")
        current_pose = tracker.get_current_pose()
        current_joints = tracker.get_current_joints()
        
        print(f"Current pose: {[f'{x:.3f}' for x in current_pose]}")
        print(f"Current joints: {[f'{math.degrees(j):.1f}°' for j in current_joints]}")
        
        # Test a simple constrained movement
        test_position = [current_pose[0] + 0.05, current_pose[1], current_pose[2]]  # 5cm forward
        test_orientation = current_pose[3:]  # Same orientation
        
        print(f"\nTesting movement to: {[f'{x:.3f}' for x in test_position]}")
        
        success = tracker.move_to_pose_with_constrained_wrists(test_position, test_orientation)
        
        if success:
            print("✓ Basic constrained movement successful!")
            
            # Optionally run full demonstration
            user_input = input("\nRun full movement demonstration? (y/n): ")
            if user_input.lower() == 'y':
                tracker.demonstrate_constrained_movements()
        else:
            print("✗ Basic movement failed. Check robot state and IK solver.")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        tracker.disconnect_robot()
        print("Program ended.") 