"""
Visualization and testing script for UR IK solver.
This script allows you to verify the kinematic model against known UR10e specifications.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from ur_ik_solver import URIKSolver

class URVisualization:
    """Visualization and testing class for UR robot kinematics."""
    
    def __init__(self, robot_model="ur10e"):
        self.robot_model = robot_model
        self.ik_solver = URIKSolver(robot_model)
        
        # UR10e specifications for verification
        self.ur10e_specs = {
            'reach': 1.300,  # 1300mm max reach
            'base_height': 0.1807,  # Base height
            'shoulder_offset': 0.1198,  # Shoulder offset
            'upper_arm_length': 0.6127,  # Upper arm length
            'forearm_length': 0.5716,  # Forearm length  
            'wrist_1_length': 0.1639,  # Wrist 1 length
            'wrist_2_length': 0.1157,  # Wrist 2 length
            'wrist_3_length': 0.0922,  # Wrist 3 length
        }
        
        print(f"✓ Visualization initialized for {robot_model.upper()}")
        print("UR10e Specifications:")
        for key, value in self.ur10e_specs.items():
            print(f"  {key}: {value*1000:.0f}mm" if value < 10 else f"  {key}: {value:.4f}m")
    
    def plot_robot_configuration(self, joints, title="Robot Configuration"):
        """Plot the robot in 3D for given joint angles."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get link positions by computing FK for each link
        link_positions = self._compute_link_positions(joints)
        
        # Extract x, y, z coordinates
        x_coords = [pos[0] for pos in link_positions]
        y_coords = [pos[1] for pos in link_positions]
        z_coords = [pos[2] for pos in link_positions]
        
        # Plot the robot structure
        ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=3, label='Robot Links')
        ax.scatter(x_coords, y_coords, z_coords, c='red', s=50, label='Joints')
        
        # Plot end-effector position
        end_pos = link_positions[-1]
        ax.scatter(end_pos[0], end_pos[1], end_pos[2], c='green', s=100, label='End Effector')
        
        # Plot coordinate frame at base
        ax.quiver(0, 0, 0, 0.2, 0, 0, color='red', arrow_length_ratio=0.1, label='X')
        ax.quiver(0, 0, 0, 0, 0.2, 0, color='green', arrow_length_ratio=0.1, label='Y')  
        ax.quiver(0, 0, 0, 0, 0, 0.2, color='blue', arrow_length_ratio=0.1, label='Z')
        
        # Set equal aspect ratio and labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        # Set workspace limits
        limit = 1.5  # 1.5m workspace
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([0, limit])
        
        plt.show()
        
        return end_pos
    
    def _compute_link_positions(self, joints):
        """Compute positions of all links for visualization."""
        positions = [(0, 0, 0)]  # Base position
        
        # Add each joint position by computing forward kinematics for partial chains
        for i in range(1, len(joints) + 1):
            partial_joints = [0.0] + joints[:i] + [0.0] * (6 - i)
            transform = self.ik_solver.chain.forward_kinematics(partial_joints)
            pos = transform[:3, 3]
            positions.append((pos[0], pos[1], pos[2]))
        
        return positions
    
    def test_known_configurations(self):
        """Test known robot configurations to verify the model."""
        print("\n=== Testing Known Configurations ===")
        
        test_configs = {
            "Home Position": [0, -math.pi/2, 0, -math.pi/2, 0, 0],
            "Extended Forward": [0, 0, 0, 0, 0, 0],
            "Fully Extended": [0, 0, -math.pi/2, 0, 0, 0],
            "Compact Position": [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
        }
        
        results = {}
        
        for config_name, joints in test_configs.items():
            print(f"\n--- {config_name} ---")
            print(f"Joint angles: {[f'{math.degrees(j):.1f}°' for j in joints]}")
            
            # Compute forward kinematics
            position, orientation = self.ik_solver.forward_kinematics(joints)
            
            print(f"End effector position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]m")
            print(f"End effector orientation: [{math.degrees(orientation[0]):.1f}°, {math.degrees(orientation[1]):.1f}°, {math.degrees(orientation[2]):.1f}°]")
            
            # Calculate reach from base
            reach = math.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
            print(f"Reach from base: {reach:.3f}m")
            
            # Check against UR10e max reach
            if reach > self.ur10e_specs['reach']:
                print(f"⚠️  Warning: Reach ({reach:.3f}m) exceeds UR10e spec ({self.ur10e_specs['reach']:.3f}m)")
            else:
                print(f"✓ Reach within UR10e specifications")
            
            results[config_name] = {
                'joints': joints,
                'position': position,
                'orientation': orientation,
                'reach': reach
            }
        
        return results
    
    def test_ik_accuracy(self):
        """Test IK solver accuracy by solving FK->IK->FK roundtrip."""
        print("\n=== Testing IK Accuracy ===")
        
        # Test several joint configurations
        test_joints = [
            [0, -math.pi/4, math.pi/6, -math.pi/3, math.pi/2, 0],
            [math.pi/6, -math.pi/3, math.pi/4, -math.pi/2, -math.pi/4, math.pi/6],
            [math.pi/4, -math.pi/6, math.pi/3, -math.pi/4, math.pi/3, -math.pi/6]
        ]
        
        for i, joints in enumerate(test_joints):
            print(f"\n--- Test {i+1} ---")
            print(f"Original joints: {[f'{math.degrees(j):.1f}°' for j in joints]}")
            
            # Forward kinematics
            target_pos, target_orient = self.ik_solver.forward_kinematics(joints)
            print(f"Target position: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
            
            # Inverse kinematics
            ik_solution = self.ik_solver.solve_ik(target_pos, target_orient, joints)
            
            if ik_solution:
                print(f"IK solution: {[f'{math.degrees(j):.1f}°' for j in ik_solution]}")
                
                # Forward kinematics of IK solution
                verify_pos, verify_orient = self.ik_solver.forward_kinematics(ik_solution)
                
                # Calculate errors
                pos_error = np.linalg.norm(np.array(target_pos) - np.array(verify_pos))
                orient_error = np.linalg.norm(np.array(target_orient) - np.array(verify_orient))
                
                print(f"Position error: {pos_error*1000:.3f}mm")
                print(f"Orientation error: {math.degrees(orient_error):.3f}°")
                
                if pos_error < 0.001 and orient_error < 0.01:  # 1mm, 0.57° tolerance
                    print("✓ IK solution accurate")
                else:
                    print("⚠️ IK solution may have accuracy issues")
            else:
                print("✗ IK solution failed")
    
    def test_constrained_ik(self):
        """Test constrained IK with wrist constraints."""
        print("\n=== Testing Constrained IK ===")
        
        # Target pose
        target_pos = [0.6, 0.2, 0.4]  # Reachable position
        target_orient = [0, 0, 0]     # No rotation
        
        # Wrist constraints
        wrist1_fixed = math.radians(-90)
        wrist3_fixed = math.radians(45)
        
        print(f"Target position: {target_pos}")
        print(f"Wrist1 fixed at: {math.degrees(wrist1_fixed):.1f}°")
        print(f"Wrist3 fixed at: {math.degrees(wrist3_fixed):.1f}°")
        
        # Solve constrained IK
        solution = self.ik_solver.solve_constrained_wrist_ik(
            target_pos, target_orient, wrist1_fixed, wrist3_fixed
        )
        
        if solution:
            print(f"Solution found: {[f'{math.degrees(j):.1f}°' for j in solution]}")
            print(f"Wrist1 result: {math.degrees(solution[3]):.1f}° (target: {math.degrees(wrist1_fixed):.1f}°)")
            print(f"Wrist3 result: {math.degrees(solution[5]):.1f}° (target: {math.degrees(wrist3_fixed):.1f}°)")
            
            # Verify with FK
            verify_pos, verify_orient = self.ik_solver.forward_kinematics(solution)
            pos_error = np.linalg.norm(np.array(target_pos) - np.array(verify_pos))
            print(f"Position accuracy: {pos_error*1000:.3f}mm")
            
            return solution
        else:
            print("✗ No constrained IK solution found")
            return None
    
    def verify_ur10e_workspace(self):
        """Verify that our model matches UR10e workspace specifications."""
        print("\n=== Verifying UR10e Workspace ===")
        
        # Test maximum reach in different directions
        test_directions = [
            [1, 0, 0],   # +X direction
            [0, 1, 0],   # +Y direction  
            [0, 0, 1],   # +Z direction
            [-1, 0, 0],  # -X direction
            [0, -1, 0],  # -Y direction
        ]
        
        max_reaches = []
        
        for direction in test_directions:
            print(f"\nTesting reach in direction {direction}:")
            
            # Try different distances along this direction
            for distance in np.arange(0.5, 1.6, 0.1):  # 0.5m to 1.5m
                target_pos = [d * distance for d in direction]
                target_orient = [0, 0, 0]
                
                solution = self.ik_solver.solve_ik(target_pos, target_orient)
                
                if solution is None:
                    max_reach = distance - 0.1  # Previous successful distance
                    max_reaches.append(max_reach)
                    print(f"  Max reach: {max_reach:.3f}m")
                    break
            else:
                max_reaches.append(1.5)  # If we reached the end of our test range
                print(f"  Max reach: >1.5m")
        
        overall_max = max(max_reaches)
        print(f"\nOverall maximum reach: {overall_max:.3f}m")
        print(f"UR10e specification: {self.ur10e_specs['reach']:.3f}m")
        
        if abs(overall_max - self.ur10e_specs['reach']) < 0.1:
            print("✓ Workspace matches UR10e specifications")
        else:
            print("⚠️ Workspace may not match UR10e specifications")


def main():
    """Main testing function."""
    print("=== UR10e IK Solver Verification ===")
    
    # Create visualization instance
    vis = URVisualization("ur10e")
    
    # Test 1: Known configurations
    known_results = vis.test_known_configurations()
    
    # Test 2: IK accuracy
    vis.test_ik_accuracy()
    
    # Test 3: Constrained IK
    constrained_solution = vis.test_constrained_ik()
    
    # Test 4: Workspace verification
    vis.verify_ur10e_workspace()
    
    # Visualization options
    print("\n=== Visualization Options ===")
    print("1. Plot home position")
    print("2. Plot extended position") 
    print("3. Plot constrained IK solution")
    print("4. Plot all test configurations")
    
    choice = input("\nSelect visualization (1-4) or 'q' to quit: ")
    
    if choice == '1':
        vis.plot_robot_configuration([0, -math.pi/2, 0, -math.pi/2, 0, 0], "Home Position")
    elif choice == '2':
        vis.plot_robot_configuration([0, 0, 0, 0, 0, 0], "Extended Position")
    elif choice == '3' and constrained_solution:
        vis.plot_robot_configuration(constrained_solution, "Constrained IK Solution")
    elif choice == '4':
        for name, result in known_results.items():
            vis.plot_robot_configuration(result['joints'], name)
    elif choice != 'q':
        print("Invalid choice")

if __name__ == "__main__":
    main() 