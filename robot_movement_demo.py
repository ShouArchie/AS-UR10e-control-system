"""
Robot Movement Visualization Demo for UR10e
This script demonstrates various robot movements using forward kinematics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from ur_ik_solver import URIKSolver
import time

class RobotMovementDemo:
    """
    Demo class for visualizing UR10e robot movements.
    """
    
    def __init__(self):
        self.solver = URIKSolver("ur10e")
        print("✓ Robot Movement Demo initialized")
    
    def plot_robot_3d(self, joints, title="Robot Configuration", ax=None):
        """Plot the robot in 3D for given joint angles."""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Get link positions by computing FK for each link
        link_positions = self._compute_link_positions(joints)
        
        # Extract x, y, z coordinates
        x_coords = [pos[0] for pos in link_positions]
        y_coords = [pos[1] for pos in link_positions]
        z_coords = [pos[2] for pos in link_positions]
        
        # Plot the robot structure
        ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=4, label='Robot Links')
        ax.scatter(x_coords, y_coords, z_coords, c='red', s=80, label='Joints', zorder=5)
        
        # Plot end-effector position
        end_pos = link_positions[-1]
        ax.scatter(end_pos[0], end_pos[1], end_pos[2], c='green', s=150, label='End Effector', zorder=5)
        
        # Plot coordinate frame at base
        ax.quiver(0, 0, 0, 0.3, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=3, label='X (Red)')
        ax.quiver(0, 0, 0, 0, 0.3, 0, color='green', arrow_length_ratio=0.1, linewidth=3, label='Y (Green)')  
        ax.quiver(0, 0, 0, 0, 0, 0.3, color='blue', arrow_length_ratio=0.1, linewidth=3, label='Z (Blue)')
        
        # Add tool frame at end effector
        pos, orient = self.solver.forward_kinematics(joints)
        # Simplified tool frame representation
        ax.quiver(pos[0], pos[1], pos[2], 0.1, 0, 0, color='orange', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(pos[0], pos[1], pos[2], 0, 0.1, 0, color='yellow', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(pos[0], pos[1], pos[2], 0, 0, 0.1, color='purple', arrow_length_ratio=0.1, linewidth=2)
        
        # Set equal aspect ratio and labels
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        
        # Set workspace limits
        limit = 1.5  # 1.5m workspace
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([0, limit])
        
        # Calculate and display reach
        reach = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        ax.text2D(0.02, 0.98, f"Reach: {reach:.3f}m", transform=ax.transAxes, 
                 fontsize=12, verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        return ax, end_pos
    
    def _compute_link_positions(self, joints):
        """Compute positions of all links for visualization."""
        positions = [(0, 0, 0)]  # Base position
        
        # Add each joint position by computing forward kinematics for partial chains
        for i in range(1, len(joints) + 1):
            partial_joints = [0.0] + joints[:i] + [0.0] * (6 - i)
            transform = self.solver.chain.forward_kinematics(partial_joints)
            pos = transform[:3, 3]
            positions.append((pos[0], pos[1], pos[2]))
        
        return positions
    
    def demo_basic_configurations(self):
        """Demonstrate basic robot configurations."""
        print("\n=== Basic Robot Configurations ===")
        
        configs = {
            "Home Position": [0, -math.pi/2, 0, -math.pi/2, 0, 0],
            "Extended Forward": [0, 0, 0, 0, 0, 0],
            "Extended Up": [0, -math.pi/4, -math.pi/4, -math.pi/2, 0, 0],
            "Compact Position": [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0],
            "Side Reach": [math.pi/2, -math.pi/4, 0, -math.pi/2, 0, 0]
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for i, (name, joints) in enumerate(configs.items()):
            if i < len(axes):
                ax = axes[i]
                self.plot_robot_3d(joints, name, ax)
                
                # Print joint angles
                print(f"\n{name}:")
                print(f"  Joint angles: {[f'{math.degrees(j):.1f}°' for j in joints]}")
                
                # Get end effector position
                pos, orient = self.solver.forward_kinematics(joints)
                reach = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                print(f"  End effector: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m")
                print(f"  Reach: {reach:.3f}m")
        
        # Hide the last subplot if not used
        if len(configs) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def demo_joint_sweep(self, joint_index=0, steps=8):
        """Demonstrate sweeping a single joint through its range."""
        print(f"\n=== Joint {joint_index} Sweep Demo ===")
        
        # Define joint limits
        if joint_index == 2:  # Elbow joint
            limits = (-math.pi, math.pi)
        else:
            limits = (-math.pi, math.pi)  # Simplified for demo
        
        # Create joint angle sequence
        angles = np.linspace(limits[0], limits[1], steps)
        
        # Base configuration (home position)
        base_joints = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        print(f"Sweeping joint {joint_index} from {math.degrees(limits[0]):.1f}° to {math.degrees(limits[1]):.1f}°")
        
        for i, angle in enumerate(angles):
            if i < len(axes):
                # Set the joint angle
                joints = base_joints.copy()
                joints[joint_index] = angle
                
                ax = axes[i]
                self.plot_robot_3d(joints, f"Joint {joint_index}: {math.degrees(angle):.1f}°", ax)
                
                print(f"  Step {i+1}: {math.degrees(angle):.1f}°")
        
        plt.tight_layout()
        plt.show()
    
    def demo_workspace_boundary(self, num_points=20):
        """Demonstrate the robot's workspace boundary."""
        print(f"\n=== Workspace Boundary Demo ===")
        
        # Test different orientations to find maximum reach
        test_configs = []
        max_reaches = []
        
        # Generate test configurations
        for base_angle in np.linspace(0, 2*math.pi, 8):
            for shoulder_angle in np.linspace(-math.pi/2, math.pi/2, 5):
                for elbow_angle in np.linspace(-math.pi, math.pi, 5):
                    joints = [base_angle, shoulder_angle, elbow_angle, -shoulder_angle-elbow_angle, 0, 0]
                    
                    pos, orient = self.solver.forward_kinematics(joints)
                    reach = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                    
                    test_configs.append((joints, pos, reach))
                    max_reaches.append(reach)
        
        # Find configurations with maximum reach
        max_reach = max(max_reaches)
        max_configs = [config for config in test_configs if abs(config[2] - max_reach) < 0.01]
        
        print(f"Maximum reach found: {max_reach:.3f}m")
        print(f"Number of max-reach configurations: {len(max_configs)}")
        
        # Plot a few maximum reach configurations
        num_plots = min(6, len(max_configs))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for i in range(num_plots):
            joints, pos, reach = max_configs[i]
            ax = axes[i]
            self.plot_robot_3d(joints, f"Max Reach Config {i+1}\nReach: {reach:.3f}m", ax)
        
        # Hide unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return max_reach
    
    def demo_trajectory(self, start_joints, end_joints, steps=10):
        """Demonstrate a trajectory between two joint configurations."""
        print(f"\n=== Trajectory Demo ===")
        
        # Create trajectory
        trajectory = []
        for i in range(steps):
            t = i / (steps - 1)  # Parameter from 0 to 1
            joints = []
            for j in range(6):
                joint_angle = start_joints[j] + t * (end_joints[j] - start_joints[j])
                joints.append(joint_angle)
            trajectory.append(joints)
        
        # Plot trajectory
        fig = plt.figure(figsize=(15, 10))
        
        # Plot multiple frames
        for i, joints in enumerate(trajectory):
            ax = fig.add_subplot(2, 5, i+1, projection='3d')
            self.plot_robot_3d(joints, f"Step {i+1}", ax)
        
        plt.tight_layout()
        plt.show()
        
        # Plot end effector path
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get end effector positions
        positions = []
        for joints in trajectory:
            pos, orient = self.solver.forward_kinematics(joints)
            positions.append(pos)
        
        positions = np.array(positions)
        
        # Plot path
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=3, label='End Effector Path')
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, label='End')
        
        # Plot robot at start and end
        self.plot_robot_3d(start_joints, "Trajectory", ax)
        
        ax.legend()
        plt.title("End Effector Trajectory")
        plt.show()


def main():
    """Main demo function."""
    print("=== UR10e Robot Movement Visualization Demo ===")
    print("This demo shows various robot movements using official URDF parameters")
    
    demo = RobotMovementDemo()
    
    while True:
        print("\n=== Demo Options ===")
        print("1. Basic Configurations")
        print("2. Joint Sweep (Base Joint)")
        print("3. Joint Sweep (Shoulder Joint)")  
        print("4. Joint Sweep (Elbow Joint)")
        print("5. Workspace Boundary")
        print("6. Trajectory Demo")
        print("0. Exit")
        
        choice = input("\nSelect demo (0-6): ").strip()
        
        if choice == '0':
            print("Demo completed!")
            break
        elif choice == '1':
            demo.demo_basic_configurations()
        elif choice == '2':
            demo.demo_joint_sweep(joint_index=0)  # Base
        elif choice == '3':
            demo.demo_joint_sweep(joint_index=1)  # Shoulder
        elif choice == '4':
            demo.demo_joint_sweep(joint_index=2)  # Elbow
        elif choice == '5':
            max_reach = demo.demo_workspace_boundary()
            print(f"\nMaximum reach achieved: {max_reach:.3f}m")
        elif choice == '6':
            # Demo trajectory from home to extended
            start = [0, -math.pi/2, 0, -math.pi/2, 0, 0]  # Home
            end = [0, 0, 0, 0, 0, 0]  # Extended
            demo.demo_trajectory(start, end)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 