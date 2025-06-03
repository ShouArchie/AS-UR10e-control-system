"""
UR10e IK Solver using official URDF file
This version loads the kinematic chain directly from the URDF file for maximum accuracy.
"""

import numpy as np
import ikpy.chain
import math
from typing import List, Tuple, Optional
import os

class URIKSolverURDF:
    """
    Inverse Kinematics solver for UR10e using official URDF file.
    This ensures exact match with official Universal Robots specifications.
    """
    
    def __init__(self, urdf_file_path: str = "ur10e.urdf"):
        """
        Initialize the IK solver using URDF file.
        
        Args:
            urdf_file_path: Path to the URDF file
        """
        self.urdf_file_path = urdf_file_path
        
        # Check if URDF file exists
        if not os.path.exists(urdf_file_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_file_path}")
        
        # Load kinematic chain from URDF
        try:
            # Load the chain from base_link, which will automatically create the full kinematic chain
            # IKpy will parse the URDF and create links for each joint
            self.chain = ikpy.chain.Chain.from_urdf_file(
                urdf_file_path,
                base_elements=["base_link"],  # Start from base_link
                name="ur10e_chain"
            )
            
            print(f"✓ Successfully loaded URDF: {urdf_file_path}")
            print(f"✓ Kinematic chain created with {len(self.chain.links)} links")
            
            # Print joint information
            joint_names = []
            for i, link in enumerate(self.chain.links):
                if hasattr(link, 'name'):
                    joint_names.append(link.name)
                else:
                    joint_names.append(f"Link_{i}")
            
            print(f"✓ Joints: {joint_names}")
            print(f"✓ Active links mask: {self.chain.active_links_mask}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF file: {e}")
        
        # Joint limits for UR10e (in radians) - these are for the 6 robot joints
        self.joint_limits = [
            (-2*math.pi, 2*math.pi),  # shoulder_pan_joint
            (-2*math.pi, 2*math.pi),  # shoulder_lift_joint
            (-math.pi, math.pi),      # elbow_joint  
            (-2*math.pi, 2*math.pi),  # wrist_1_joint
            (-2*math.pi, 2*math.pi),  # wrist_2_joint
            (-2*math.pi, 2*math.pi)   # wrist_3_joint
        ]
        
        print("✓ UR10e IK Solver (URDF) initialized")
    
    def forward_kinematics(self, joints: List[float]) -> Tuple[List[float], List[float]]:
        """
        Compute forward kinematics using URDF-based kinematic chain.
        
        Args:
            joints: List of 6 joint angles in radians
            
        Returns:
            Tuple of (position [x,y,z], orientation [rx,ry,rz])
        """
        if len(joints) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joints)}")
        
        # IKpy expects values for all links in the chain
        # The chain likely has: [origin, base_link, shoulder_joint, shoulder_lift_joint, ...]
        # We need to provide values for all of them, including the fixed/origin links
        
        # Create the joint array for IKpy (with fixed joint values as 0)
        ikpy_joints = [0.0] * len(self.chain.links)
        
        # Find which indices correspond to the 6 robot joints and fill them
        joint_index = 0
        for i, is_active in enumerate(self.chain.active_links_mask):
            if is_active and joint_index < 6:
                ikpy_joints[i] = joints[joint_index]
                joint_index += 1
        
        # Compute forward kinematics
        transformation_matrix = self.chain.forward_kinematics(ikpy_joints)
        
        # Extract position
        position = transformation_matrix[:3, 3].tolist()
        
        # Extract rotation matrix and convert to rotation vector (axis-angle)
        R = transformation_matrix[:3, :3]
        
        # Convert rotation matrix to rotation vector using Rodrigues' formula
        try:
            trace_R = np.trace(R)
            if trace_R > 3.0:
                trace_R = 3.0
            elif trace_R < -1.0:
                trace_R = -1.0
                
            angle = math.acos((trace_R - 1) / 2)
            
            if abs(angle) < 1e-6:
                # No rotation
                orientation = [0.0, 0.0, 0.0]
            elif abs(angle - math.pi) < 1e-6:
                # 180-degree rotation - special case
                # Find the eigenvector corresponding to eigenvalue 1
                axis = np.array([
                    math.sqrt(max(0, (R[0,0] + 1) / 2)),
                    math.sqrt(max(0, (R[1,1] + 1) / 2)),
                    math.sqrt(max(0, (R[2,2] + 1) / 2))
                ])
                if R[0,1] < 0: axis[1] = -axis[1]
                if R[0,2] < 0: axis[2] = -axis[2]
                orientation = (angle * axis).tolist()
            else:
                # General case
                axis = (1 / (2 * math.sin(angle))) * np.array([
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0], 
                    R[1, 0] - R[0, 1]
                ])
                orientation = (angle * axis).tolist()
                
        except Exception as e:
            print(f"Warning: Could not convert rotation matrix to axis-angle: {e}")
            orientation = [0.0, 0.0, 0.0]
        
        return position, orientation
    
    def solve_ik(self, 
                 target_position: List[float], 
                 target_orientation: Optional[List[float]] = None,
                 current_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """
        Solve inverse kinematics using URDF-based kinematic chain.
        
        Args:
            target_position: [x, y, z] target position in meters
            target_orientation: [rx, ry, rz] target orientation in radians (optional)
            current_joints: Current joint angles as initial guess (optional)
            
        Returns:
            List of 6 joint angles in radians, or None if no solution found
        """
        try:
            # Create target transformation matrix
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = target_position
            
            # Handle orientation if provided
            if target_orientation is not None:
                # Convert axis-angle to rotation matrix
                angle = np.linalg.norm(target_orientation)
                if angle > 1e-6:
                    axis = np.array(target_orientation) / angle
                    # Rodrigues' rotation formula
                    K = np.array([[0, -axis[2], axis[1]],
                                 [axis[2], 0, -axis[0]], 
                                 [-axis[1], axis[0], 0]])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                    target_matrix[:3, :3] = R
            
            # Set up initial guess
            if current_joints is None:
                initial_joints = [0.0] * len(self.chain.links)
            else:
                # Map the 6 robot joints to the full chain
                initial_joints = [0.0] * len(self.chain.links)
                joint_index = 0
                for i, is_active in enumerate(self.chain.active_links_mask):
                    if is_active and joint_index < 6:
                        initial_joints[i] = current_joints[joint_index]
                        joint_index += 1
            
            # Solve IK
            solution = self.chain.inverse_kinematics_frame(
                target_matrix,
                initial_position=initial_joints
            )
            
            # Extract the 6 robot joints from the solution
            robot_joints = []
            joint_index = 0
            for i, is_active in enumerate(self.chain.active_links_mask):
                if is_active and joint_index < 6:
                    robot_joints.append(solution[i])
                    joint_index += 1
            
            # Check joint limits
            if len(robot_joints) == 6 and self._check_joint_limits(robot_joints):
                return robot_joints
            else:
                print("Solution violates joint limits or incorrect number of joints")
                return None
                
        except Exception as e:
            print(f"IK solution failed: {e}")
            return None
    
    def solve_constrained_wrist_ik(self,
                                   target_position: List[float],
                                   target_orientation: Optional[List[float]],
                                   wrist1_angle: float,
                                   wrist3_angle: float,
                                   current_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """
        Solve IK with wrist1 and wrist3 joints constrained to specific values.
        
        Args:
            target_position: [x, y, z] target position in meters
            target_orientation: [rx, ry, rz] target orientation in radians (optional)
            wrist1_angle: Fixed angle for wrist1 joint (joint 3) in radians
            wrist3_angle: Fixed angle for wrist3 joint (joint 5) in radians
            current_joints: Current joint angles as initial guess (optional)
            
        Returns:
            List of joint angles in radians, or None if no solution found
        """
        # This is a simplified implementation - for more advanced constraints,
        # you might want to use optimization-based IK solvers
        
        solution = self.solve_ik(target_position, target_orientation, current_joints)
        
        if solution is not None:
            # Force the constrained joints to the desired values
            solution[3] = wrist1_angle  # wrist_1_joint
            solution[5] = wrist3_angle  # wrist_3_joint
            
            # Verify the solution still works
            test_pos, test_orient = self.forward_kinematics(solution)
            pos_error = np.linalg.norm(np.array(target_position) - np.array(test_pos))
            
            if pos_error < 0.01:  # 1cm tolerance
                return solution
            else:
                print(f"Constrained solution has high position error: {pos_error:.4f}m")
                return None
        
        return None
    
    def _check_joint_limits(self, joints: List[float]) -> bool:
        """Check if joint angles are within limits."""
        for i, (joint_angle, (min_limit, max_limit)) in enumerate(zip(joints, self.joint_limits)):
            if joint_angle < min_limit or joint_angle > max_limit:
                print(f"Joint {i} ({math.degrees(joint_angle):.1f}°) exceeds limits "
                      f"[{math.degrees(min_limit):.1f}°, {math.degrees(max_limit):.1f}°]")
                return False
        return True
    
    def get_chain_info(self):
        """Get information about the kinematic chain."""
        print(f"\n=== Kinematic Chain Information ===")
        print(f"URDF file: {self.urdf_file_path}")
        print(f"Number of links: {len(self.chain.links)}")
        print(f"Active joints: {sum(self.chain.active_links_mask)}")
        
        print("\nLinks in chain:")
        for i, link in enumerate(self.chain.links):
            active = "ACTIVE" if self.chain.active_links_mask[i] else "FIXED"
            name = getattr(link, 'name', f'Link_{i}')
            print(f"  {i}: {name} ({active})")
        
        print(f"\nJoint limits (degrees):")
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        for i, (name, (min_deg, max_deg)) in enumerate(zip(joint_names, 
                                                          [(math.degrees(lim[0]), math.degrees(lim[1])) 
                                                           for lim in self.joint_limits])):
            print(f"  {name}: [{min_deg:.1f}°, {max_deg:.1f}°]")


def test_urdf_solver():
    """Test the URDF-based IK solver."""
    print("=== Testing URDF-based UR10e IK Solver ===")
    
    try:
        # Initialize solver
        solver = URIKSolverURDF("ur10e.urdf")
        
        # Get chain information
        solver.get_chain_info()
        
        # Test forward kinematics with known poses
        print("\n=== Forward Kinematics Tests ===")
        test_configs = {
            "Home Position": [0, -math.pi/2, 0, -math.pi/2, 0, 0],
            "Extended Forward": [0, 0, 0, 0, 0, 0],
            "Side Position": [math.pi/2, -math.pi/4, 0, -math.pi/2, 0, 0]
        }
        
        for name, joints in test_configs.items():
            pos, orient = solver.forward_kinematics(joints)
            reach = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            
            print(f"\n{name}:")
            print(f"  Joints: {[f'{math.degrees(j):.1f}°' for j in joints]}")
            print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]m")
            print(f"  Reach: {reach:.3f}m")
            print(f"  Orientation: [{math.degrees(orient[0]):.2f}°, {math.degrees(orient[1]):.2f}°, {math.degrees(orient[2]):.2f}°]")
        
        # Test inverse kinematics
        print("\n=== Inverse Kinematics Test ===")
        target_pos = [0.5, 0.2, 0.8]
        target_orient = [0, 0, 0]
        
        print(f"Target position: {target_pos}")
        print(f"Target orientation: {target_orient}")
        
        solution = solver.solve_ik(target_pos, target_orient)
        
        if solution:
            print(f"IK Solution: {[f'{math.degrees(j):.1f}°' for j in solution]}")
            
            # Verify with forward kinematics
            fk_pos, fk_orient = solver.forward_kinematics(solution)
            pos_error = np.linalg.norm(np.array(target_pos) - np.array(fk_pos))
            orient_error = np.linalg.norm(np.array(target_orient) - np.array(fk_orient))
            
            print(f"Forward kinematics verification:")
            print(f"  Position: {fk_pos} (error: {pos_error:.6f}m)")
            print(f"  Orientation: {fk_orient} (error: {math.degrees(orient_error):.3f}°)")
        else:
            print("❌ No IK solution found")
        
        print("\n✅ URDF-based solver test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_urdf_solver() 