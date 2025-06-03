"""
UR10e Inverse Kinematics Solver
Enhanced version with optional URDF loading capability
Uses official UR10e URDF parameters when available, falls back to manual definition.
"""

import numpy as np
import ikpy.chain
import ikpy.link
import math
from typing import List, Tuple, Optional
import os

class URIKSolver:
    """
    Inverse Kinematics solver for UR10e.
    Can load from URDF file or use manual kinematic chain definition.
    """
    
    def __init__(self, urdf_file_path: Optional[str] = "ur10e.urdf", use_urdf: bool = True):
        """
        Initialize the IK solver.
        
        Args:
            urdf_file_path: Path to the URDF file (optional)
            use_urdf: Whether to attempt loading from URDF first
        """
        self.urdf_file_path = urdf_file_path
        self.using_urdf = False
        
        # Try loading from URDF first if requested and file exists
        if use_urdf and urdf_file_path and os.path.exists(urdf_file_path):
            try:
                self._load_from_urdf()
                self.using_urdf = True
                print("✓ Successfully loaded kinematic chain from URDF")
            except Exception as e:
                print(f"⚠ URDF loading failed: {e}")
                print("➤ Falling back to manual kinematic chain definition...")
                self._create_manual_chain()
        else:
            print("➤ Using manual kinematic chain definition...")
            self._create_manual_chain()
        
        # Joint limits for UR10e (in radians)
        self.joint_limits = [
            (-2*math.pi, 2*math.pi),  # shoulder_pan_joint
            (-2*math.pi, 2*math.pi),  # shoulder_lift_joint  
            (-math.pi, math.pi),      # elbow_joint
            (-2*math.pi, 2*math.pi),  # wrist_1_joint
            (-2*math.pi, 2*math.pi),  # wrist_2_joint
            (-2*math.pi, 2*math.pi)   # wrist_3_joint
        ]
        
        print(f"✓ UR10e IK Solver initialized ({'URDF' if self.using_urdf else 'Manual'})")
    
    def _load_from_urdf(self):
        """Load kinematic chain from URDF file."""
        # Try to load using IKpy's URDF parser
        self.chain = ikpy.chain.Chain.from_urdf_file(
            self.urdf_file_path,
            base_elements=["base_link"]
        )
        
        # Validate that we got a reasonable chain
        if len(self.chain.links) < 7:  # Should have at least 7 links for 6 DOF robot
            raise ValueError(f"URDF chain has only {len(self.chain.links)} links, expected at least 7")
        
        # Count active joints
        active_joints = sum(self.chain.active_links_mask)
        if active_joints != 6:
            raise ValueError(f"URDF chain has {active_joints} active joints, expected 6")
    
    def _create_manual_chain(self):
        """Create kinematic chain manually using official UR10e URDF parameters."""
        # Official UR10e URDF parameters (from ur10e.urdf file you provided)
        
        # Define the kinematic chain using official URDF transforms
        links = [
            # Origin link (fixed)
            ikpy.link.OriginLink(),
            
            # Shoulder pan joint (joint 0)
            ikpy.link.URDFLink(
                name="shoulder_pan_joint",
                origin_translation=[0, 0, 0.1807],  # From URDF: xyz="0 0 0.1807"
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1]  # Z-axis rotation
            ),
            
            # Shoulder lift joint (joint 1) 
            ikpy.link.URDFLink(
                name="shoulder_lift_joint", 
                origin_translation=[0, 0, 0],       # From URDF: xyz="0 0 0"
                origin_orientation=[1.570796327, 0, 0],  # From URDF: rpy="1.570796327 0 0"
                rotation=[0, 1, 0]  # Y-axis rotation
            ),
            
            # Elbow joint (joint 2)
            ikpy.link.URDFLink(
                name="elbow_joint",
                origin_translation=[-0.6127, 0, 0],  # From URDF: xyz="-0.6127 0 0"
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0]  # Y-axis rotation
            ),
            
            # Wrist 1 joint (joint 3)
            ikpy.link.URDFLink(
                name="wrist_1_joint",
                origin_translation=[-0.57155, 0, 0.17415],  # From URDF: xyz="-0.57155 0 0.17415"
                origin_orientation=[0, 1.570796327, 0],      # From URDF: rpy="0 1.570796327 0"
                rotation=[0, 1, 0]  # Y-axis rotation
            ),
            
            # Wrist 2 joint (joint 4)
            ikpy.link.URDFLink(
                name="wrist_2_joint",
                origin_translation=[0, -0.11985, 0],  # From URDF: xyz="0 -0.11985 -2.458e-11"
                origin_orientation=[0, 0, 1.570796327],  # From URDF: rpy="0 0 1.570796327"
                rotation=[0, 0, 1]  # Z-axis rotation  
            ),
            
            # Wrist 3 joint (joint 5)
            ikpy.link.URDFLink(
                name="wrist_3_joint",
                origin_translation=[0, 0.11655, 0],  # From URDF: xyz="0 0.11655 -2.458e-11"
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0]  # Y-axis rotation
            ),
            
            # Tool frame (fixed end effector) - use OriginLink for fixed joint
            ikpy.link.OriginLink()
        ]
        
        # Create the kinematic chain
        # Active links mask: [False, True, True, True, True, True, True, False]
        #                   [origin, j0,   j1,   j2,   j3,   j4,   j5,   tool0]
        active_links_mask = [False, True, True, True, True, True, True, False]
        
        self.chain = ikpy.chain.Chain(
            links=links,
            active_links_mask=active_links_mask,
            name="ur10e_manual_chain"
        )
        
        print(f"✓ Manual kinematic chain created with {len(self.chain.links)} links")
        print(f"✓ Active joints: {sum(self.chain.active_links_mask)}")
    
    def forward_kinematics(self, joints: List[float]) -> Tuple[List[float], List[float]]:
        """
        Compute forward kinematics.
        
        Args:
            joints: List of 6 joint angles in radians
            
        Returns:
            Tuple of (position [x,y,z], orientation [rx,ry,rz])
        """
        if len(joints) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joints)}")
        
        # Create full joint array for IKpy (including fixed joints)
        if self.using_urdf:
            # For URDF-loaded chain, map joints to active positions
            ikpy_joints = [0.0] * len(self.chain.links)
            joint_index = 0
            for i, is_active in enumerate(self.chain.active_links_mask):
                if is_active and joint_index < 6:
                    ikpy_joints[i] = joints[joint_index]
                    joint_index += 1
        else:
            # For manual chain: [origin, j0, j1, j2, j3, j4, j5, tool0]
            ikpy_joints = [0.0] + list(joints) + [0.0]
        
        # Compute forward kinematics
        transformation_matrix = self.chain.forward_kinematics(ikpy_joints)
        
        # Extract position
        position = transformation_matrix[:3, 3].tolist()
        
        # Extract rotation matrix and convert to rotation vector (axis-angle)
        R = transformation_matrix[:3, :3]
        
        # Convert rotation matrix to rotation vector using Rodrigues' formula
        try:
            trace_R = np.trace(R)
            trace_R = max(-1.0, min(3.0, trace_R))  # Clamp to valid range
            
            angle = math.acos((trace_R - 1) / 2)
            
            if abs(angle) < 1e-6:
                # No rotation
                orientation = [0.0, 0.0, 0.0]
            elif abs(angle - math.pi) < 1e-6:
                # 180-degree rotation - special case
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
        Solve inverse kinematics.
        
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
                if self.using_urdf:
                    # Map to URDF chain structure
                    initial_joints = [0.0] * len(self.chain.links)
                    joint_index = 0
                    for i, is_active in enumerate(self.chain.active_links_mask):
                        if is_active and joint_index < 6:
                            initial_joints[i] = current_joints[joint_index]
                            joint_index += 1
                else:
                    # Map to manual chain structure
                    initial_joints = [0.0] + list(current_joints) + [0.0]
            
            # Solve IK
            solution = self.chain.inverse_kinematics_frame(
                target_matrix,
                initial_position=initial_joints
            )
            
            # Extract robot joints
            if self.using_urdf:
                # Extract from URDF chain
                robot_joints = []
                joint_index = 0
                for i, is_active in enumerate(self.chain.active_links_mask):
                    if is_active and joint_index < 6:
                        robot_joints.append(solution[i])
                        joint_index += 1
            else:
                # Extract from manual chain
                robot_joints = solution[1:7]
            
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
        print(f"Source: {'URDF file: ' + self.urdf_file_path if self.using_urdf else 'Manual definition'}")
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


def test_ik_solver():
    """Test the enhanced IK solver."""
    print("=== Testing Enhanced UR10e IK Solver ===")
    
    try:
        # Test with URDF first, then fallback to manual
        solver = URIKSolver("ur10e.urdf", use_urdf=True)
        
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
        
        print(f"Target position: {target_pos}")
        
        solution = solver.solve_ik(target_pos)
        
        if solution:
            print(f"IK Solution: {[f'{math.degrees(j):.1f}°' for j in solution]}")
            
            # Verify with forward kinematics
            fk_pos, fk_orient = solver.forward_kinematics(solution)
            pos_error = np.linalg.norm(np.array(target_pos) - np.array(fk_pos))
            
            print(f"Forward kinematics verification:")
            print(f"  Position: {fk_pos} (error: {pos_error:.6f}m)")
        else:
            print("❌ No IK solution found")
        
        print("\n✅ Enhanced IK solver test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_ik_solver() 