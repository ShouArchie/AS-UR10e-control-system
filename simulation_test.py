"""
UR10e Simulation Test Script
Tests kinematic model with UR simulator and face tracking starting position.
Enhanced with URDF-based IK solver for maximum accuracy.
"""

import urx
import math
import time
import sys
from ur_ik_solver import URIKSolver

class UR10eSimulationTest:
    """
    Test class for UR10e simulation with enhanced kinematic model verification.
    Uses URDF-based IK solver when available, falls back to manual definition.
    """
    
    def __init__(self, sim_ip="192.168.10.223"):
        """Initialize simulation test with UR simulator IP."""
        self.sim_ip = sim_ip
        self.robot = None
        
        # Initialize enhanced IK solver with URDF support
        print("Initializing enhanced IK solver...")
        try:
            # Try URDF first, fallback to manual
            self.ik_solver = URIKSolver("ur10e.urdf", use_urdf=True)
        except Exception as e:
            print(f"Warning: Could not initialize IK solver: {e}")
            print("Falling back to basic initialization...")
            self.ik_solver = URIKSolver(use_urdf=False)
        
        # Starting joint values from face_tracking_camera.py
        self.face_tracking_start_joints = [
            0.0,                            # Base joint (joint 0) = 0°
            math.radians(-60.0),           # Shoulder joint (joint 1) = -60°
            math.radians(80.0),            # Elbow joint (joint 2) = 80°
            math.radians(-110.0),          # Wrist 1 joint (joint 3) = -110°
            3*math.pi/2,                   # Wrist 2 joint (joint 4) = 270°
            -math.pi/2                     # Wrist 3 joint (joint 5) = -90°
        ]
        
        print("✓ UR10e Simulation Test initialized")
        print(f"Simulator IP: {self.sim_ip}")
        print(f"Starting joints: {[f'{math.degrees(j):.1f}°' for j in self.face_tracking_start_joints]}")
    
    def connect_to_simulator(self):
        """Connect to UR simulator."""
        try:
            print(f"\nConnecting to UR simulator at {self.sim_ip}...")
            self.robot = urx.Robot(self.sim_ip)
            print("✓ Successfully connected to UR simulator!")
            
            # Get current robot state
            current_joints = self.robot.getj()
            
            # Get pose using getl() to avoid PoseVector issues
            pose_vector = self.robot.getl()
            current_pose = [
                pose_vector[0],  # x
                pose_vector[1],  # y
                pose_vector[2],  # z
                pose_vector[3],  # rx
                pose_vector[4],  # ry
                pose_vector[5]   # rz
            ]
            
            print(f"Current joint angles: {[f'{math.degrees(j):.1f}°' for j in current_joints]}")
            print(f"Current pose: {[f'{x:.3f}' for x in current_pose]}")
            
            # Check robot mode and warn if needed
            try:
                robot_mode = self.robot.get_mode()
                print(f"Robot mode: {robot_mode}")
                if robot_mode != 7:  # 7 = Running mode
                    print("⚠️ WARNING: Robot may not be in proper mode for remote control")
                    print("Please ensure the robot is in 'Remote' mode in PolyScope")
                    print("Steps: PolyScope → Menu → Settings → System → Remote Control = Enable")
            except:
                print("⚠️ Could not check robot mode - ensure robot is in Remote mode")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to simulator: {e}")
            return False
    
    def disconnect_from_simulator(self):
        """Disconnect from simulator."""
        if self.robot:
            self.robot.close()
            print("✓ Disconnected from simulator")
    
    def test_kinematic_model_accuracy(self):
        """Test forward kinematics accuracy against actual robot position."""
        print("\n=== Testing Enhanced Kinematic Model Accuracy ===")
        
        try:
            # Get actual robot joint positions
            actual_joints = self.robot.getj()
            
            # Get pose using getl() to avoid PoseVector issues
            pose_vector = self.robot.getl()
            actual_pose = [
                pose_vector[0],  # x
                pose_vector[1],  # y
                pose_vector[2],  # z
                pose_vector[3],  # rx
                pose_vector[4],  # ry
                pose_vector[5]   # rz
            ]
            
            # Compute forward kinematics with our enhanced model
            predicted_pos, predicted_orient = self.ik_solver.forward_kinematics(actual_joints)
            
            # Display solver information
            solver_type = "URDF-based" if self.ik_solver.using_urdf else "Manual"
            print(f"Using {solver_type} kinematic solver")
            
            # Compare positions
            pos_error = [
                actual_pose[0] - predicted_pos[0],
                actual_pose[1] - predicted_pos[1], 
                actual_pose[2] - predicted_pos[2]
            ]
            pos_error_magnitude = math.sqrt(sum(e**2 for e in pos_error))
            
            # Compare orientations  
            orient_error = [
                actual_pose[3] - predicted_orient[0],
                actual_pose[4] - predicted_orient[1],
                actual_pose[5] - predicted_orient[2]
            ]
            orient_error_magnitude = math.sqrt(sum(e**2 for e in orient_error))
            
            print("\nForward Kinematics Accuracy Test:")
            print(f"  Joint angles: {[f'{math.degrees(j):.1f}°' for j in actual_joints]}")
            print(f"  Actual position:    [{actual_pose[0]:.4f}, {actual_pose[1]:.4f}, {actual_pose[2]:.4f}]")
            print(f"  Predicted position: [{predicted_pos[0]:.4f}, {predicted_pos[1]:.4f}, {predicted_pos[2]:.4f}]")
            print(f"  Position error:     [{pos_error[0]:.4f}, {pos_error[1]:.4f}, {pos_error[2]:.4f}]")
            print(f"  Position error magnitude: {pos_error_magnitude:.4f}m ({pos_error_magnitude*1000:.1f}mm)")
            
            print(f"  Actual orientation:    [{math.degrees(actual_pose[3]):.2f}°, {math.degrees(actual_pose[4]):.2f}°, {math.degrees(actual_pose[5]):.2f}°]")
            print(f"  Predicted orientation: [{math.degrees(predicted_orient[0]):.2f}°, {math.degrees(predicted_orient[1]):.2f}°, {math.degrees(predicted_orient[2]):.2f}°]")
            print(f"  Orientation error:     [{math.degrees(orient_error[0]):.2f}°, {math.degrees(orient_error[1]):.2f}°, {math.degrees(orient_error[2]):.2f}°]")
            print(f"  Orientation error magnitude: {math.degrees(orient_error_magnitude):.2f}°")
            
            # Enhanced accuracy assessment with improved thresholds
            print("\nAccuracy Assessment:")
            if pos_error_magnitude < 0.005:  # 5mm tolerance
                print("  ✅ Position accuracy: EXCELLENT (< 5mm)")
            elif pos_error_magnitude < 0.02:  # 20mm tolerance
                print("  ✅ Position accuracy: GOOD (< 20mm)")
            elif pos_error_magnitude < 0.1:  # 100mm tolerance
                print("  ⚠️ Position accuracy: ACCEPTABLE (< 100mm)")
            else:
                print("  ❌ Position accuracy: POOR (> 100mm)")
                
            if math.degrees(orient_error_magnitude) < 2:  # 2° tolerance
                print("  ✅ Orientation accuracy: EXCELLENT (< 2°)")
            elif math.degrees(orient_error_magnitude) < 10:  # 10° tolerance
                print("  ✅ Orientation accuracy: GOOD (< 10°)")
            elif math.degrees(orient_error_magnitude) < 30:  # 30° tolerance
                print("  ⚠️ Orientation accuracy: ACCEPTABLE (< 30°)")
            else:
                print("  ❌ Orientation accuracy: POOR (> 30°)")
            
            # Test inverse kinematics if accuracy is reasonable
            if pos_error_magnitude < 0.1:  # Only test IK if FK is reasonable
                print("\nTesting Inverse Kinematics:")
                try:
                    ik_solution = self.ik_solver.solve_ik(predicted_pos, predicted_orient, actual_joints)
                    if ik_solution:
                        # Check if IK solution is close to original joints
                        joint_errors = [abs(ik_solution[i] - actual_joints[i]) for i in range(6)]
                        max_joint_error = max(joint_errors)
                        
                        print(f"  IK solution: {[f'{math.degrees(j):.1f}°' for j in ik_solution]}")
                        print(f"  Joint errors: {[f'{math.degrees(e):.1f}°' for e in joint_errors]}")
                        print(f"  Max joint error: {math.degrees(max_joint_error):.1f}°")
                        
                        if max_joint_error < math.radians(5):
                            print("  ✅ Inverse kinematics: EXCELLENT")
                        elif max_joint_error < math.radians(15):
                            print("  ✅ Inverse kinematics: GOOD")
                        else:
                            print("  ⚠️ Inverse kinematics: NEEDS IMPROVEMENT")
                    else:
                        print("  ❌ Inverse kinematics: FAILED to find solution")
                except Exception as e:
                    print(f"  ❌ Inverse kinematics: ERROR - {e}")
            
            return pos_error_magnitude, orient_error_magnitude
            
        except Exception as e:
            print(f"✗ Error testing kinematic accuracy: {e}")
            return None, None
    
    def move_to_face_tracking_start_position(self):
        """Move robot to face tracking starting position."""
        print("\n=== Moving to Face Tracking Start Position ===")
        
        try:
            print("Target joint angles:")
            for i, angle in enumerate(self.face_tracking_start_joints):
                print(f"  Joint {i}: {math.degrees(angle):.1f}°")
            
            # Move to starting position
            print("Moving robot...")
            self.robot.movej(self.face_tracking_start_joints, acc=0.3, vel=0.2)
            
            # Wait for movement to complete
            print("Waiting for movement to complete...")
            time.sleep(1)
            while self.robot.is_program_running():
                time.sleep(0.1)
            
            print("✓ Movement completed!")
            
            # Verify final position
            final_joints = self.robot.getj()
            final_pose = [self.robot.x, self.robot.y, self.robot.z, 
                         self.robot.rx, self.robot.ry, self.robot.rz]
            
            print("Final robot state:")
            print(f"  Joint angles: {[f'{math.degrees(j):.1f}°' for j in final_joints]}")
            print(f"  Cartesian pose: {[f'{x:.3f}' for x in final_pose]}")
            
            # Calculate reach from base
            reach = math.sqrt(final_pose[0]**2 + final_pose[1]**2 + final_pose[2]**2)
            print(f"  Reach from base: {reach:.3f}m")
            
            return True
            
        except Exception as e:
            print(f"✗ Error moving to start position: {e}")
            return False
    
    def test_joint_movements(self):
        """Test individual joint movements."""
        print("\n=== Testing Individual Joint Movements ===")
        
        try:
            # Get starting position
            start_joints = self.robot.getj()
            print(f"Starting from: {[f'{math.degrees(j):.1f}°' for j in start_joints]}")
            
            # Test each joint with small movements
            joint_names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
            test_movements = [10, 10, 10, 15, 15, 15]  # degrees
            
            for i, (name, movement_deg) in enumerate(zip(joint_names, test_movements)):
                print(f"\nTesting {name} joint (Joint {i}):")
                
                # Calculate target joints
                target_joints = start_joints.copy()
                target_joints[i] += math.radians(movement_deg)
                
                print(f"  Moving {movement_deg}° → {math.degrees(target_joints[i]):.1f}°")
                
                # Move joint
                self.robot.movej(target_joints, acc=0.5, vel=0.3)
                time.sleep(0.5)
                while self.robot.is_program_running():
                    time.sleep(0.1)
                
                # Verify movement
                actual_joints = self.robot.getj()
                actual_movement = math.degrees(actual_joints[i] - start_joints[i])
                error = abs(actual_movement - movement_deg)
                
                print(f"  Actual movement: {actual_movement:.1f}° (error: {error:.1f}°)")
                
                if error < 1.0:
                    print(f"  ✅ {name} joint movement: ACCURATE")
                elif error < 5.0:
                    print(f"  ✅ {name} joint movement: ACCEPTABLE")
                else:
                    print(f"  ⚠️ {name} joint movement: HIGH ERROR")
                
                # Return to start position
                self.robot.movej(start_joints, acc=0.5, vel=0.3)
                time.sleep(0.5)
                while self.robot.is_program_running():
                    time.sleep(0.1)
            
            print("\n✓ Joint movement tests completed")
            return True
            
        except Exception as e:
            print(f"✗ Error testing joint movements: {e}")
            return False
    
    def test_cartesian_movements(self):
        """Test small Cartesian movements for face tracking."""
        print("\n=== Testing Cartesian Movements ===")
        
        try:
            # Get starting pose
            start_pose = [self.robot.x, self.robot.y, self.robot.z, 
                         self.robot.rx, self.robot.ry, self.robot.rz]
            print(f"Starting pose: {[f'{x:.3f}' for x in start_pose]}")
            
            # Test movements (similar to face tracking)
            movements = {
                "Forward (X+)": [0.05, 0, 0, 0, 0, 0],
                "Right (Y+)": [0, 0.05, 0, 0, 0, 0], 
                "Up (Z+)": [0, 0, 0.05, 0, 0, 0],
                "Backward (X-)": [-0.05, 0, 0, 0, 0, 0],
                "Left (Y-)": [0, -0.05, 0, 0, 0, 0],
                "Down (Z-)": [0, 0, -0.05, 0, 0, 0]
            }
            
            for movement_name, movement_vector in movements.items():
                print(f"\nTesting {movement_name}:")
                
                # Calculate target pose
                target_pose = [start_pose[i] + movement_vector[i] for i in range(6)]
                print(f"  Target: {[f'{x:.3f}' for x in target_pose]}")
                
                # Move robot
                self.robot.movel(target_pose, acc=0.3, vel=0.1)
                time.sleep(0.5)
                while self.robot.is_program_running():
                    time.sleep(0.1)
                
                # Verify movement
                actual_pose = [self.robot.x, self.robot.y, self.robot.z, 
                              self.robot.rx, self.robot.ry, self.robot.rz]
                
                # Calculate actual movement
                actual_movement = [actual_pose[i] - start_pose[i] for i in range(3)]
                movement_error = [abs(actual_movement[i] - movement_vector[i]) for i in range(3)]
                total_error = sum(movement_error)
                
                print(f"  Actual: {[f'{x:.3f}' for x in actual_pose]}")
                print(f"  Error: {[f'{e:.3f}' for e in movement_error]} (total: {total_error:.3f})")
                
                if total_error < 0.005:  # 5mm total error
                    print(f"  ✅ {movement_name}: EXCELLENT")
                elif total_error < 0.015:  # 15mm total error
                    print(f"  ✅ {movement_name}: GOOD")
                else:
                    print(f"  ⚠️ {movement_name}: HIGH ERROR")
                
                # Return to start position
                self.robot.movel(start_pose, acc=0.3, vel=0.1)
                time.sleep(0.5)
                while self.robot.is_program_running():
                    time.sleep(0.1)
            
            print("\n✓ Cartesian movement tests completed")
            return True
            
        except Exception as e:
            print(f"✗ Error testing Cartesian movements: {e}")
            return False
    
    def run_full_test_suite(self):
        """Run complete simulation test suite."""
        print("="*60)
        print("UR10e SIMULATION TEST SUITE")
        print("="*60)
        
        # Connect to simulator
        if not self.connect_to_simulator():
            return False
        
        try:
            # Test 1: Move to face tracking start position
            success1 = self.move_to_face_tracking_start_position()
            
            # Test 2: Test kinematic model accuracy
            pos_error, orient_error = self.test_kinematic_model_accuracy()
            
            # Test 3: Test individual joint movements
            success3 = self.test_joint_movements()
            
            # Test 4: Test Cartesian movements
            success4 = self.test_cartesian_movements()
            
            # Final summary
            print("\n" + "="*60)
            print("ENHANCED TEST RESULTS SUMMARY")
            print("="*60)
            
            # Enhanced accuracy assessment
            pos_acceptable = pos_error and pos_error < 0.1  # 100mm threshold
            orient_acceptable = orient_error and math.degrees(orient_error) < 30  # 30° threshold
            kinematic_accuracy = pos_acceptable and orient_acceptable
            
            solver_type = "URDF-based" if hasattr(self.ik_solver, 'using_urdf') and self.ik_solver.using_urdf else "Manual"
            print(f"IK Solver Type: {solver_type}")
            print(f"1. Face tracking start position: {'✅ PASS' if success1 else '❌ FAIL'}")
            print(f"2. Enhanced kinematic model accuracy: {'✅ PASS' if kinematic_accuracy else '❌ FAIL'}")
            if pos_error and orient_error:
                print(f"   - Position error: {pos_error*1000:.1f}mm")
                print(f"   - Orientation error: {math.degrees(orient_error):.1f}°")
                
                # Detailed accuracy rating
                if pos_error < 0.005:
                    print(f"   - Position rating: EXCELLENT")
                elif pos_error < 0.02:
                    print(f"   - Position rating: GOOD")
                elif pos_error < 0.1:
                    print(f"   - Position rating: ACCEPTABLE")
                else:
                    print(f"   - Position rating: POOR")
                    
                if math.degrees(orient_error) < 2:
                    print(f"   - Orientation rating: EXCELLENT")
                elif math.degrees(orient_error) < 10:
                    print(f"   - Orientation rating: GOOD")
                elif math.degrees(orient_error) < 30:
                    print(f"   - Orientation rating: ACCEPTABLE")
                else:
                    print(f"   - Orientation rating: POOR")
                    
            print(f"3. Joint movements: {'✅ PASS' if success3 else '❌ FAIL'}")
            print(f"4. Cartesian movements: {'✅ PASS' if success4 else '❌ FAIL'}")
            
            overall_success = success1 and kinematic_accuracy and success3 and success4
            print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")
            
            if overall_success:
                print("\n🎉 Enhanced UR10e simulation model is ready for face tracking!")
                if solver_type == "URDF-based":
                    print("   Using official URDF parameters for maximum accuracy!")
                else:
                    print("   Using manually defined parameters based on official URDF.")
            else:
                print("\n⚠️ Please review failed tests before proceeding.")
                if not kinematic_accuracy:
                    print("   Consider checking coordinate system differences between")
                    print("   simulator and kinematic model, or robot-specific calibration.")
            
            return overall_success
            
        finally:
            self.disconnect_from_simulator()


def main():
    """Main test function."""
    print("UR10e Simulation Test with Face Tracking Start Position")
    print("This script tests the kinematic model with UR simulator")
    
    # Create test instance
    sim_test = UR10eSimulationTest("192.168.10.223")
    
    # Run tests
    success = sim_test.run_full_test_suite()
    
    if success:
        print("\n✅ All tests passed! Ready for face tracking simulation.")
    else:
        print("\n⚠️ Some tests failed. Check the results above.")
    
    return success


if __name__ == "__main__":
    main() 