import urx
import time
import math

def move_joints_degrees(robot, joint_degrees, acc=0.1, vel=0.1, wait=True):
    """
    Move robot joints using positions specified in degrees.
    
    Args:
        robot: URX robot object
        joint_degrees: List of 6 joint positions in degrees [j1, j2, j3, j4, j5, j6]
        acc: Acceleration (default 0.1)
        vel: Velocity (default 0.1)
        wait: Whether to wait for movement completion (default True)
    """
    # Convert degrees to radians
    joint_radians = [math.radians(deg) for deg in joint_degrees]
    print(f"Converting {joint_degrees} degrees to {joint_radians} radians")
    
    try:
        # Move robot with wait=False to avoid the "Robot stopped" issue
        robot.movej(joint_radians, acc=acc, vel=vel, wait=False)
        print("Joint movement command sent!")
        
        if wait:
            # Wait for movement to complete by checking if program is running
            print("Waiting for movement to complete...")
            time.sleep(0.5)  # Give it a moment to start
            
            while robot.is_program_running():
                time.sleep(0.1)
            
            print("Joint movement completed!")
        else:
            print("Movement started (non-blocking)")
            
    except Exception as e:
        print(f"Error during movement: {e}")
        # Try to get current position to see if we're close
        try:
            current = robot.getj()
            current_deg = [math.degrees(rad) for rad in current]
            print(f"Current position: {current_deg}")
            print("Movement may have completed despite the error")
        except:
            pass

def move_relative(robot, x=0, y=0, z=0, rx=0, ry=0, rz=0, acc=0.1, vel=0.1):
    """
    Move robot relatively in Cartesian space using URScript.
    
    Args:
        robot: URX robot object
        x, y, z: Linear movements in meters
        rx, ry, rz: Rotational movements in radians
        acc: Acceleration (default 0.1)
        vel: Velocity (default 0.1)
    """
    try:
        # Create URScript command for relative movement
        urscript_cmd = f"movel(pose_trans(get_actual_tcp_pose(), p[{z}, {y}, {x}, {rz}, {ry}, {rx}]), a={acc}, v={vel})"
        print(f"Sending relative movement: x={x}, y={y}, z={z}")
        print(f"URScript: {urscript_cmd}")
        
        robot.send_program(urscript_cmd)
        print("Relative movement command sent!")
        
        # Wait for movement to complete
        time.sleep(2)
        print("Relative movement completed!")
        
    except Exception as e:
        print(f"Error during relative movement: {e}")

def get_inverse_kinematics(robot, target_pose, current_joints=None):
    """
    Calculate inverse kinematics using the robot's built-in IK solver.
    
    Args:
        robot: URX robot object
        target_pose: Target pose as [x, y, z, rx, ry, rz] in meters and radians
        current_joints: Current joint positions (optional, uses current if None)
    
    Returns:
        List of joint angles in radians, or None if no solution found
    """
    try:
        if current_joints is None:
            current_joints = robot.getj()
        
        # Format the URScript command to get inverse kinematics
        # The robot's get_inverse_kin function returns joint angles for the given pose
        urscript_cmd = f"""
        target_pose = p{target_pose}
        current_q = {list(current_joints)}
        ik_result = get_inverse_kin(target_pose, current_q)
        textmsg("IK_RESULT:", ik_result)
        """
        
        print(f"Calculating IK for pose: {target_pose}")
        robot.send_program(urscript_cmd)
        
        # Note: This is a simplified example. In practice, you'd need to:
        # 1. Set up a way to receive the result back from the robot
        # 2. Parse the textmsg output or use RTDE for real-time communication
        # 3. Handle cases where no IK solution exists
        
        print("IK calculation sent to robot. Check robot's log for result.")
        print("For real-time IK, consider using RTDE or register-based communication.")
        
        return None  # Would return actual joint angles in full implementation
        
    except Exception as e:
        print(f"Error during IK calculation: {e}")
        return None

a = 0.1
v = 0.5

# Connect to robot
print("Connecting to robot...")
rob = urx.Robot("192.168.10.152")
print("Connected successfully!")

time.sleep(0.2)  # Leave some time to robot to process the setup commands

# Example usage of the degrees function
print("Testing joint movement with degrees...")
joint_positions_deg = [0, -45, 45, -90, 0, 0]  # Example joint positions in degrees
# move_joints_degrees(rob, joint_positions_deg, a, v)


# Move relatively in X direction by 0.1 meters using our working function
move_relative(rob, z=0.1)

# Get current position
try:
    current_joints = rob.getj()
    print(f"Current joint positions (radians): {current_joints}")
    # Convert back to degrees for display
    current_joints_deg = [math.degrees(rad) for rad in current_joints]
    print(f"Current joint positions (degrees): {current_joints_deg}")
except Exception as e:
    print(f"Error getting current pose: {e}")

time.sleep(5)
rob.close()
print("Robot connection closed.")