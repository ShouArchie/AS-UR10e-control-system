import urx
import time

ROBOT_IP = "192.168.0.101"

def try_methods(robot):
    print("\n--- Testing getl() direct ---")
    try:
        pose = robot.getl()
        print("getl() result:", pose)
        if isinstance(pose, (list, tuple)) and len(pose) == 6:
            print("Direct getl() is a list/tuple of 6 floats.")
        else:
            print("Direct getl() is not a list/tuple of 6.")
    except Exception as e:
        print("getl() direct failed:", e)

    print("\n--- Testing getl().tolist() ---")
    try:
        pose = robot.getl()
        pose_list = pose.tolist()
        print("getl().tolist() result:", pose_list)
    except Exception as e:
        print("getl().tolist() failed:", e)

    print("\n--- Testing list(getl()) ---")
    try:
        pose = robot.getl()
        pose_list = list(pose)
        print("list(getl()) result:", pose_list)
    except Exception as e:
        print("list(getl()) failed:", e)

    print("\n--- Testing getl().array ---")
    try:
        pose = robot.getl()
        arr = pose.array
        print("getl().array result:", arr)
        try:
            arr_list = arr.tolist()
            print("getl().array.tolist() result:", arr_list)
        except Exception as e2:
            print("getl().array.tolist() failed:", e2)
        try:
            arr_list = list(arr)
            print("list(getl().array) result:", arr_list)
        except Exception as e2:
            print("list(getl().array) failed:", e2)
        try:
            arr_list = [float(x) for x in arr]
            print("[float(x) for x in getl().array] result:", arr_list)
        except Exception as e2:
            print("[float(x) for x in getl().array] failed:", e2)
    except Exception as e:
        print("getl().array failed:", e)

    print("\n--- Testing getl().get_array() ---")
    try:
        pose = robot.getl()
        arr = pose.get_array()
        print("getl().get_array() result:", arr)
        try:
            arr_list = arr.tolist()
            print("getl().get_array().tolist() result:", arr_list)
        except Exception as e2:
            print("getl().get_array().tolist() failed:", e2)
        try:
            arr_list = list(arr)
            print("list(getl().get_array()) result:", arr_list)
        except Exception as e2:
            print("list(getl().get_array()) failed:", e2)
    except Exception as e:
        print("getl().get_array() failed:", e)

    print("\n--- Testing [float(x) for x in getl()] ---")
    try:
        pose = robot.getl()
        pose_list = [float(x) for x in pose]
        print("[float(x) for x in getl()] result:", pose_list)
    except Exception as e:
        print("[float(x) for x in getl()] failed:", e)

    print("\n--- Testing [float(getl()[i]) for i in range(6)] ---")
    try:
        pose = robot.getl()
        pose_list = [float(pose[i]) for i in range(6)]
        print("[float(getl()[i]) for i in range(6)] result:", pose_list)
    except Exception as e:
        print("[float(getl()[i]) for i in range(6)] failed:", e)

    print("\n--- Testing get_pose().pose_vector.tolist() ---")
    try:
        t = robot.get_pose()
        pose_list = t.pose_vector.tolist()
        print("get_pose().pose_vector.tolist() result:", pose_list)
    except Exception as e:
        print("get_pose().pose_vector.tolist() failed:", e)

    print("\n--- Testing get_pose().pose_vector ---")
    try:
        t = robot.get_pose()
        pv = t.pose_vector
        print("get_pose().pose_vector result:", pv)
        try:
            pv_list = list(pv)
            print("list(get_pose().pose_vector) result:", pv_list)
        except Exception as e2:
            print("list(get_pose().pose_vector) failed:", e2)
    except Exception as e:
        print("get_pose().pose_vector failed:", e)

    print("\n--- Testing attribute access (robot.x, robot.y, ...) ---")
    try:
        x = float(robot.x) if hasattr(robot, 'x') else 0.0
        y = float(robot.y) if hasattr(robot, 'y') else 0.0
        z = float(robot.z) if hasattr(robot, 'z') else 0.0
        rx = float(robot.rx) if hasattr(robot, 'rx') else 0.0
        ry = float(robot.ry) if hasattr(robot, 'ry') else 0.0
        rz = float(robot.rz) if hasattr(robot, 'rz') else 0.0
        print("robot.x, y, z, rx, ry, rz:", [x, y, z, rx, ry, rz])
    except Exception as e:
        print("attribute access failed:", e)

if __name__ == "__main__":
    print("Connecting to robot at", ROBOT_IP)
    robot = urx.Robot(ROBOT_IP)
    time.sleep(0.2)
    try:
        try_methods(robot)
    finally:
        robot.close()
        print("Robot disconnected.") 