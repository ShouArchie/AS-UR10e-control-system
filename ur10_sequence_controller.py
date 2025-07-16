"""Simple UR10 sequence:
1. movej to home (START_JOINTS)
2. wait until reached (getj)
3. rotate +Ry by DEG_ROTATE with movel
4. wait until reached (getl)
5. rotate −Ry back to original with movel
6. done
"""

import math
import time
from typing import List, Sequence

import urx
from urx.urrobot import RobotException

# ── User parameters ────────────────────────────────────────────────────────
ROBOT_IP = "192.168.10.223"      # UR10 IP address
DEG_ROTATE = 90                 # Ry rotation magnitude (degrees)
ACC = 0.5                         # Acceleration for moves
VEL = 0.25                        # Velocity for moves

# Starting joint angles in degrees → radians
START_JOINTS_DEG = [0, -60, 80, -110, 270, -90]
START_JOINTS = [math.radians(a) for a in START_JOINTS_DEG]

# Thresholds
JOINT_EPS = math.radians(1.0)     # 1° tolerance
POS_EPS = 1e-3                    # 1 mm tolerance
ORI_EPS = math.radians(1.0)       # 1° orientation tolerance
POLL = 0.05                       # polling period
TIMEOUT = 20.0                    # max wait per move (s)

# ── Helpers ────────────────────────────────────────────────────────────────

def wait_until_joints(robot: urx.Robot, target: Sequence[float]) -> None:
    start = time.time()
    while True:
        cur = robot.getj()
        if max(abs(c - t) for c, t in zip(cur, target)) < JOINT_EPS:
            return
        if time.time() - start > TIMEOUT:
            print("⚠️  joint wait timeout; continuing")
            return
        time.sleep(POLL)

def get_tcp_pose(robot: urx.Robot) -> List[float]:
    pose = robot.getl()
    if pose is None or len(pose) != 6:
        raise RuntimeError("Invalid TCP pose from robot")
    return list(map(float, pose))

def wait_until_pose(robot: urx.Robot, target: Sequence[float]) -> None:
    start = time.time()
    while True:
        cur = get_tcp_pose(robot)
        pos_err = max(abs(c - t) for c, t in zip(cur[:3], target[:3]))
        ori_err = max(abs(c - t) for c, t in zip(cur[3:], target[3:]))
        if pos_err < POS_EPS and ori_err < ORI_EPS:
            return
        if time.time() - start > TIMEOUT:
            print("⚠️  pose wait timeout; continuing")
            return
        time.sleep(POLL)

def send_movel(robot: urx.Robot, pose: Sequence[float]):
    pose_str = ", ".join(f"{v:.6f}" for v in pose)
    robot.send_program(f"movel(p[{pose_str}], a={ACC}, v={VEL})")

# ── Main sequence ──────────────────────────────────────────────────────────

def main() -> None:
    robot = urx.Robot(ROBOT_IP)
    try:
        print(f"✓ Connected to UR10 at {ROBOT_IP}")

        # Move to home joints
        print("Moving to start joints …")
        try:
            robot.movej(START_JOINTS, acc=ACC, vel=VEL, wait=False)
        except RobotException as e:
            if "Robot stopped" not in str(e):
                raise
        wait_until_joints(robot, START_JOINTS)

        start_pose = get_tcp_pose(robot)
        print("Home TCP pose:", [round(v, 4) for v in start_pose])

        # Rotate +Ry
        pose_up = start_pose.copy()
        pose_up[4] += math.radians(DEG_ROTATE)
        print(f"Rotating +Ry by {DEG_ROTATE}° …")
        send_movel(robot, pose_up)
        wait_until_pose(robot, pose_up)

        # Rotate back down
        print("Rotating −Ry back …")
        send_movel(robot, start_pose)
        wait_until_pose(robot, start_pose)

        print("✓ Sequence complete")
    finally:
        print("Closing connection …")
        robot.close()

if __name__ == "__main__":
    main() 