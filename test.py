import socket
import time
import math
HOST = "192.168.0.101"  # Replace with your robot's IP
PORT = 30003            # Real-time client interface

shoulder_angle_deg = -60.0
elbow_angle_deg = 80.0

shoulder_angle_rad = math.radians(shoulder_angle_deg)
elbow_angle_rad = math.radians(elbow_angle_deg)

# Apply wrist1 formula: 90° + elbow_angle - abs(shoulder_angle)
wrist1_angle_deg = 90 + elbow_angle_deg - abs(shoulder_angle_deg)
wrist1_angle_rad = math.radians(-wrist1_angle_deg)

# Apply wrist3 formula: -90° + base_angle
base_angle_deg = 0.0
wrist3_angle_deg = -90 + base_angle_deg
wrist3_angle_rad = math.radians(wrist3_angle_deg)

start_joints = [
                0.0,                    # Base joint = 0°
                shoulder_angle_rad,     # Shoulder joint = -60°
                elbow_angle_rad,        # Elbow joint = 80°
                wrist1_angle_rad,       # Wrist 1 (calculated)
                3*math.pi/2,            # Wrist 2 (keep current or specific value)
                wrist3_angle_rad        # Wrist 3 (calculated)
            ]


start_joints2 = [
                0.0,                    # Base joint = 0°
                shoulder_angle_rad,     # Shoulder joint = -60°
                elbow_angle_rad,        # Elbow joint = 80°
                wrist1_angle_rad,       # Wrist 1 (calculated)
                2*math.pi/2,            # Wrist 2 (keep current or specific value)
                wrist3_angle_rad        # Wrist 3 (calculated)
            ]

# Example joint positions (in radians)
joint_positions = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]

# Create the URScript command
servoj_cmd = (
    "servoj([{}, {}, {}, {}, {}, {}], t = 1, lookahead_time=0.005, gain=300)\n"
    .format(*start_joints)
)

servoj_cmd2 = (
    "servoj([{}, {}, {}, {}, {}, {}], t = 1, lookahead_time=0.000001, gain=300)\n"
    .format(*start_joints2)
)
# Open socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
for i in range(100):
    s.send(servoj_cmd2.encode('utf8'))
    time.sleep(0.05)


# Send the command
for i in range(100):
    s.send(servoj_cmd.encode('utf8'))
    time.sleep(0.008)


# Keep the connection open for a short time
time.sleep(0.1)

# Close the socket
s.close()