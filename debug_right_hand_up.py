import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import action_map

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} networkInterface")
    sys.exit(-1)

print("WARNING: Please ensure there are no obstacles around the robot.")
input("Press Enter to continue...")

ChannelFactoryInitialize(0, sys.argv[1])

client = G1ArmActionClient()
client.SetTimeout(10.0)
client.Init()

print("action_map:", action_map)

print("Calling release arm...")
ret = client.ExecuteAction(action_map.get("release arm"))
print("release arm ret:", ret)

time.sleep(2)

print("Calling right hand up...")
ret = client.ExecuteAction(action_map.get("right hand up"))
print("right hand up ret:", ret)

time.sleep(3)

print("Calling release arm again...")
ret = client.ExecuteAction(action_map.get("release arm"))
print("release arm ret:", ret)
