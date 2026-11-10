import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} networkInterface")
    sys.exit(-1)

print("WARNING: Please ensure the robot is stable, supported if necessary, and no obstacles are around it.")
input("Press Enter to continue...")

ChannelFactoryInitialize(0, sys.argv[1])

client = LocoClient()
client.SetTimeout(10.0)
client.Init()

print("Calling SetFsmId(500)...")
ret = client.SetFsmId(500)
print("SetFsmId(500) ret:", ret)

time.sleep(2)
print("Done.")
