import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} networkInterface")
    sys.exit(-1)

print("WARNING: Please ensure the robot is stable and no obstacles are around it.")
input("Press Enter to continue...")

ChannelFactoryInitialize(0, sys.argv[1])

msc = MotionSwitcherClient()
msc.SetTimeout(10.0)
msc.Init()

print("SelectMode(ai)...")
print("SelectMode ret:", msc.SelectMode("ai"))

print("MotionSwitcher WaitLeaseApplied...")
print("MotionSwitcher WaitLeaseApplied ret:", msc.WaitLeaseApplied())

print("CheckMode:", msc.CheckMode())

loco = LocoClient()
loco.SetTimeout(10.0)
loco.Init()

print("Loco WaitLeaseApplied...")
print("Loco WaitLeaseApplied ret:", loco.WaitLeaseApplied())

print("Calling Start()...")
try:
    print("Start ret:", loco.Start())
except Exception as e:
    print("Start exception:", e)

time.sleep(2)

print("Calling BalanceStand()...")
try:
    print("BalanceStand ret:", loco.BalanceStand())
except Exception as e:
    print("BalanceStand exception:", e)

time.sleep(2)

print("Calling HighStand()...")
try:
    print("HighStand ret:", loco.HighStand())
except Exception as e:
    print("HighStand exception:", e)

time.sleep(2)

print("Done preparing G1 ready state.")
