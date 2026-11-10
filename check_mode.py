import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

if len(sys.argv) > 1:
    ChannelFactoryInitialize(0, sys.argv[1])
else:
    ChannelFactoryInitialize(0)

msc = MotionSwitcherClient()
msc.SetTimeout(5.0)
msc.Init()

ret = msc.CheckMode()
print("CheckMode ret:", ret)
