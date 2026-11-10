import argparse
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


RIGHT_HAND = slice(0, 6)
LEFT_HAND = slice(6, 12)
INSPIRE_JOINTS = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class InspireStateReader:
    def __init__(self):
        self.state = None

    def init(self):
        self.subscriber = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self.subscriber.Init(self.handler, 10)

    def handler(self, msg: MotorStates_):
        self.state = msg

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.state is None and time.monotonic() < deadline:
            time.sleep(0.02)
        return self.state


def make_cmd(q_values):
    cmd = MotorCmds_()
    cmd.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
    for i, q in enumerate(q_values):
        cmd.cmds[i].q = float(q)
    return cmd


def state_positions(state):
    if len(state.states) < 12:
        raise RuntimeError(f"Expected 12 Inspire hand states, got {len(state.states)}.")
    return np.array([state.states[i].q for i in range(12)], dtype=float)


def print_hand_state(q_values):
    for hand_name, hand_slice in (("right", RIGHT_HAND), ("left", LEFT_HAND)):
        pairs = [
            f"{joint}={q_values[hand_slice.start + i]:.3f}"
            for i, joint in enumerate(INSPIRE_JOINTS)
        ]
        print(f"{hand_name}: " + ", ".join(pairs))


def publish_pose(publisher, q_values, duration, rate):
    cmd = make_cmd(q_values)
    period = 1.0 / rate
    end_time = time.monotonic() + duration
    next_tick = time.monotonic()
    while time.monotonic() < end_time:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(next_tick - now, 0.005))
            continue
        next_tick += period
        publisher.Write(cmd)


def move_pose(publisher, q_start, q_goal, duration, rate):
    period = 1.0 / rate
    start_time = time.monotonic()
    next_tick = start_time
    while True:
        now = time.monotonic()
        elapsed = now - start_time
        if elapsed > duration:
            break

        if now < next_tick:
            time.sleep(min(next_tick - now, 0.005))
            continue
        next_tick += period

        r = smoothstep(elapsed / duration)
        q = (1.0 - r) * q_start + r * q_goal
        publisher.Write(make_cmd(q))


def main():
    parser = argparse.ArgumentParser(
        description="Run one Inspire RH56DFX grasp. q is normalized: 0=close, 1=open."
    )
    parser.add_argument("interface", help="Network interface connected to the robot, e.g. enx68da73a54ce1")
    parser.add_argument("--hand", choices=["right", "left", "both"], default="right", help="Hand to close.")
    parser.add_argument("--close-q", type=float, default=0.0, help="Target close value, 0=full close, 1=open.")
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Move selected hand relative to current q. Negative closes, positive opens.",
    )
    parser.add_argument("--thumb-rotation-q", type=float, default=None, help="Optional thumb rotation target.")
    parser.add_argument("--move-duration", type=float, default=2.0, help="Seconds to close the hand.")
    parser.add_argument("--hold-duration", type=float, default=2.0, help="Seconds to hold the grasp.")
    parser.add_argument("--return-to-start", action="store_true", help="Return to the starting hand posture afterward.")
    parser.add_argument("--open-after", action="store_true", help="Open the selected hand afterward.")
    parser.add_argument("--rate", type=float, default=50.0, help="Command publish rate in Hz.")
    parser.add_argument("--state-timeout", type=float, default=3.0, help="Seconds to wait for inspire state.")
    parser.add_argument(
        "--allow-no-state",
        action="store_true",
        help="Send commands even if rt/inspire/state is unavailable; assumes all joints start open.",
    )
    parser.add_argument("--read-only", action="store_true", help="Read and print rt/inspire/state without moving.")
    parser.add_argument("--yes", action="store_true", help="Confirm the hand has clear space and can safely move.")
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to move the Inspire hand without --yes.")
    if not (0.0 <= args.close_q <= 1.0):
        raise SystemExit("--close-q must be between 0 and 1.")
    if args.thumb_rotation_q is not None and not (0.0 <= args.thumb_rotation_q <= 1.0):
        raise SystemExit("--thumb-rotation-q must be between 0 and 1.")
    if args.move_duration <= 0 or args.hold_duration < 0 or args.rate <= 0:
        raise SystemExit("--move-duration and --rate must be positive; --hold-duration cannot be negative.")

    ChannelFactoryInitialize(0, args.interface)
    reader = InspireStateReader()
    reader.init()
    state = reader.wait(args.state_timeout)
    if state is None:
        if not args.allow_no_state:
            raise SystemExit("No rt/inspire/state received; is the Inspire hand service running on the robot?")
        print("warning: no rt/inspire/state received; assuming both hands start open.")
        q_start = np.ones(12, dtype=float)
    else:
        q_start = np.clip(state_positions(state), 0.0, 1.0)

    print("current_q:")
    print_hand_state(q_start)

    if args.read_only:
        return

    q_goal = q_start.copy()

    selected = []
    if args.hand in ("right", "both"):
        selected.append(("right", RIGHT_HAND))
    if args.hand in ("left", "both"):
        selected.append(("left", LEFT_HAND))

    for _, hand_slice in selected:
        if args.delta is None:
            q_goal[hand_slice] = args.close_q
        else:
            q_goal[hand_slice] = np.clip(q_goal[hand_slice] + args.delta, 0.0, 1.0)
        if args.thumb_rotation_q is not None:
            q_goal[hand_slice.start + 5] = args.thumb_rotation_q

    publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
    publisher.Init()

    print(f"hand: {args.hand}")
    print(f"joints: {INSPIRE_JOINTS}")
    print(f"start_q: {q_start.tolist()}")
    print(f"target_q: {q_goal.tolist()}")

    move_pose(publisher, q_start, q_goal, args.move_duration, args.rate)
    publish_pose(publisher, q_goal, args.hold_duration, args.rate)

    if args.open_after:
        q_open = q_goal.copy()
        for _, hand_slice in selected:
            q_open[hand_slice] = 1.0
        move_pose(publisher, q_goal, q_open, args.move_duration, args.rate)
    elif args.return_to_start:
        move_pose(publisher, q_goal, q_start, args.move_duration, args.rate)

    print("Done!")


if __name__ == "__main__":
    main()
