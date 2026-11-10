#!/usr/bin/env python3
"""
Move the G1 right hand toward a navigation-frame target using a conservative IK layer.

This is intentionally a small, self-contained example on top of Unitree's
``rt/arm_sdk`` joint command channel.  The right-arm kinematic chain below is
copied from the G1 29-DOF URDF: pelvis -> waist -> torso -> right_rubber_hand.
By default, ``--target`` is interpreted in the navigation ``base_link`` frame
and converted to the URDF ``pelvis`` frame with an approximate URDF-derived
base_link/MID360 offset.

Example dry run:
    python3 g1_right_arm_body_ik_example.py --target 0.30 -0.20 0.10

Example execution:
    python3 g1_right_arm_body_ik_example.py eth0 --target 0.30 -0.20 0.10 --execute
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC


class G1JointIndex:
    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    # Unitree arm_sdk enable weight in the official examples.
    kNotUsedJoint = 29


JOINT_NAMES: Dict[int, str] = {
    G1JointIndex.LeftShoulderPitch: "LeftShoulderPitch",
    G1JointIndex.LeftShoulderRoll: "LeftShoulderRoll",
    G1JointIndex.LeftShoulderYaw: "LeftShoulderYaw",
    G1JointIndex.LeftElbow: "LeftElbow",
    G1JointIndex.LeftWristRoll: "LeftWristRoll",
    G1JointIndex.LeftWristPitch: "LeftWristPitch",
    G1JointIndex.LeftWristYaw: "LeftWristYaw",
    G1JointIndex.RightShoulderPitch: "RightShoulderPitch",
    G1JointIndex.RightShoulderRoll: "RightShoulderRoll",
    G1JointIndex.RightShoulderYaw: "RightShoulderYaw",
    G1JointIndex.RightElbow: "RightElbow",
    G1JointIndex.RightWristRoll: "RightWristRoll",
    G1JointIndex.RightWristPitch: "RightWristPitch",
    G1JointIndex.RightWristYaw: "RightWristYaw",
    G1JointIndex.WaistYaw: "WaistYaw",
    G1JointIndex.WaistRoll: "WaistRoll",
    G1JointIndex.WaistPitch: "WaistPitch",
}


# Software limits from the G1 29-DOF URDF right-arm joints.
JOINT_LIMITS: Dict[int, Tuple[float, float]] = {
    G1JointIndex.LeftShoulderPitch: (-3.0892, 2.6704),
    G1JointIndex.LeftShoulderRoll: (-1.5882, 2.2515),
    G1JointIndex.LeftShoulderYaw: (-2.6180, 2.6180),
    G1JointIndex.LeftElbow: (-1.0472, 2.0944),
    G1JointIndex.LeftWristRoll: (-1.972222054, 1.972222054),
    G1JointIndex.LeftWristPitch: (-1.614429558, 1.614429558),
    G1JointIndex.LeftWristYaw: (-1.614429558, 1.614429558),
    G1JointIndex.RightShoulderPitch: (-3.0892, 2.6704),
    G1JointIndex.RightShoulderRoll: (-2.2515, 1.5882),
    G1JointIndex.RightShoulderYaw: (-2.6180, 2.6180),
    G1JointIndex.RightElbow: (-1.0472, 2.0944),
    G1JointIndex.RightWristRoll: (-1.972222054, 1.972222054),
    G1JointIndex.RightWristPitch: (-1.614429558, 1.614429558),
    G1JointIndex.RightWristYaw: (-1.614429558, 1.614429558),
}


RIGHT_ARM_7DOF_JOINTS = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

RIGHT_ARM_SIGN_INDEX = {joint: index for index, joint in enumerate(RIGHT_ARM_7DOF_JOINTS)}

LEFT_ARM_7DOF_JOINTS = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
]

LEFT_ARM_SIGN_INDEX = {joint: index for index, joint in enumerate(LEFT_ARM_7DOF_JOINTS)}


def right_arm_joints(arm_dof: int) -> List[int]:
    joints = [
        G1JointIndex.RightShoulderPitch,
        G1JointIndex.RightShoulderRoll,
        G1JointIndex.RightShoulderYaw,
        G1JointIndex.RightElbow,
        G1JointIndex.RightWristRoll,
    ]
    if arm_dof == 7:
        joints.extend([G1JointIndex.RightWristPitch, G1JointIndex.RightWristYaw])
    return joints


def left_arm_joints(arm_dof: int) -> List[int]:
    joints = [
        G1JointIndex.LeftShoulderPitch,
        G1JointIndex.LeftShoulderRoll,
        G1JointIndex.LeftShoulderYaw,
        G1JointIndex.LeftElbow,
        G1JointIndex.LeftWristRoll,
    ]
    if arm_dof == 7:
        joints.extend([G1JointIndex.LeftWristPitch, G1JointIndex.LeftWristYaw])
    return joints


def arm_sdk_joints(arm_dof: int) -> List[int]:
    left = [
        G1JointIndex.LeftShoulderPitch,
        G1JointIndex.LeftShoulderRoll,
        G1JointIndex.LeftShoulderYaw,
        G1JointIndex.LeftElbow,
        G1JointIndex.LeftWristRoll,
    ]
    right = right_arm_joints(arm_dof)
    if arm_dof == 7:
        left.extend([G1JointIndex.LeftWristPitch, G1JointIndex.LeftWristYaw])
    return left + right + [
        G1JointIndex.WaistYaw,
        G1JointIndex.WaistRoll,
        G1JointIndex.WaistPitch,
    ]


@dataclass(frozen=True)
class KinematicStep:
    name: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    joint_index: Optional[int] = None
    axis_xyz: Optional[Tuple[float, float, float]] = None


RIGHT_HAND_CHAIN = [
    KinematicStep("waist_yaw_joint", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), G1JointIndex.WaistYaw, (0.0, 0.0, 1.0)),
    KinematicStep("waist_roll_joint", (-0.0039635, 0.0, 0.044), (0.0, 0.0, 0.0), G1JointIndex.WaistRoll, (1.0, 0.0, 0.0)),
    KinematicStep("waist_pitch_joint", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), G1JointIndex.WaistPitch, (0.0, 1.0, 0.0)),
    KinematicStep("right_shoulder_pitch_joint", (0.0039563, -0.10021, 0.24778), (-0.27931, 5.4949e-05, 0.00019159), G1JointIndex.RightShoulderPitch, (0.0, 1.0, 0.0)),
    KinematicStep("right_shoulder_roll_joint", (0.0, -0.038, -0.013831), (0.27925, 0.0, 0.0), G1JointIndex.RightShoulderRoll, (1.0, 0.0, 0.0)),
    KinematicStep("right_shoulder_yaw_joint", (0.0, -0.00624, -0.1032), (0.0, 0.0, 0.0), G1JointIndex.RightShoulderYaw, (0.0, 0.0, 1.0)),
    KinematicStep("right_elbow_joint", (0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), G1JointIndex.RightElbow, (0.0, 1.0, 0.0)),
    KinematicStep("right_wrist_roll_joint", (0.100, -0.00188791, -0.010), (0.0, 0.0, 0.0), G1JointIndex.RightWristRoll, (1.0, 0.0, 0.0)),
    KinematicStep("right_wrist_pitch_joint", (0.038, 0.0, 0.0), (0.0, 0.0, 0.0), G1JointIndex.RightWristPitch, (0.0, 1.0, 0.0)),
    KinematicStep("right_wrist_yaw_joint", (0.051, 0.0, 0.0), (0.0, 0.0, 0.0), G1JointIndex.RightWristYaw, (0.0, 0.0, 1.0)),
    KinematicStep("right_hand_palm_joint", (0.0415, -0.003, 0.0), (0.0, 0.0, 0.0)),
]

LEFT_HAND_CHAIN = [
    KinematicStep("waist_yaw_joint", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), G1JointIndex.WaistYaw, (0.0, 0.0, 1.0)),
    KinematicStep("waist_roll_joint", (-0.0039635, 0.0, 0.044), (0.0, 0.0, 0.0), G1JointIndex.WaistRoll, (1.0, 0.0, 0.0)),
    KinematicStep("waist_pitch_joint", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), G1JointIndex.WaistPitch, (0.0, 1.0, 0.0)),
    KinematicStep("left_shoulder_pitch_joint", (0.0039563, 0.10021, 0.24778), (0.27931, 5.4949e-05, -0.00019159), G1JointIndex.LeftShoulderPitch, (0.0, 1.0, 0.0)),
    KinematicStep("left_shoulder_roll_joint", (0.0, 0.038, -0.013831), (-0.27925, 0.0, 0.0), G1JointIndex.LeftShoulderRoll, (1.0, 0.0, 0.0)),
    KinematicStep("left_shoulder_yaw_joint", (0.0, 0.00624, -0.1032), (0.0, 0.0, 0.0), G1JointIndex.LeftShoulderYaw, (0.0, 0.0, 1.0)),
    KinematicStep("left_elbow_joint", (0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), G1JointIndex.LeftElbow, (0.0, 1.0, 0.0)),
    KinematicStep("left_wrist_roll_joint", (0.100, 0.00188791, -0.010), (0.0, 0.0, 0.0), G1JointIndex.LeftWristRoll, (1.0, 0.0, 0.0)),
    KinematicStep("left_wrist_pitch_joint", (0.038, 0.0, 0.0), (0.0, 0.0, 0.0), G1JointIndex.LeftWristPitch, (0.0, 1.0, 0.0)),
    KinematicStep("left_wrist_yaw_joint", (0.051, 0.0, 0.0), (0.0, 0.0, 0.0), G1JointIndex.LeftWristYaw, (0.0, 0.0, 1.0)),
    KinematicStep("left_hand_palm_joint", (0.0415, 0.003, 0.0), (0.0, 0.0, 0.0)),
]


# Approximate navigation base_link/body origin in the URDF pelvis frame.
# With no extra calibration, this uses the URDF MID360 mounting position:
# pelvis -> waist_roll origin + torso -> mid360_link origin.
DEFAULT_BASE_LINK_IN_PELVIS_XYZ = (-0.00368, 0.00003, 0.46018)
DEFAULT_BASE_LINK_IN_PELVIS_RPY = (0.0, 0.0, 0.0)


@dataclass
class IkResult:
    success: bool
    q_model: np.ndarray
    reached: np.ndarray
    residual: float
    iterations: int
    message: str


def rot_x(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def axis_angle_to_matrix(axis: Sequence[float], theta: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_array))
    if norm <= 0.0:
        return np.eye(3)
    x, y, z = axis_array / norm
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def transform_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def transform_from_rotation(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def transform_point(transform: np.ndarray, point: Sequence[float]) -> np.ndarray:
    point_h = np.ones(4)
    point_h[:3] = np.asarray(point, dtype=float)
    return (transform @ point_h)[:3]


def lookup_map_point_in_base_link(point_map: np.ndarray, timeout: float = 3.0) -> np.ndarray:
    try:
        import rospy
        import tf
    except ImportError as exc:
        raise RuntimeError("target-frame=map requires ROS rospy/tf Python packages") from exc

    if not rospy.core.is_initialized():
        rospy.init_node("g1_right_arm_body_ik_tf_lookup", anonymous=True, disable_signals=True)

    listener = tf.TransformListener()
    listener.waitForTransform("base_link", "map", rospy.Time(0), rospy.Duration(timeout))
    trans, quat = listener.lookupTransform("base_link", "map", rospy.Time(0))
    transform = tf.transformations.quaternion_matrix(quat)
    transform[:3, 3] = np.asarray(trans, dtype=float)
    return transform_point(transform, point_map)


def target_to_pelvis_frame(
    target: np.ndarray,
    target_frame: str,
    base_link_in_pelvis_xyz: Sequence[float],
    base_link_in_pelvis_rpy: Sequence[float],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if target_frame == "pelvis":
        return target, None

    target_base = target
    if target_frame == "map":
        target_base = lookup_map_point_in_base_link(target)
    elif target_frame not in ("base_link", "body"):
        raise ValueError(f"unsupported target frame: {target_frame}")

    pelvis_from_base = transform_from_xyz_rpy(base_link_in_pelvis_xyz, base_link_in_pelvis_rpy)
    return transform_point(pelvis_from_base, target_base), target_base


def right_hand_pose(q_model: Sequence[float], waist_q: Sequence[float], joint_indices: Sequence[int]) -> np.ndarray:
    """Return pelvis -> right_rubber_hand pose from the URDF chain."""
    return hand_pose(q_model, waist_q, joint_indices, RIGHT_HAND_CHAIN)


def left_hand_pose(q_model: Sequence[float], waist_q: Sequence[float], joint_indices: Sequence[int]) -> np.ndarray:
    """Return pelvis -> left hand palm pose from the mirrored URDF chain."""
    return hand_pose(q_model, waist_q, joint_indices, LEFT_HAND_CHAIN)


def hand_pose(
    q_model: Sequence[float],
    waist_q: Sequence[float],
    joint_indices: Sequence[int],
    chain: Sequence[KinematicStep],
) -> np.ndarray:
    q_lookup = {
        G1JointIndex.WaistYaw: float(waist_q[0]),
        G1JointIndex.WaistRoll: float(waist_q[1]),
        G1JointIndex.WaistPitch: float(waist_q[2]),
    }
    q_lookup.update({joint: float(value) for joint, value in zip(joint_indices, q_model)})

    transform = np.eye(4)
    for step in chain:
        transform = transform @ transform_from_xyz_rpy(step.origin_xyz, step.origin_rpy)
        if step.joint_index is not None and step.axis_xyz is not None:
            transform = transform @ transform_from_rotation(
                axis_angle_to_matrix(step.axis_xyz, q_lookup.get(step.joint_index, 0.0))
            )
    return transform


def right_hand_fk(q_model: Sequence[float], waist_q: Sequence[float], joint_indices: Sequence[int]) -> np.ndarray:
    return right_hand_pose(q_model, waist_q, joint_indices)[:3, 3]


def left_hand_fk(q_model: Sequence[float], waist_q: Sequence[float], joint_indices: Sequence[int]) -> np.ndarray:
    return left_hand_pose(q_model, waist_q, joint_indices)[:3, 3]


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = math.atan2(-rotation[2, 0], math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw])


def numerical_jacobian(
    q_model: np.ndarray,
    waist_q: Sequence[float],
    joint_indices: Sequence[int],
    eps: float = 1e-5,
) -> np.ndarray:
    base = right_hand_fk(q_model, waist_q, joint_indices)
    jac = np.zeros((3, len(q_model)))
    for col in range(len(q_model)):
        q_eps = q_model.copy()
        q_eps[col] += eps
        jac[:, col] = (right_hand_fk(q_eps, waist_q, joint_indices) - base) / eps
    return jac


def numerical_jacobian_for(
    fk_func,
    q_model: np.ndarray,
    waist_q: Sequence[float],
    joint_indices: Sequence[int],
    eps: float = 1e-5,
) -> np.ndarray:
    base = fk_func(q_model, waist_q, joint_indices)
    jac = np.zeros((3, len(q_model)))
    for col in range(len(q_model)):
        q_eps = q_model.copy()
        q_eps[col] += eps
        jac[:, col] = (fk_func(q_eps, waist_q, joint_indices) - base) / eps
    return jac


def limits_for_ik(joint_indices: Sequence[int], signs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for idx in joint_indices:
        lo, hi = JOINT_LIMITS[idx]
        sign_index = RIGHT_ARM_SIGN_INDEX.get(idx, LEFT_ARM_SIGN_INDEX.get(idx))
        if sign_index is None:
            raise KeyError(f"joint {idx} is not in an arm sign index map")
        sign = signs[sign_index]
        if sign < 0:
            lo, hi = -hi, -lo
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower), np.asarray(upper)


def clamp_to_limits(q_model: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(q_model, lower), upper)


def solve_position_ik(
    target: np.ndarray,
    waist_q: Sequence[float],
    joint_indices: Sequence[int],
    signs: np.ndarray,
    seed: Optional[np.ndarray],
    tolerance: float,
    max_iterations: int = 240,
    damping: float = 0.08,
) -> IkResult:
    return solve_position_ik_for(
        target,
        waist_q,
        joint_indices,
        signs,
        seed,
        tolerance,
        fk_func=right_hand_fk,
        max_iterations=max_iterations,
        damping=damping,
    )


def solve_left_position_ik(
    target: np.ndarray,
    waist_q: Sequence[float],
    joint_indices: Sequence[int],
    signs: np.ndarray,
    seed: Optional[np.ndarray],
    tolerance: float,
    max_iterations: int = 240,
    damping: float = 0.08,
) -> IkResult:
    return solve_position_ik_for(
        target,
        waist_q,
        joint_indices,
        signs,
        seed,
        tolerance,
        fk_func=left_hand_fk,
        max_iterations=max_iterations,
        damping=damping,
    )


def solve_position_ik_for(
    target: np.ndarray,
    waist_q: Sequence[float],
    joint_indices: Sequence[int],
    signs: np.ndarray,
    seed: Optional[np.ndarray],
    tolerance: float,
    fk_func,
    max_iterations: int = 240,
    damping: float = 0.08,
) -> IkResult:
    lower, upper = limits_for_ik(joint_indices, signs)
    neutral = np.zeros(len(joint_indices))
    if G1JointIndex.RightElbow in joint_indices:
        neutral[joint_indices.index(G1JointIndex.RightElbow)] = 0.9
    if G1JointIndex.LeftElbow in joint_indices:
        neutral[joint_indices.index(G1JointIndex.LeftElbow)] = 0.9

    seeds = [
        neutral,
        np.array([-0.8, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0])[: len(joint_indices)],
        np.array([-1.6, -0.2, 0.2, 1.8, 0.0, 0.0, 0.0])[: len(joint_indices)],
        np.array([0.6, -0.6, -0.4, 1.0, 0.0, 0.0, 0.0])[: len(joint_indices)],
    ]
    if seed is not None:
        seeds.insert(0, seed)

    best_q = None
    best_reached = None
    best_residual = float("inf")
    best_iterations = 0

    for seed_q in seeds:
        q = clamp_to_limits(np.asarray(seed_q, dtype=float), lower, upper)
        preferred_q = q.copy()
        for iteration in range(1, max_iterations + 1):
            reached = fk_func(q, waist_q, joint_indices)
            error = target - reached
            residual = float(np.linalg.norm(error))
            if residual < best_residual:
                best_q = q.copy()
                best_reached = reached.copy()
                best_residual = residual
                best_iterations = iteration
            if residual <= tolerance:
                return IkResult(True, q, reached, residual, iteration, "IK converged")

            jac = numerical_jacobian_for(fk_func, q, waist_q, joint_indices)
            lhs = jac @ jac.T + (damping * damping) * np.eye(3)
            try:
                jac_pinv = jac.T @ np.linalg.solve(lhs, np.eye(3))
            except np.linalg.LinAlgError:
                jac_pinv = jac.T @ np.linalg.pinv(lhs)

            dq = jac_pinv @ error
            nullspace = np.eye(len(q)) - jac_pinv @ jac
            dq += nullspace @ (-0.04 * (q - preferred_q))

            step_norm = float(np.linalg.norm(dq))
            if step_norm > 0.12:
                dq *= 0.12 / step_norm

            q_next = clamp_to_limits(q + dq, lower, upper)
            if float(np.linalg.norm(q_next - q)) < 1e-6:
                break
            q = q_next

    assert best_q is not None and best_reached is not None
    return IkResult(
        False,
        best_q,
        best_reached,
        best_residual,
        best_iterations,
        "IK did not reach tolerance",
    )


def validate_target(target: np.ndarray) -> None:
    if not np.all(np.isfinite(target)):
        raise ValueError("target contains non-finite values")
    if float(np.linalg.norm(target)) > 1.5:
        raise ValueError(
            "target is far outside the pelvis-frame working volume: "
            f"norm={float(np.linalg.norm(target)):.3f} m"
        )


def motor_to_model_q(motor_q: Sequence[float], signs: np.ndarray) -> np.ndarray:
    return np.asarray(motor_q, dtype=float) * signs[: len(motor_q)]


def model_to_motor_q(model_q: Sequence[float], signs: np.ndarray) -> np.ndarray:
    return np.asarray(model_q, dtype=float) * signs[: len(model_q)]


def smoothstep(ratio: float) -> float:
    ratio = min(max(ratio, 0.0), 1.0)
    return ratio * ratio * (3.0 - 2.0 * ratio)


class ArmSdkController:
    def __init__(self, network_interface: Optional[str], arm_dof: int, control_dt: float):
        self.network_interface = network_interface
        self.arm_dof = arm_dof
        self.control_dt = control_dt
        self.low_state: Optional[LowState_] = None
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.publisher: Optional[ChannelPublisher] = None
        self.subscriber: Optional[ChannelSubscriber] = None

    def init(self) -> None:
        if self.network_interface:
            ChannelFactoryInitialize(0, self.network_interface)
        else:
            ChannelFactoryInitialize(0)

        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._low_state_handler, 10)

    def _low_state_handler(self, msg: LowState_) -> None:
        self.low_state = msg

    def wait_for_low_state(self, timeout: float) -> LowState_:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.low_state is not None:
                return self.low_state
            time.sleep(0.02)
        raise TimeoutError("timed out waiting for rt/lowstate")

    def current_positions(self, joints: Iterable[int]) -> Dict[int, float]:
        state = self.wait_for_low_state(5.0)
        return {joint: float(state.motor_state[joint].q) for joint in joints}

    def write_arm_command(self, targets: Dict[int, float], kp: float, kd: float, enable_weight: float) -> None:
        assert self.publisher is not None
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = enable_weight
        for joint, q_des in targets.items():
            cmd = self.low_cmd.motor_cmd[joint]
            cmd.mode = 1
            cmd.tau = 0.0
            cmd.q = float(q_des)
            cmd.dq = 0.0
            cmd.kp = kp
            cmd.kd = kd
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def move_to(
        self,
        goal_positions: Dict[int, float],
        duration: float,
        hold_time: float,
        kp: float,
        kd: float,
        release: bool,
    ) -> None:
        controlled_joints = arm_sdk_joints(self.arm_dof)
        start_positions = self.current_positions(controlled_joints)
        all_goals = dict(start_positions)
        all_goals.update(goal_positions)

        start_time = time.monotonic()
        next_tick = start_time
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            ratio = smoothstep(elapsed / duration)
            targets = {
                joint: start_positions[joint] + ratio * (all_goals[joint] - start_positions[joint])
                for joint in controlled_joints
            }
            self.write_arm_command(targets, kp, kd, 1.0)
            if elapsed >= duration:
                break
            next_tick += self.control_dt
            time.sleep(max(0.0, next_tick - time.monotonic()))

        hold_deadline = time.monotonic() + hold_time
        while time.monotonic() < hold_deadline:
            self.write_arm_command(all_goals, kp, kd, 1.0)
            time.sleep(self.control_dt)

        if release:
            self.release(all_goals, kp, kd, release_time=1.0)

    def release(self, hold_positions: Dict[int, float], kp: float, kd: float, release_time: float) -> None:
        start_time = time.monotonic()
        while True:
            elapsed = time.monotonic() - start_time
            ratio = min(max(elapsed / release_time, 0.0), 1.0)
            self.write_arm_command(hold_positions, kp, kd, 1.0 - ratio)
            if ratio >= 1.0:
                break
            time.sleep(self.control_dt)


def format_joint_table(joints: Sequence[int], values: Sequence[float]) -> str:
    rows = []
    for joint, value in zip(joints, values):
        rows.append(f"  {JOINT_NAMES[joint]:>19s}: {value:+.4f} rad")
    return "\n".join(rows)


def format_joint_delta_table(
    joints: Sequence[int],
    current_values: Sequence[float],
    goal_values: Sequence[float],
) -> str:
    rows = []
    for joint, current, goal in zip(joints, current_values, goal_values):
        rows.append(
            f"  {JOINT_NAMES[joint]:>19s}: current={current:+.4f}, "
            f"goal={goal:+.4f}, delta={abs(goal - current):.4f} rad"
        )
    return "\n".join(rows)


def release_arm_sdk_safely(controller: ArmSdkController, arm_dof: int, kp: float, kd: float) -> None:
    try:
        hold_positions = controller.current_positions(arm_sdk_joints(arm_dof))
        controller.release(hold_positions, kp, kd, release_time=1.0)
        print("arm_sdk released.")
    except Exception as exc:
        print(f"Failed to release arm_sdk cleanly: {exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and optionally execute a right-hand navigation-frame target on G1 arm_sdk."
    )
    parser.add_argument("network_interface", nargs="?", help="DDS network interface, e.g. eth0")
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.30, -0.20, 0.10),
        help="Right hand target in --target-frame, meters. x forward, y left, z up for base_link/body.",
    )
    parser.add_argument(
        "--target-frame",
        choices=("base_link", "body", "pelvis", "map"),
        default="base_link",
        help="Frame of --target. Default base_link uses the navigation robot frame.",
    )
    parser.add_argument(
        "--base-link-in-pelvis-xyz",
        nargs=3,
        type=float,
        default=DEFAULT_BASE_LINK_IN_PELVIS_XYZ,
        metavar=("X", "Y", "Z"),
        help="Approximate navigation base_link origin expressed in URDF pelvis frame.",
    )
    parser.add_argument(
        "--base-link-in-pelvis-rpy",
        nargs=3,
        type=float,
        default=DEFAULT_BASE_LINK_IN_PELVIS_RPY,
        metavar=("R", "P", "Y"),
        help="Approximate base_link orientation in pelvis frame.",
    )
    parser.add_argument(
        "--joint-signs",
        nargs=7,
        type=float,
        default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        metavar=("SP", "SR", "SY", "EL", "WR", "WP", "WY"),
        help="Motor sign calibration for right arm joints: motor_q = sign * model_q.",
    )
    parser.add_argument("--arm-dof", type=int, choices=(5, 7), default=7, help="Use 5-DOF or 7-DOF G1 arm layout.")
    parser.add_argument("--tolerance", type=float, default=0.025, help="IK position tolerance in meters.")
    parser.add_argument("--execute", action="store_true", help="Actually send commands to rt/arm_sdk.")
    parser.add_argument("--no-prompt", action="store_true", help="Do not ask for confirmation before execution.")
    parser.add_argument("--duration", type=float, default=4.0, help="Minimum motion duration in seconds.")
    parser.add_argument("--max-joint-speed", type=float, default=0.45, help="Maximum planned joint speed in rad/s.")
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=2.80,
        help="Reject execution above this per-joint delta in rad.",
    )
    parser.add_argument("--hold", type=float, default=2.0, help="Hold target duration in seconds.")
    parser.add_argument("--kp", type=float, default=40.0, help="Joint position kp.")
    parser.add_argument("--kd", type=float, default=1.5, help="Joint position kd.")
    parser.add_argument("--keep-enabled", action="store_true", help="Deprecated; arm_sdk is always released on exit.")
    parser.add_argument("--allow-large-motion", action="store_true", help="Disable max-joint-delta rejection.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = np.asarray(args.target, dtype=float)
    signs = np.asarray(args.joint_signs, dtype=float)
    signs = np.where(signs >= 0.0, 1.0, -1.0)
    right_goal_joints = right_arm_joints(args.arm_dof)

    try:
        pelvis_target, base_link_target = target_to_pelvis_frame(
            target=target,
            target_frame=args.target_frame,
            base_link_in_pelvis_xyz=args.base_link_in_pelvis_xyz,
            base_link_in_pelvis_rpy=args.base_link_in_pelvis_rpy,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Refusing target: {exc}", file=sys.stderr)
        return 2

    try:
        validate_target(pelvis_target)
    except ValueError as exc:
        print(f"Refusing target: {exc}", file=sys.stderr)
        return 2

    controller: Optional[ArmSdkController] = None
    current_motor_q = None
    current_model_seed = None
    current_waist_q = np.zeros(3)

    if args.execute:
        controller = ArmSdkController(args.network_interface, args.arm_dof, control_dt=0.02)
        controller.init()
        controlled = arm_sdk_joints(args.arm_dof)
        current_positions = controller.current_positions(controlled)
        current_motor_q = np.array([current_positions[joint] for joint in right_goal_joints])
        current_model_seed = motor_to_model_q(current_motor_q, signs)
        current_waist_q = np.array(
            [
                current_positions[G1JointIndex.WaistYaw],
                current_positions[G1JointIndex.WaistRoll],
                current_positions[G1JointIndex.WaistPitch],
            ]
        )

    ik = solve_position_ik(
        target=pelvis_target,
        waist_q=current_waist_q,
        joint_indices=right_goal_joints,
        signs=signs,
        seed=current_model_seed,
        tolerance=args.tolerance,
    )
    motor_ik_q = model_to_motor_q(ik.q_model, signs)
    right_goal_q = motor_ik_q.copy()
    reached_pose = right_hand_pose(ik.q_model, current_waist_q, right_goal_joints)
    reached_rpy = matrix_to_rpy(reached_pose[:3, :3])

    print(f"Input target ({args.target_frame}):", np.array2string(target, precision=3))
    if base_link_target is not None:
        print("Base_link-frame target:", np.array2string(base_link_target, precision=3))
    print("Pelvis-frame IK target:", np.array2string(pelvis_target, precision=3))
    print("URDF reached position:", np.array2string(ik.reached, precision=3))
    print("URDF reached RPY:", np.array2string(reached_rpy, precision=3))
    print(f"IK residual: {ik.residual:.4f} m after {ik.iterations} iterations ({ik.message})")
    print("Right arm command:")
    print(format_joint_table(right_goal_joints, right_goal_q))

    if not ik.success:
        print("Refusing execution because IK did not meet tolerance.", file=sys.stderr)
        return 3

    for joint, value in zip(right_goal_joints, right_goal_q):
        if joint in JOINT_LIMITS:
            lo, hi = JOINT_LIMITS[joint]
            if value < lo or value > hi:
                print(
                    f"Refusing execution: {JOINT_NAMES[joint]}={value:.3f} outside [{lo:.3f}, {hi:.3f}]",
                    file=sys.stderr,
                )
                return 4

    if not args.execute:
        print("Dry run only. Add --execute to send commands to the robot.")
        return 0

    assert controller is not None and current_motor_q is not None
    deltas = np.abs(right_goal_q[: len(current_motor_q)] - current_motor_q)
    max_delta = float(np.max(deltas)) if len(deltas) else 0.0
    if max_delta > args.max_joint_delta and not args.allow_large_motion:
        print(
            f"Refusing execution: max joint delta {max_delta:.3f} rad exceeds "
            f"--max-joint-delta {args.max_joint_delta:.3f}.",
            file=sys.stderr,
        )
        print("Joint deltas:", file=sys.stderr)
        print(format_joint_delta_table(right_goal_joints, current_motor_q, right_goal_q), file=sys.stderr)
        print(
            "If the area is clear and this large move is intentional, rerun with "
            "--allow-large-motion or increase --max-joint-delta.",
            file=sys.stderr,
        )
        return 5

    duration = max(args.duration, max_delta / max(args.max_joint_speed, 1e-3))
    if not args.no_prompt:
        print("WARNING: robot arm will move. Ensure the area is clear and E-stop is available.")
        input("Press Enter to continue, or Ctrl-C to abort...")

    goal_positions = {joint: float(value) for joint, value in zip(right_goal_joints, right_goal_q)}
    exit_code = 0
    try:
        controller.move_to(
            goal_positions=goal_positions,
            duration=duration,
            hold_time=args.hold,
            kp=args.kp,
            kd=args.kd,
            release=False,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        exit_code = 130
    finally:
        release_arm_sdk_safely(controller, args.arm_dof, args.kp, args.kd)

    if exit_code == 0:
        print("Done.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
