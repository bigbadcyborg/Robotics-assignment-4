#!/usr/bin/env python3
"""Autonomous bottle pick-and-place for TurtleBot3 + OpenManipulator.

Builds on Assignment 3 teleop motion/manipulator primitives and
Assignment 4 YOLO JSON subscriber format.
"""

import json
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

# Motion limits (Burger)
BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

# Camera / detection assumptions
IMAGE_WIDTH = 1280.0
DESIRED_CENTER_X = IMAGE_WIDTH / 2.0
CENTER_TOLERANCE_PX = 40.0
DETECTION_TIMEOUT_SEC = 0.75

# Distance proxy using bbox area (in pixels^2)
GRASP_AREA_THRESHOLD = 110000.0
APPROACH_AREA_TARGET = 90000.0

# Tuned proportional gains
KP_ANG = 0.001
KP_LIN = 0.0015

# Arm/gripper presets from A3 sample workflow
ARM_POSE_HOME = [0.0, 0.0, 0.0, 0.0]
ARM_POSE_EXTEND_FORWARD = [0.0, -1.10, 0.75, 0.35]
GRIPPER_OPEN = 0.01
GRIPPER_CLOSE = -0.01


class TaskState(Enum):
    """High-level mission states for the autonomous routine."""

    # Rotate in place until a bottle detection is available.
    SEARCH = auto()
    # Keep bottle centered in camera before driving forward.
    ALIGN = auto()
    # Achieved stable center alignment lock for 3 seconds.
    LOCKED_ON = auto()
    # Drive while maintaining center alignment.
    APPROACH = auto()
    # Run arm/gripper sequence to grab bottle from ground.
    PICK = auto()
    # Reorient base toward drop location.
    TURN_TO_PLACE = auto()
    # Translate toward drop location.
    DRIVE_TO_PLACE = auto()
    # Run reverse arm/gripper sequence to release bottle.
    PLACE = auto()
    # Final idle state: stop base motion.
    DONE = auto()


@dataclass
class Detection:
    """Container for one YOLO detection entry already normalized to floats."""

    class_name: str
    confidence: float
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


class BottleTracker:
    """Keeps most relevant bottle detection parsed from YOLO JSON stream."""

    def __init__(self) -> None:
        self._best_detection: Optional[Detection] = None
        self._last_update_time: float = 0.0

    def update_from_json(self, payload: str, now_sec: float) -> None:
        # Payload format is the same JSON schema emitted by Part 2 publisher.
        data = json.loads(payload)
        detections = data.get("detections", [])

        # Filter to target class only; Part 3 objective is bottle pickup.
        bottle_candidates: List[Detection] = []
        for det in detections:
            class_name = str(det.get("class_name", "")).lower()
            if class_name != "bottle":
                continue
            bbox = det.get("bbox", {})
            bottle_candidates.append(
                Detection(
                    class_name=class_name,
                    confidence=float(det.get("confidence", 0.0)),
                    cx=float(bbox.get("cx", 0.0)),
                    cy=float(bbox.get("cy", 0.0)),
                    w=float(bbox.get("w", 0.0)),
                    h=float(bbox.get("h", 0.0)),
                )
            )

        self._best_detection = max(
            bottle_candidates,
            # Prefer highest confidence and use area as a tie breaker.
            key=lambda d: (d.confidence, d.area),
            default=None,
        )
        # Timestamp of last message (even if no bottle), used for stale-data checks.
        self._last_update_time = now_sec

    def get_fresh_bottle(self, now_sec: float) -> Optional[Detection]:
        # Avoid acting on stale detections when frame stream pauses or lags.
        if (now_sec - self._last_update_time) > DETECTION_TIMEOUT_SEC:
            return None
        return self._best_detection


class ManipulatorClient:
    """Encapsulates arm and gripper action interactions."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self._arm_client = ActionClient(
            node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self._gripper_client = ActionClient(
            node, GripperCommand, "/gripper_controller/gripper_cmd"
        )
        self._arm_joint_names = ["joint1", "joint2", "joint3", "joint4"]

    def send_arm_goal(self, positions: List[float], duration_sec: float = 2.0) -> bool:
        # Defensive check keeps control loop responsive when arm node is absent.
        if not self._arm_client.wait_for_server(timeout_sec=2.0):
            self._node.get_logger().warn("Arm action server unavailable")
            return False

        # Single-point trajectory for simple pose-to-pose commands.
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self._arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1.0) * 1e9),
        )
        goal_msg.trajectory.points.append(point)
        self._arm_client.send_goal_async(goal_msg)
        return True

    def send_gripper_goal(self, position: float) -> bool:
        # Same availability guard for the gripper action server.
        if not self._gripper_client.wait_for_server(timeout_sec=2.0):
            self._node.get_logger().warn("Gripper action server unavailable")
            return False

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 1.0
        self._gripper_client.send_goal_async(goal_msg)
        return True


class AutonomousBottlePickPlace(Node):
    """Coordinates perception, base control, and manipulator actions."""

    def __init__(self) -> None:
        super().__init__("autonomous_bottle_pick_place")
        # Publisher for differential-drive base control.
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Subscriber to YOLO JSON detections produced by Part 2 publisher.
        self._sub = self.create_subscription(
            String, "/yolo/detections_json", self._on_detections, 10
        )

        self._tracker = BottleTracker()
        self._manipulator = ManipulatorClient(self)

        self._state = TaskState.SEARCH
        self._state_start_time = self._now_sec()
        self._align_centered_start_time = None
        self._pick_started = False
        self._place_started = False

        self._control_timer = self.create_timer(0.05, self._control_loop)
        self.get_logger().info("Autonomous part 3 node started.")

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_detections(self, msg: String) -> None:
        # Keep callback focused on perception ingestion only (SRP).
        try:
            self._tracker.update_from_json(msg.data, self._now_sec())
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid JSON from YOLO publisher: {exc}")

    def _publish_twist(self, linear_x: float, angular_z: float) -> None:
        # Clamp commands to TurtleBot3 Burger limits for safety.
        twist = Twist()
        twist.linear.x = max(-BURGER_MAX_LIN_VEL, min(BURGER_MAX_LIN_VEL, linear_x))
        twist.angular.z = max(-BURGER_MAX_ANG_VEL, min(BURGER_MAX_ANG_VEL, angular_z))
        self._cmd_vel_pub.publish(twist)

    def _transition(self, new_state: TaskState) -> None:
        # Centralizing transition logic keeps timing code consistent.
        self._state = new_state
        self._state_start_time = self._now_sec()
        self.get_logger().info(f"Transition -> {new_state.name}")

    def _control_loop(self) -> None:
        # Run fixed-frequency controller (20 Hz).
        now = self._now_sec()
        det = self._tracker.get_fresh_bottle(now)

        if self._state == TaskState.SEARCH:
            # Spin in place to sweep camera FOV until bottle appears.
            if det is None:
                self._publish_twist(0.0, 0.35)
                return
            self._transition(TaskState.ALIGN)

        if self._state == TaskState.ALIGN:
            # If we lose the target, restart search behavior.
            if det is None:
                self._align_centered_start_time = None
                self._transition(TaskState.SEARCH)
                return
            # Horizontal pixel error drives proportional angular correction.
            error_x = det.cx - DESIRED_CENTER_X
            if math.fabs(error_x) <= CENTER_TOLERANCE_PX:
                if self._align_centered_start_time is None:
                    self._align_centered_start_time = now
                elif now - self._align_centered_start_time >= 3.0:
                    self._publish_twist(0.0, 0.0)
                    self._align_centered_start_time = None
                    self._transition(TaskState.LOCKED_ON)
                    return
                # Stop rotating while in the center tolerance to build up the 3 seconds
                self._publish_twist(0.0, 0.0)
                return
            else:
                self._align_centered_start_time = None
                self._publish_twist(0.0, -KP_ANG * error_x)
            return

        if self._state == TaskState.LOCKED_ON:
            # Small delay or logic before immediately going to approach could go here.
            self._transition(TaskState.APPROACH)
            return

        if self._state == TaskState.APPROACH:
            if det is None:
                self._transition(TaskState.SEARCH)
                return

            # Steering term keeps bottle centered while moving.
            error_x = det.cx - DESIRED_CENTER_X
            ang_cmd = -KP_ANG * error_x
            # Area is a depth proxy: bigger box means object is closer.
            area_error = APPROACH_AREA_TARGET - det.area
            lin_cmd = KP_LIN * area_error

            # Stop base once bottle is in grasping distance range.
            if det.area >= GRASP_AREA_THRESHOLD:
                self._publish_twist(0.0, 0.0)
                self._transition(TaskState.PICK)
                return

            self._publish_twist(lin_cmd, ang_cmd)
            return

        if self._state == TaskState.PICK:
            # Keep base stationary during arm operations.
            self._publish_twist(0.0, 0.0)
            elapsed = now - self._state_start_time
            if not self._pick_started:
                self._pick_started = True
                # 1) Home + open to start from a deterministic grasp state.
                self._manipulator.send_arm_goal(ARM_POSE_HOME, duration_sec=1.5)
                self._manipulator.send_gripper_goal(GRIPPER_OPEN)
            elif elapsed > 2.0 and elapsed <= 4.5:
                # 2) Extend arm toward bottle.
                self._manipulator.send_arm_goal(ARM_POSE_EXTEND_FORWARD, duration_sec=2.0)
            elif elapsed > 4.5 and elapsed <= 5.8:
                # 3) Close gripper to grasp.
                self._manipulator.send_gripper_goal(GRIPPER_CLOSE)
            elif elapsed > 5.8 and elapsed <= 8.2:
                # 4) Retract to safer carrying pose.
                self._manipulator.send_arm_goal(ARM_POSE_HOME, duration_sec=2.0)
            elif elapsed > 8.2:
                self._transition(TaskState.TURN_TO_PLACE)
            return

        if self._state == TaskState.TURN_TO_PLACE:
            # Timed in-place rotation to face nominal drop zone.
            elapsed = now - self._state_start_time
            if elapsed < 3.2:
                self._publish_twist(0.0, 0.6)
            else:
                self._publish_twist(0.0, 0.0)
                self._transition(TaskState.DRIVE_TO_PLACE)
            return

        if self._state == TaskState.DRIVE_TO_PLACE:
            # Timed forward motion to create transport phase.
            elapsed = now - self._state_start_time
            if elapsed < 1.8:
                self._publish_twist(0.09, 0.0)
            else:
                self._publish_twist(0.0, 0.0)
                self._transition(TaskState.PLACE)
            return

        if self._state == TaskState.PLACE:
            self._publish_twist(0.0, 0.0)
            elapsed = now - self._state_start_time
            if not self._place_started:
                self._place_started = True
                # 1) Extend toward floor.
                self._manipulator.send_arm_goal(ARM_POSE_EXTEND_FORWARD, duration_sec=2.0)
            elif elapsed > 2.4 and elapsed <= 3.8:
                # 2) Open to release object.
                self._manipulator.send_gripper_goal(GRIPPER_OPEN)
            elif elapsed > 3.8 and elapsed <= 6.0:
                # 3) Return home.
                self._manipulator.send_arm_goal(ARM_POSE_HOME, duration_sec=2.0)
            elif elapsed > 6.0:
                self._transition(TaskState.DONE)
            return

        if self._state == TaskState.DONE:
            # Hold safe zero velocity when mission is complete.
            self._publish_twist(0.0, 0.0)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AutonomousBottlePickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_twist(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
