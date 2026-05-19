#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous discrete navigator for ROS 2 / Gazebo.

What this node does:
- Subscribes to odometry from /hexapod_0/odom
- Generates a waypoint trajectory (square / circle / polygon / figure8)
- Chooses one of four discrete commands: w / a / s / d
- Publishes the command as std_msgs/String to /hexapod_0/command

This file is intentionally controller-agnostic:
any gait / locomotion controller can subscribe to /<ns>/command
and interpret the same four commands in its own way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String


Point = Tuple[float, float]


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # Standard yaw extraction from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class TrajectoryConfig:
    kind: str = "square"          # square / circle / polygon / figure8
    side: float = 12.0              # square side length
    radius: float = 4.0            # circle / polygon radius
    points_per_edge: int = 8      # square / polygon sampling density
    points_count: int = 32        # circle / figure8 sampling density
    polygon_sides: int = 5    # for regular polygon
    loop: bool = True


class TrajectoryLibrary:
    @staticmethod
    def square(center: Point, side: float, samples_per_edge: int = 12) -> List[Point]:
        cx, cy = center
        h = side / 2.0
        corners = [
            (cx - h, cy - h),
            (cx + h, cy - h),
            (cx + h, cy + h),
            (cx - h, cy + h),
        ]
        pts: List[Point] = []
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            for k in range(samples_per_edge):
                t = k / float(samples_per_edge)
                pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        return pts

    @staticmethod
    def circle(center: Point, radius: float, count: int = 60) -> List[Point]:
        cx, cy = center
        pts: List[Point] = []
        for i in range(count):
            ang = 2.0 * math.pi * i / float(count)
            pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
        return pts

    @staticmethod
    def polygon(center: Point, radius: float, sides: int = 6, samples_per_edge: int = 12) -> List[Point]:
        if sides < 3:
            raise ValueError("polygon_sides must be >= 3")
        cx, cy = center
        verts: List[Point] = []
        for i in range(sides):
            ang = 2.0 * math.pi * i / float(sides)
            verts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))

        pts: List[Point] = []
        for i in range(sides):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % sides]
            for k in range(samples_per_edge):
                t = k / float(samples_per_edge)
                pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        return pts

    @staticmethod
    def figure8(center: Point, radius: float, count: int = 80) -> List[Point]:
        # Lemniscate-like figure eight
        cx, cy = center
        pts: List[Point] = []
        for i in range(count):
            t = 2.0 * math.pi * i / float(count)
            x = radius * math.sin(t)
            y = radius * math.sin(t) * math.cos(t)
            pts.append((cx + x, cy + y))
        return pts

    @staticmethod
    def build(cfg: TrajectoryConfig, center: Point) -> List[Point]:
        kind = cfg.kind.lower().strip()
        if kind == "square":
            return TrajectoryLibrary.square(center, cfg.side, cfg.points_per_edge)
        if kind == "circle":
            return TrajectoryLibrary.circle(center, cfg.radius, cfg.points_count)
        if kind == "polygon":
            return TrajectoryLibrary.polygon(center, cfg.radius, cfg.polygon_sides, cfg.points_per_edge)
        if kind in ("8", "figure8", "figure_8", "lemniscate"):
            return TrajectoryLibrary.figure8(center, cfg.radius, cfg.points_count)
        raise ValueError(f"Unknown trajectory kind: {cfg.kind}")


class DiscreteCommandPolicy:
    """Maps robot pose + target waypoint into one of w/a/s/d."""

    def __init__(self, turn_deadband: float = 0.35, reverse_threshold: float = 2.25):
        self.turn_deadband = turn_deadband
        self.reverse_threshold = reverse_threshold

    def decide(self, x: float, y: float, yaw: float, target: Point) -> str:
        tx, ty = target
        dx = tx - x
        dy = ty - y
        desired = math.atan2(dy, dx)
        err = wrap_to_pi(desired - yaw)
        dist = math.hypot(dx, dy)

        # Far behind: reverse can be more natural than doing a long turn.
        if dist > 0.25 and abs(err) > self.reverse_threshold:
            return "s"

        if err > self.turn_deadband:
            return "a"
        if err < -self.turn_deadband:
            return "d"
        return "w"


class AutonomousDiscreteNavigator(Node):
    def __init__(self):
        super().__init__("autonomous_discrete_navigator")

        # Parameters
        self.declare_parameter("namespace", "hexapod_0")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("command_topic", "command")
        self.declare_parameter("trajectory_kind", "square")
        self.declare_parameter("loop", True)
        self.declare_parameter("waypoint_tolerance", 0.18)
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("trajectory_center_mode", "start")  # start / fixed
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("square_side", 12.0)
        self.declare_parameter("radius", 4.0)
        self.declare_parameter("points_per_edge", 8)
        self.declare_parameter("points_count", 32)
        self.declare_parameter("polygon_sides", 5)
        self.declare_parameter("publish_stop_when_done", True)

        self.ns = self.get_parameter("namespace").value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_topic = self.get_parameter("command_topic").value

        self.tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self.publish_stop_when_done = bool(self.get_parameter("publish_stop_when_done").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)

        cfg = TrajectoryConfig(
            kind=str(self.get_parameter("trajectory_kind").value),
            side=float(self.get_parameter("square_side").value),
            radius=float(self.get_parameter("radius").value),
            points_per_edge=int(self.get_parameter("points_per_edge").value),
            points_count=int(self.get_parameter("points_count").value),
            polygon_sides=int(self.get_parameter("polygon_sides").value),
            loop=bool(self.get_parameter("loop").value),
        )
        self.trajectory_cfg = cfg
        self.trajectory_center_mode = str(self.get_parameter("trajectory_center_mode").value)
        self.fixed_center = (
            float(self.get_parameter("center_x").value),
            float(self.get_parameter("center_y").value),
        )

        self.publisher = self.create_publisher(String, f"/{self.ns}/{cmd_topic}", 10)
        self.create_subscription(Odometry, f"/{self.ns}/{odom_topic}", self.odom_callback, 10)

        self.policy = DiscreteCommandPolicy()

        self.last_odom: Optional[Odometry] = None
        self.have_trajectory = False
        self.waypoints: List[Point] = []
        self.current_idx = 0
        self.last_command: Optional[str] = None

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.control_step)

        self.get_logger().info(
            f"Navigator started: ns=/{self.ns}, odom=/{self.ns}/{odom_topic}, cmd=/{self.ns}/{cmd_topic}, trajectory={cfg.kind}"
        )

    def odom_callback(self, msg: Odometry):
        self.last_odom = msg

        if not self.have_trajectory:
            if self.trajectory_center_mode == "fixed":
                center = self.fixed_center
            else:
                p = msg.pose.pose.position
                center = (float(p.x), float(p.y))

            try:
                self.waypoints = TrajectoryLibrary.build(self.trajectory_cfg, center)
            except Exception as exc:
                self.get_logger().error(f"Failed to build trajectory: {exc}")
                return

            self.current_idx = 0
            self.have_trajectory = True
            self.get_logger().info(f"Trajectory generated with {len(self.waypoints)} points")

    def control_step(self):
        if self.last_odom is None or not self.have_trajectory or not self.waypoints:
            return

        pose = self.last_odom.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        q = pose.orientation
        yaw = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))

        target = self.waypoints[self.current_idx]
        dx = target[0] - x
        dy = target[1] - y
        dist = math.hypot(dx, dy)

        if dist <= self.tolerance:
            self.current_idx += 1
            if self.current_idx >= len(self.waypoints):
                if self.trajectory_cfg.loop:
                    self.current_idx = 0
                else:
                    if self.publish_stop_when_done:
                        self.publish_command(None)
                    self.get_logger().info("Trajectory completed")
                    self.have_trajectory = False
                    return
            target = self.waypoints[self.current_idx]

        cmd = self.policy.decide(x, y, yaw, target)
        self.publish_command(cmd)

    def publish_command(self, cmd: Optional[str]):
        if cmd == self.last_command:
            return

        msg = String()
        msg.data = "" if cmd is None else cmd
        self.publisher.publish(msg)
        self.last_command = cmd

        if cmd is None:
            self.get_logger().info("Published STOP")
        else:
            self.get_logger().debug(f"Published command: {cmd}")


def main():
    rclpy.init()
    node = AutonomousDiscreteNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
