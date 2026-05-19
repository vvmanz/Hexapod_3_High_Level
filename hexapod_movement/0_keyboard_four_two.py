#!/usr/bin/env python3
import sys
import time
import threading
import select
import termios
import tty

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

from hexapod_logger import HexapodLogger


class HexapodController(Node):

    def __init__(self):
        super().__init__('hexapod_controller')

        self.ns = 'hexapod_0'

        # ===== Logger =====
        self.logger = HexapodLogger(
            namespace=self.ns,
            Ct=6.12,
            log_interval=0.01,
            spawn_x=0.0,
            spawn_y=0.0
        )

        # ===== Motion =====
        self.step_size = 6.28
        self.motion_eps = 0.01
        self.key_timeout = 1.0

        # ===== Base pose =====
        self.base_offset = -0.05
        self.base_positions = [self.base_offset] * 6
        self.relative_targets = [0.0] * 6
        self.target_positions = self.base_positions[:]

        # ===== Groups =====
        self.groupA = [0, 3, 5]
        self.groupB = [1, 2, 4]

        # ===== Thresholds =====
        self.thresholds = {
            'w': 2.6,
            's': 2.2,
            'a': 2.65,
            'd': 2.65
        }

        # ===== FSM =====
        self.active_command = None
        self.step_phase = 'IDLE'
        self.active_group = None
        self.pos_at_start = [0.0] * 6
        self.last_key_time = 0.0

        # ===== Thread safety =====
        self.state_lock = threading.Lock()
        self._running = True

        # ===== ROS =====
        self.publisher = self.create_publisher(
            Float64MultiArray,
            f'/{self.ns}/set_position_targets',
            10
        )

        self.create_subscription(
            JointState,
            f'/{self.ns}/joint_states',
            self.joint_callback,
            10
        )

        self.create_subscription(
            Float64MultiArray,
            f'/{self.ns}/dataset',
            self.dataset_callback,
            10
        )

        self.create_subscription(
            Odometry,
            f'/{self.ns}/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info(
            "Hexapod controller started (NEW GAIT, keyboard control). "
            "Use W/A/S/D, SPACE or X to stop."
        )

        # ===== Keyboard =====
        self._stdin_is_tty = sys.stdin.isatty()
        self._old_term_settings = None

        if self._stdin_is_tty:
            self._old_term_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

            self._keyboard_thread = threading.Thread(
                target=self._keyboard_loop,
                daemon=True
            )
            self._keyboard_thread.start()
        else:
            self.get_logger().warn("stdin is not TTY — keyboard disabled")

        self.publish_targets()

    # =========================
    # KEYBOARD INPUT
    # =========================

    def _keyboard_loop(self):
        while self._running and rclpy.ok():
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1).lower()

                    if ch in ('w', 'a', 's', 'd'):
                        with self.state_lock:
                            self.active_command = ch
                            self.last_key_time = time.time()
                        self.logger.start()

                    elif ch in (' ', 'x', '\x03'):
                        with self.state_lock:
                            self.active_command = None
                            self.step_phase = 'IDLE'
                        self.logger.stop()

            except Exception as e:
                self.get_logger().error(f"Keyboard error: {e}")
                break

    # =========================
    # FSM
    # =========================

    def joint_callback(self, msg):

        if len(msg.position) < 6:
            return

        pos = msg.position[:6]

        with self.state_lock:
            cmd = self.active_command
            last_time = self.last_key_time

        if cmd:
            if time.time() - last_time > self.key_timeout:
                with self.state_lock:
                    self.active_command = None
                    self.step_phase = 'IDLE'
                self.logger.stop()
                return

        if not cmd:
            return

        threshold = self.thresholds[cmd]

        # === направления ===

        if cmd == 'w':
            incA = +self.step_size
            incB = +self.step_size

        elif cmd == 's':
            incA = -self.step_size
            incB = -self.step_size

        elif cmd == 'd':
            incA = -self.step_size
            incB = +self.step_size

        elif cmd == 'a':
            incA = +self.step_size
            incB = -self.step_size

        # === FSM ===

        if self.step_phase == 'IDLE':
            self.start_step('A', incA, incB, pos)
            return

        if self.step_phase == 'WAIT_MOTION':
            if self.motion_started(pos):
                with self.state_lock:
                    self.step_phase = self.active_group
            return

        if self.step_phase == 'A':
            if self.reached(self.groupA, pos, threshold):
                self.start_step('B', incA, incB, pos)

        elif self.step_phase == 'B':
            if self.reached(self.groupB, pos, threshold):
                self.start_step('A', incA, incB, pos)

    def start_step(self, group, incA, incB, pos):

        if group == 'A':
            self.apply_increment(self.groupA, incA)
        else:
            self.apply_increment(self.groupB, incB)

        with self.state_lock:
            self.pos_at_start = pos[:]
            self.active_group = group
            self.step_phase = 'WAIT_MOTION'

        self.publish_targets()

    def motion_started(self, pos):
        return any(abs(pos[i] - self.pos_at_start[i]) > self.motion_eps for i in range(6))

    def reached(self, group, pos, threshold):
        return all(abs(self.target_positions[i] - pos[i]) <= threshold for i in group)

    def apply_increment(self, group, inc):
        for i in group:
            self.relative_targets[i] += inc

    # =========================
    # TARGETS
    # =========================

    def compute_targets(self):
        self.target_positions = [
            self.base_positions[i] + self.relative_targets[i]
            for i in range(6)
        ]

    def publish_targets(self):
        self.compute_targets()
        msg = Float64MultiArray()
        msg.data = self.target_positions
        self.publisher.publish(msg)

    # =========================
    # LOGGER
    # =========================

    def dataset_callback(self, msg):
        self.logger.set_dataset(msg)

    def odom_callback(self, msg):
        self.logger.set_odom(msg)

    # =========================

    def destroy_node(self):
        self._running = False
        self.logger.stop()

        if self._stdin_is_tty and self._old_term_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term_settings)
            except Exception:
                pass

        super().destroy_node()


def main():
    rclpy.init()
    node = HexapodController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()