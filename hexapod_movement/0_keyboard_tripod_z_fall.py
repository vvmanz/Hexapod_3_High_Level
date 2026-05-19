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

        # ===== Tripod groups =====
        self.group1 = [0, 3, 4]
        self.group2 = [1, 2, 5]

        # ===== Thresholds =====
        self.thresholds = {
            'w': 1.5,
            's': 1.5,
            'a': 1.25,
            'd': 1.25
        }

        # ===== Emergency / pose presets =====
        self.fall_pose = [
            -1.57079632679,   #  90°
            -1.57079632679,   #  90°
            -1.57079632679,   #  90°
            1.57079632679,  # -90°
            1.57079632679,  # -90°
            1.57079632679,  # -90°
        ]

        # ===== FSM =====
        self.active_command = None
        self.step_phase = 'IDLE'
        self.active_group = None
        self.pos_at_start = [0.0] * 6
        self.last_key_time = 0.0

        # ===== Thread safety =====
        self.state_lock = threading.Lock()
        self._running = True

        # ===== Publisher =====
        self.publisher = self.create_publisher(
            Float64MultiArray,
            f'/{self.ns}/set_position_targets',
            10
        )

        # ===== Subscribers =====
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
            "Hexapod controller started (tripod gait, keyboard control). "
            "Use W/A/S/D in terminal, SPACE or X to stop, Z for fall pose."
        )

        # Сохраняем положение терминала и переводим его в режим чтения символов
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
            self.get_logger().warn(
                "stdin is not a TTY. Keyboard control is unavailable."
            )

        self.publish_targets()

    # =========================
    # KEYBOARD INPUT
    # =========================

    def _keyboard_loop(self):
        while self._running and rclpy.ok():
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)

                    if not ch:
                        continue

                    ch = ch.lower()

                    if ch in ('w', 'a', 's', 'd'):
                        with self.state_lock:
                            self.active_command = ch
                            self.last_key_time = time.time()
                        self.logger.start()

                    elif ch == 'z':
                        self.set_fall_pose()

                    elif ch in (' ', 'x', '\x03'):
                        with self.state_lock:
                            self.active_command = None
                            self.step_phase = 'IDLE'
                        self.logger.stop()

            except Exception as e:
                self.get_logger().error(f"Keyboard loop error: {e}")
                break

    # =========================
    # POSES
    # =========================

    def set_fall_pose(self):
        with self.state_lock:
            self.active_command = None
            self.step_phase = 'IDLE'
            self.active_group = None
            self.pos_at_start = [0.0] * 6
            self.last_key_time = 0.0
            self.relative_targets = [
                self.fall_pose[i] - self.base_positions[i]
                for i in range(6)
            ]
        self.logger.stop()
        self.publish_targets()

    # =========================
    # FSM
    # =========================

    def joint_callback(self, msg):
        if len(msg.position) < 6:
            return

        pos = msg.position[:6]

        with self.state_lock:
            active_command = self.active_command
            last_key_time = self.last_key_time
            step_phase = self.step_phase
            active_group = self.active_group
            pos_at_start = self.pos_at_start[:]

        # timeout
        if active_command:
            if time.time() - last_key_time > self.key_timeout:
                with self.state_lock:
                    self.active_command = None
                    self.step_phase = 'IDLE'
                self.logger.stop()
                return

        if not active_command:
            return

        threshold = self.thresholds[active_command]

        if active_command == 'w':
            inc1, inc2 = +self.step_size, +self.step_size
            first = 'GROUP1'

        elif active_command == 's':
            inc1, inc2 = -self.step_size, -self.step_size
            first = 'GROUP1'

        elif active_command == 'a':
            inc1, inc2 = -self.step_size, +self.step_size
            first = 'GROUP1'

        elif active_command == 'd':
            inc1, inc2 = +self.step_size, -self.step_size
            first = 'GROUP2'

        else:
            return

        if step_phase == 'IDLE':
            self.start_step(first, inc1, inc2, pos)
            return

        if step_phase == 'WAIT_MOTION':
            if self.motion_started(pos, pos_at_start):
                with self.state_lock:
                    self.step_phase = active_group
            return

        if step_phase == 'GROUP1':
            if self.reached(self.group1, pos, threshold):
                self.start_step('GROUP2', inc1, inc2, pos)

        elif step_phase == 'GROUP2':
            if self.reached(self.group2, pos, threshold):
                self.start_step('GROUP1', inc1, inc2, pos)

    def start_step(self, group, inc1, inc2, pos):
        if group == 'GROUP1':
            self.apply_increment(self.group1, inc1)
        else:
            self.apply_increment(self.group2, inc2)

        with self.state_lock:
            self.pos_at_start = pos[:]
            self.active_group = group
            self.step_phase = 'WAIT_MOTION'

        self.publish_targets()

    def motion_started(self, pos, pos_at_start):
        return any(abs(pos[i] - pos_at_start[i]) > self.motion_eps for i in range(6))

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
    # CLEANUP
    # =========================

    def destroy_node(self):
        self._running = False
        self.logger.stop()

        if self._stdin_is_tty and self._old_term_settings is not None:
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