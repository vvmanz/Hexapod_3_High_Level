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
        self.key_timeout = 1.0

        # ===== Диагональные паттерны =====
        self.patterns = [
            [[0, 3], [2, 5], [1, 4]],
            [[0, 5], [1, 2], [3, 4]]
        ]

        self.current_pattern = 0
        self.phase_index = 0

        self.phase_active = False
        self.phase_start_pos = [0.0] * 6

        self.phase_trigger = 5.0
        self.reach_eps = 0.3

        # ===== Base =====
        self.base_offset = -0.05
        self.base_positions = [self.base_offset] * 6
        self.relative_targets = [0.0] * 6
        self.target_positions = self.base_positions[:]

        # ===== Command =====
        self.active_command = None
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
            "Hexapod controller started (diagonal wave, keyboard). "
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
                            self.reset_gait()
                        self.logger.stop()

            except Exception as e:
                self.get_logger().error(f"Keyboard error: {e}")
                break

    # =========================
    # MAIN CONTROL
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
                    self.reset_gait()
                self.logger.stop()
                return

        if not cmd:
            return

        signs = self.get_signs(cmd)

        if not self.phase_active:
            self.start_phase(pos, signs)
            return

        pair = self.get_current_pair()

        if self.almost_done(pair, pos):
            self.next_phase()
            self.start_phase(pos, signs)

        if self.reached(pair, pos):
            self.phase_active = False

    # =========================
    # LOGIC
    # =========================

    def get_current_pair(self):
        return self.patterns[self.current_pattern][self.phase_index]

    def next_phase(self):
        self.phase_index += 1

        if self.phase_index >= 3:
            self.phase_index = 0
            self.current_pattern = 1 - self.current_pattern

    def get_signs(self, cmd):
        if cmd == 'w':
            return [1] * 6
        elif cmd == 's':
            return [-1] * 6
        elif cmd == 'd':
            return [1, -1, 1, -1, 1, -1]
        elif cmd == 'a':
            return [-1, 1, -1, 1, -1, 1]

    def start_phase(self, pos, signs):

        pair = self.get_current_pair()

        for i in pair:
            self.relative_targets[i] += signs[i] * self.step_size

        self.phase_active = True
        self.phase_start_pos = pos[:]

        self.publish_targets()

    def almost_done(self, pair, pos):
        return all(abs(self.target_positions[i] - pos[i]) <= self.phase_trigger for i in pair)

    def reached(self, pair, pos):
        return all(abs(self.target_positions[i] - pos[i]) <= self.reach_eps for i in pair)

    def reset_gait(self):
        self.phase_active = False
        self.phase_index = 0

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