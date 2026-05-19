import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import threading
import time
import sys
import select
import tty
import termios
import csv
from datetime import datetime


class HexapodController(Node):

    def __init__(self, enable_logging=False):
        super().__init__('hexapod_controller')

        self.ns = 'hexapod_0'
        self.Ct = 6.12

        # === Motion ===
        self.step_size = 6.28
        self.key_timeout = 1.0

        # === Wave gait params ===
        self.pairs = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        self.phase_active = [False] * 3
        self.phase_start_pos = [[0.0]*6 for _ in range(3)]

        self.phase_trigger = 5.0   # когда запускать следующую фазу (~60°)
        self.reach_eps = 0.3       # когда считаем, что достигли цели

        # === Logging ===
        self.enable_logging = enable_logging
        self.log_interval = 0.1
        self.last_log_time = 0.0

        # === Base pose ===
        self.base_offset = -0.05
        self.base_positions = [self.base_offset] * 6
        self.relative_targets = [0.0] * 6
        self.target_positions = self.base_positions[:]

        # === Command ===
        self.active_command = None
        self.last_key_time = 0.0

        # === Publisher ===
        self.publisher = self.create_publisher(
            Float64MultiArray,
            f'{self.ns}/set_position_targets',
            10
        )

        # === Subscribers ===
        self.create_subscription(
            JointState,
            f'{self.ns}/joint_states',
            self.joint_callback,
            10
        )

        self.create_subscription(
            Float64MultiArray,
            f'{self.ns}/dataset',
            self.dataset_callback,
            10
        )

        # === CSV ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_name = f"{self.ns}_{timestamp}.csv"

        headers = (
            [f'angle_{i}' for i in range(1, 7)] +
            [f'speed_{i}' for i in range(1, 7)] +
            [f'curr_{i}' for i in range(1, 7)] +
            ['x', 'y', 'phi']
        )

        if self.enable_logging:
            with open(self.csv_name, 'w', newline='') as f:
                csv.writer(f).writerow(headers)

        # === Keyboard ===
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        threading.Thread(
            target=self.keyboard_listener,
            daemon=True
        ).start()

        self.get_logger().info("Hexapod controller started (wave gait)")

        self.publish_targets()

    # =========================
    # KEYBOARD
    # =========================

    def keyboard_listener(self):
        while rclpy.ok():

            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1)

                if key in ['w', 'a', 's', 'd']:
                    self.active_command = key
                    self.last_key_time = time.time()

                elif key == 'q':
                    rclpy.shutdown()
                    return

            time.sleep(0.02)

    # =========================
    # MAIN CONTROL (WAVE GAIT)
    # =========================

    def joint_callback(self, msg):

        if len(msg.position) < 6:
            return

        pos = msg.position[:6]

        # timeout
        if self.active_command:
            if time.time() - self.last_key_time > self.key_timeout:
                self.active_command = None
                self.reset_gait()
                return

        if not self.active_command:
            return

        signs = self.get_signs(self.active_command)

        # старт первой фазы
        if not any(self.phase_active):
            self.start_phase(0, signs, pos)
            return

        for i in range(3):

            if not self.phase_active[i]:
                continue

            pair = self.pairs[i]

            # === запуск следующей фазы (циклический!) ===
            next_i = (i + 1) % 3

            if not self.phase_active[next_i]:
                if self.almost_done(pair, pos):
                    self.start_phase(next_i, signs, pos)

            # === завершение фазы ===
            if self.reached(pair, pos):
                self.phase_active[i] = False
    # =========================
    # GAIT LOGIC
    # =========================

    def get_signs(self, cmd):

        if cmd == 'w':
            return [1, 1, 1, 1, 1, 1]

        elif cmd == 's':
            return [-1, -1, -1, -1, -1, -1]

        elif cmd == 'd':
            return [1, -1, 1, -1, 1, -1]

        elif cmd == 'a':
            return [-1, 1, -1, 1, -1, 1]

    def start_phase(self, idx, signs, pos):

        pair = self.pairs[idx]

        for i in pair:
            self.relative_targets[i] += signs[i] * self.step_size

        self.phase_active[idx] = True
        self.phase_start_pos[idx] = pos[:]

        self.publish_targets()

    def almost_done(self, pair, pos):

        for i in pair:
            if abs(self.target_positions[i] - pos[i]) > self.phase_trigger:
                return False

        return True

    def reached(self, pair, pos):

        for i in pair:
            if abs(self.target_positions[i] - pos[i]) > self.reach_eps:
                return False

        return True

    def reset_gait(self):
        self.phase_active = [False] * 3

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
    # DATASET
    # =========================

    def dataset_callback(self, msg):

        if not self.enable_logging:
            return

        now = time.time()
        if now - self.last_log_time < self.log_interval:
            return

        data = msg.data
        if len(data) < 21:
            return

        angles = data[0:6]
        speeds = data[6:12]
        currents = [tau / self.Ct for tau in data[12:18]]

        x, y, phi = data[-3:]

        with open(self.csv_name, 'a', newline='') as f:
            csv.writer(f).writerow(
                list(angles) + list(speeds) + list(currents) + [x, y, phi]
            )

        self.last_log_time = now

    # =========================

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        super().destroy_node()


# =============================

def main():
    rclpy.init()
    node = HexapodController(enable_logging=False)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()