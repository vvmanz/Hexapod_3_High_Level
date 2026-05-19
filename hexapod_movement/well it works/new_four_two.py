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
        self.step_size = 6.28  # полный оборот
        self.motion_eps = 0.01
        self.key_timeout = 1.0

        # === Logging ===
        self.enable_logging = enable_logging
        self.log_interval = 0.1
        self.last_log_time = 0.0

        # === Base pose ===
        self.base_offset = -0.05
        self.base_positions = [self.base_offset] * 6
        self.relative_targets = [0.0] * 6
        self.target_positions = self.base_positions[:]

        # === НОВЫЕ ГРУППЫ ===
        self.groupA = [0, 3, 5]  # 1,4,6
        self.groupB = [1, 2, 4]  # 2,3,5

        # === Thresholds ===
        self.thresholds = {
            'w': 2.6,
            's': 2.2,
            'a': 2.65,
            'd': 2.65
        }

        # === FSM ===
        self.active_command = None
        self.step_phase = 'IDLE'
        self.active_group = None
        self.pos_at_start = [0.0] * 6
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

        self.get_logger().info("Hexapod controller started (NEW GAIT)")

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
    # FSM (НОВАЯ ЛОГИКА)
    # =========================

    def joint_callback(self, msg):

        if len(msg.position) < 6:
            return

        pos = msg.position[:6]

        # timeout команды
        if self.active_command:
            if time.time() - self.last_key_time > self.key_timeout:
                self.active_command = None
                self.step_phase = 'IDLE'

        if not self.active_command:
            return

        threshold = self.thresholds[self.active_command]

        # === направление движения ===

        if self.active_command == 'w':
            incA = +self.step_size
            incB = +self.step_size

        elif self.active_command == 'd':
            incA = -self.step_size
            incB = -self.step_size

        elif self.active_command == 'a':
            incA = -self.step_size
            incB = +self.step_size

        elif self.active_command == 's':
            incA = +self.step_size
            incB = -self.step_size

        # === FSM ===

        if self.step_phase == 'IDLE':
            self.start_step('A', incA, incB, pos)
            return

        if self.step_phase == 'WAIT_MOTION':
            if self.motion_started(pos):
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