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
import math
import csv
from datetime import datetime


class HexapodController(Node):

    def __init__(self, enable_logging=False):
        super().__init__('hexapod_controller')

        self.ns = 'hexapod_0'
        self.Ct = 6.12

        # =========================
        # PARAMETERS
        # =========================

        self.dt = 0.02
        self.speed = 8.0  # скорость вращения

        self.step_size = 6.28  # полный оборот

        # 🔥 СМЕЩЕНИЯ (радианы)
        self.X = 1.4  # передние (1,2)
        self.M = 1.0  # средние (3,4)
        self.N = 0.3  # задние (5,6)

        # offsets по ногам
        self.phase_offsets = [
            -self.X, -self.X,   # 1,2
            -self.M, -self.M,   # 3,4
            -self.N, -self.N    # 5,6
        ]

        # =========================
        # STATE
        # =========================

        self.phase = 0.0

        self.base_offset = -0.05
        self.base_positions = [self.base_offset] * 6
        self.target_positions = self.base_positions[:]

        self.active_command = None
        self.last_key_time = 0.0
        self.key_timeout = 1.0

        # =========================
        # LOGGING
        # =========================

        self.enable_logging = enable_logging
        self.log_interval = 0.1
        self.last_log_time = 0.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_name = f"{self.ns}_{timestamp}.csv"

        if self.enable_logging:
            with open(self.csv_name, 'w', newline='') as f:
                csv.writer(f).writerow(
                    [f'angle_{i}' for i in range(1, 7)] +
                    [f'speed_{i}' for i in range(1, 7)] +
                    [f'curr_{i}' for i in range(1, 7)] +
                    ['x', 'y', 'phi']
                )

        # =========================
        # ROS
        # =========================

        self.publisher = self.create_publisher(
            Float64MultiArray,
            f'{self.ns}/set_position_targets',
            10
        )

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

        # =========================
        # KEYBOARD
        # =========================

        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        threading.Thread(
            target=self.keyboard_listener,
            daemon=True
        ).start()

        # =========================
        # TIMER
        # =========================

        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Hexapod controller started (jump gait)")

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

    def joint_callback(self, msg):
        pass

    # =========================
    # MAIN CONTROL
    # =========================

    def control_loop(self):

        # timeout
        if self.active_command:
            if time.time() - self.last_key_time > self.key_timeout:
                self.active_command = None

        if not self.active_command:
            return

        # обновляем фазу
        self.phase += self.speed * self.dt
        if self.phase > self.step_size:
            self.phase -= self.step_size

        signs = self.get_signs(self.active_command)

        targets = []

        for i in range(6):

            phi = self.phase + self.phase_offsets[i]

            # нормализация
            while phi < 0:
                phi += self.step_size
            while phi > self.step_size:
                phi -= self.step_size

            # линейное вращение (можно заменить на синус)
            value = phi

            targets.append(self.base_positions[i] + signs[i] * value)

        self.publish_targets(targets)

    # =========================
    # SIGNS
    # =========================

    def get_signs(self, cmd):

        if cmd == 'w':
            return [1]*6

        elif cmd == 's':
            return [-1]*6

        elif cmd == 'd':
            return [1, -1, 1, -1, 1, -1]

        elif cmd == 'a':
            return [-1, 1, -1, 1, -1, 1]

    # =========================
    # PUBLISH
    # =========================

    def publish_targets(self, targets):
        msg = Float64MultiArray()
        msg.data = targets
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