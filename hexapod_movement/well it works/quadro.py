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


class HexapodController(Node):

    def __init__(self):
        super().__init__('hexapod_controller')

        self.ns = 'hexapod_0'

        # =========================
        # ВЫБОР МОСТА
        # =========================
        # 0 -> отключены 1-2
        # 1 -> отключены 3-4
        # 2 -> отключены 5-6
        self.disabled_bridge = 0

        # =========================
        # СМЕЩЕНИЯ ДЛЯ КАЖДОГО СЛУЧАЯ
        # =========================
        # формат:
        # disabled_bridge: {bridge_id: shift}
        self.bridge_shifts = {
            0: {  # отключен мост 1
                1: -0.5,   # мост 2
                2: -1.0    # мост 3
            },
            1: {  # отключен мост 2
                0: 0.0,
                2: 0.0
            },
            2: {  # отключен мост 3
                0: -0.5,
                1: -1.0
            }
        }

        # === Motion ===
        self.step_size = 6.28
        self.key_timeout = 1.0
        self.fixed_angle = math.pi

        # === Порог ===
        self.phase_trigger_map = {
            0: 4.0,
            1: 4.0,
            2: 4.4
        }

        self.phase_trigger = self.phase_trigger_map[self.disabled_bridge]
        self.reach_eps = 0.3

        # =========================
        # МОСТЫ
        # =========================
        self.bridges = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        self.fixed_legs = self.bridges[self.disabled_bridge]

        # =========================
        # АКТИВНЫЕ НОГИ
        # =========================
        active_legs = [i for i in range(6) if i not in self.fixed_legs]

        self.pairs = [
            [active_legs[0], active_legs[3]],
            [active_legs[1], active_legs[2]]
        ]

        # =========================
        # FSM
        # =========================
        self.phase_active = [False, False]
        self.phase_triggered = [False, False]

        # =========================
        # TARGETS
        # =========================
        self.base_positions = [0.0] * 6
        self.relative_targets = [0.0] * 6
        self.target_positions = [0.0] * 6

        # =========================
        # ПРИМЕНЯЕМ СМЕЩЕНИЯ
        # =========================
        shifts = self.bridge_shifts[self.disabled_bridge]

        for bridge_id, shift in shifts.items():
            for leg in self.bridges[bridge_id]:
                self.base_positions[leg] = shift

        # =========================
        # COMMAND
        # =========================
        self.active_command = None
        self.last_key_time = 0.0

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

        # =========================
        # KEYBOARD
        # =========================
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        threading.Thread(
            target=self.keyboard_listener,
            daemon=True
        ).start()

        self.get_logger().info(
            f"Started | disabled: {self.disabled_bridge} | pairs: {self.pairs}"
        )

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
    # MAIN LOGIC
    # =========================

    def joint_callback(self, msg):

        if len(msg.position) < 6:
            return

        pos = msg.position[:6]

        if self.active_command:
            if time.time() - self.last_key_time > self.key_timeout:
                self.active_command = None
                self.reset_gait()
                return

        if not self.active_command:
            return

        signs = self.get_signs(self.active_command)

        if not any(self.phase_active):
            self.start_phase(0, signs)
            return

        for i in range(2):

            if not self.phase_active[i]:
                continue

            pair = self.pairs[i]
            next_i = (i + 1) % 2

            if not self.phase_triggered[i]:
                if self.almost_done(pair, pos):
                    self.start_phase(next_i, signs)
                    self.phase_triggered[i] = True

            if self.reached(pair, pos):
                self.phase_active[i] = False
                self.phase_triggered[i] = False

    # =========================
    # SIGNS
    # =========================

    def get_signs(self, cmd):

        signs = [0] * 6

        if cmd in ['w', 's']:
            s = 1 if cmd == 'w' else -1
            for pair in self.pairs:
                for i in pair:
                    signs[i] = s

        if cmd in ['a', 'd']:
            direction = 1 if cmd == 'd' else -1

            p0 = self.pairs[0]
            signs[p0[0]] = direction
            signs[p0[1]] = -direction

            p1 = self.pairs[1]
            signs[p1[0]] = -direction
            signs[p1[1]] = direction

        return signs

    # =========================
    # GAIT
    # =========================

    def start_phase(self, idx, signs):

        pair = self.pairs[idx]

        for i in pair:
            self.relative_targets[i] += signs[i] * self.step_size

        self.phase_active[idx] = True
        self.phase_triggered[idx] = False

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
        self.phase_active = [False, False]
        self.phase_triggered = [False, False]

    # =========================
    # TARGETS
    # =========================

    def compute_targets(self):

        self.target_positions = [
            self.base_positions[i] + self.relative_targets[i]
            for i in range(6)
        ]

        # фиксируем отключенный мост
        for i in self.fixed_legs:
            self.target_positions[i] = self.fixed_angle

    def publish_targets(self):
        self.compute_targets()
        msg = Float64MultiArray()
        msg.data = self.target_positions
        self.publisher.publish(msg)

    # =========================

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        super().destroy_node()


# =============================

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