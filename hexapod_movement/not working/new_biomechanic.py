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
        # PARAMETERS
        # =========================

        self.dt = 0.02
        self.cycle_speed = 10.5  # рад/сек

        self.step_amplitude = 1.8
        self.lift_amplitude = 1.2

        self.duty_factor = 0.65  # % времени в опоре

        # фазовые сдвиги (волна)
        self.phase_offsets = [
            0.0,
            math.pi / 3,
            2 * math.pi / 3,
            math.pi,
            4 * math.pi / 3,
            5 * math.pi / 3
        ]

        # =========================
        # STATE
        # =========================

        self.phase = 0.0

        self.base_positions = [-0.05] * 6
        self.target_positions = self.base_positions[:]

        self.active_command = None
        self.last_key_time = 0.0
        self.key_timeout = 1.0

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

        threading.Thread(target=self.keyboard_listener, daemon=True).start()

        # =========================
        # TIMER
        # =========================

        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Hexapod controller started (biomechanical gait)")

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
        pass  # не нужен для этой модели

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

        # обновляем глобальную фазу
        self.phase += self.cycle_speed * self.dt
        if self.phase > 2 * math.pi:
            self.phase -= 2 * math.pi

        signs = self.get_signs(self.active_command)

        targets = []

        for i in range(6):

            # индивидуальная фаза ноги
            phi = self.phase + self.phase_offsets[i]
            phi = phi % (2 * math.pi)

            stance_limit = self.duty_factor * 2 * math.pi

            # === STANCE (толкаем назад) ===
            if phi < stance_limit:

                x = -self.step_amplitude * (phi / stance_limit)
                z = 0.0

            # === SWING (поднимаем и переносим) ===
            else:
                swing_phase = (phi - stance_limit) / (2 * math.pi - stance_limit)

                x = -self.step_amplitude + self.step_amplitude * swing_phase
                z = self.lift_amplitude * math.sin(math.pi * swing_phase)

            # применяем направление
            x *= signs[i]

            # итоговая позиция (упрощённо 1 DOF)
            targets.append(self.base_positions[i] + x)

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