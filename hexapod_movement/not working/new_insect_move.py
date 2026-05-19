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


class HexapodController(Node):

    def __init__(self):
        super().__init__('hexapod_controller')

        self.ns = 'hexapod_0'

        # =========================
        # PARAMETERS
        # =========================

        self.step_size = 6.28  # полный оборот
        self.threshold = 1.0   # когда запускать "перехват"
        self.dt = 0.02         # частота обновления

        self.slow_speed = 0.5   # медленное движение (рад/сек)
        self.fast_speed = 4.0   # быстрое докручивание

        # =========================
        # STATE
        # =========================

        self.base_positions = [-0.05] * 6
        self.relative_targets = [0.0] * 6
        self.target_positions = self.base_positions[:]

        self.current_pos = [0.0] * 6

        # для каждой ноги
        self.phase_progress = [0.0] * 6
        self.fast_mode = [False] * 6

        # активная "ведущая" нога
        self.active_leg = 0

        # команда
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
        # TIMER (главный цикл!)
        # =========================

        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Hexapod controller started (continuous gait)")

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
    # JOINT STATE
    # =========================

    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.current_pos = msg.position[:6]

    # =========================
    # CONTROL LOOP (главная магия)
    # =========================

    def control_loop(self):

        # timeout
        if self.active_command:
            if time.time() - self.last_key_time > self.key_timeout:
                self.active_command = None
                return

        if not self.active_command:
            return

        signs = self.get_signs(self.active_command)

        # === обновление всех ног ===

        for i in range(6):

            if self.fast_mode[i]:
                # быстрое движение
                self.phase_progress[i] += self.fast_speed * self.dt
            else:
                # медленное движение
                self.phase_progress[i] += self.slow_speed * self.dt

            # ограничение
            if self.phase_progress[i] > self.step_size:
                self.phase_progress[i] = 0.0
                self.fast_mode[i] = False

        # === trigger логика ===

        lead = self.active_leg

        remaining = self.step_size - self.phase_progress[lead]

        if remaining < self.threshold:
            next_leg = (lead + 1) % 6

            # включаем быстрый режим следующей ноги
            self.fast_mode[next_leg] = True

            # текущая перестаёт быть лидером
            self.active_leg = next_leg

        # === формирование target ===

        for i in range(6):
            self.relative_targets[i] = signs[i] * self.phase_progress[i]

        self.publish_targets()

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