import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

import time

from hexapod_logger import HexapodLogger


class HexapodController(Node):

    def __init__(self):
        super().__init__('hexapod_controller')

        self.ns = 'hexapod_0'

        # ===== Logger (как в waves.py) =====
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
        self.key_timeout = 1000.0

        # ===== Base pose =====
        self.base_offset = -0.05
        self.base_positions = [self.base_offset] * 6
        self.relative_targets = [0.0] * 6
        self.target_positions = self.base_positions[:]

        # ===== Tripod groups =====
        self.group1 = [0, 3, 4]
        self.group2 = [1, 2, 5]

        # ===== Thresholds =====
        # self.thresholds = {
        #     'w': 2.9,
        #     's': 2.7,
        #     'a': 1.3,
        #     'd': 1.3
        # }
        # self.thresholds = {
        #     'w': 2.5,
        #     's': 2.5,
        #     'a': 1.85,
        #     'd': 1.85
        # }
        self.thresholds = {
            'w': 1.5,
            's': 1.5,
            'a': 1.25,
            'd': 1.25
        }


        # ===== FSM =====
        self.active_command = None
        self.step_phase = 'IDLE'
        self.active_group = None
        self.pos_at_start = [0.0] * 6
        self.last_key_time = 0.0

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

        # ✅ ВАЖНО: внешний ввод команд
        self.create_subscription(
            String,
            f'/{self.ns}/command',
            self.command_callback,
            10
        )

        self.get_logger().info("Hexapod controller started (tripod gait, external commands)")

        self.publish_targets()

    # =========================
    # COMMAND INPUT
    # =========================

    def command_callback(self, msg: String):
        cmd = msg.data.strip()

        if cmd in ['w', 'a', 's', 'd']:
            self.active_command = cmd
            self.last_key_time = time.time()
            self.logger.start()

        elif cmd == '' or cmd.lower() == 'stop':
            self.active_command = None
            self.step_phase = 'IDLE'
            self.logger.stop()

    # =========================
    # FSM
    # =========================

    def joint_callback(self, msg):

        if len(msg.position) < 6:
            return

        pos = msg.position[:6]

        # timeout
        if self.active_command:
            if time.time() - self.last_key_time > self.key_timeout:
                self.active_command = None
                self.step_phase = 'IDLE'
                self.logger.stop()
                return

        if not self.active_command:
            return

        threshold = self.thresholds[self.active_command]

        if self.active_command == 'w':
            inc1, inc2 = +self.step_size, +self.step_size
            first = 'GROUP1'

        elif self.active_command == 's':
            inc1, inc2 = -self.step_size, -self.step_size
            first = 'GROUP1'

        elif self.active_command == 'a':
            inc1, inc2 = -self.step_size, +self.step_size
            first = 'GROUP1'

        elif self.active_command == 'd':
            inc1, inc2 = +self.step_size, -self.step_size
            first = 'GROUP2'

        if self.step_phase == 'IDLE':
            self.start_step(first, inc1, inc2, pos)
            return

        if self.step_phase == 'WAIT_MOTION':
            if self.motion_started(pos):
                self.step_phase = self.active_group
            return

        if self.step_phase == 'GROUP1':
            if self.reached(self.group1, pos, threshold):
                self.start_step('GROUP2', inc1, inc2, pos)

        elif self.step_phase == 'GROUP2':
            if self.reached(self.group2, pos, threshold):
                self.start_step('GROUP1', inc1, inc2, pos)

    def start_step(self, group, inc1, inc2, pos):

        if group == 'GROUP1':
            self.apply_increment(self.group1, inc1)
        else:
            self.apply_increment(self.group2, inc2)

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
        self.logger.stop()
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