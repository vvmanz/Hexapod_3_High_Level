#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
import math


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class HexapodCPGController(Node):
    def __init__(self):
        super().__init__('hexapod_cpg_controller')

        self.ns = '/hexapod_0'

        # === ПАРАМЕТРЫ ДВИЖЕНИЯ ===
        self.radius = 3.0
        self.cx = 0.0
        self.cy = 0.0

        # === CPG ===
        self.N = 6
        self.phases = [0.0] * self.N

        self.omega = 6.0           # базовая частота
        self.K = 2.0               # сила синхронизации

        # матрица желаемых фазовых сдвигов
        self.phase_bias = [[0.0]*6 for _ in range(6)]

        tripod_A = [0, 3, 4]
        tripod_B = [1, 2, 5]

        for i in range(6):
            for j in range(6):
                if (i in tripod_A and j in tripod_A) or (i in tripod_B and j in tripod_B):
                    self.phase_bias[i][j] = 0.0
                else:
                    self.phase_bias[i][j] = math.pi

        # === УПРАВЛЕНИЕ ===
        self.k_heading = 2.5
        self.turn_gain = 1.2

        # === СОСТОЯНИЕ ===
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # === ROS ===
        self.pub = self.create_publisher(
            Float64MultiArray,
            f'{self.ns}/set_velocity_targets',
            10
        )

        self.create_subscription(
            Odometry,
            f'{self.ns}/odom',
            self.odom_callback,
            10
        )

        self.timer = self.create_timer(0.02, self.loop)  # 50 Гц

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self.theta = math.atan2(siny, cosy)

    def update_cpg(self, dt):
        new_phases = self.phases.copy()

        for i in range(self.N):
            coupling = 0.0
            for j in range(self.N):
                coupling += math.sin(
                    self.phases[j] - self.phases[i] - self.phase_bias[i][j]
                )

            dphi = self.omega + self.K * coupling
            new_phases[i] += dphi * dt

        self.phases = [p % (2*math.pi) for p in new_phases]

    def leg_output(self, phi):
        """
        Генерация гладкой походки:
        sin(phi) формирует цикличное движение
        """
        return math.sin(phi)

    def loop(self):
        dt = 0.02

        # === CPG ===
        self.update_cpg(dt)

        # === движение по окружности ===
        dx = self.x - self.cx
        dy = self.y - self.cy
        r = math.hypot(dx, dy)

        alpha = math.atan2(dy, dx)
        psi_ref = alpha + math.pi/2

        heading_error = wrap_pi(psi_ref - self.theta)
        turn = self.k_heading * heading_error

        # === генерация скоростей ===
        speeds = []

        base_amp = 6.0  # амплитуда вращения ног

        for i in range(6):
            osc = self.leg_output(self.phases[i])

            # преобразуем осцилляцию в скорость
            speed = base_amp * (0.5 + 0.5 * osc)

            # поворот
            if i % 2 == 0:  # левая сторона
                speed -= self.turn_gain * turn
            else:
                speed += self.turn_gain * turn

            # ограничение
            speed = max(-13.0, min(13.0, speed))

            speeds.append(speed)

        msg = Float64MultiArray()
        msg.data = speeds
        self.pub.publish(msg)

        self.get_logger().info(f"r={r:.2f} turn={turn:.2f}")


def main():
    rclpy.init()
    node = HexapodCPGController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()