#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry


def wrap_pi(angle: float) -> float:
    """Нормализация угла в диапазон [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class HexapodCircleController(Node):
    def __init__(self):
        super().__init__('hexapod_circle_controller')

        # Namespace модели в Gazebo
        self.ns = '/hexapod_0'

        # Параметры окружности
        self.center_x = 0.0
        self.center_y = 0.0
        self.radius_target = 3.0

        # Направление движения по окружности:
        # +1 = против часовой стрелки, -1 = по часовой стрелке
        self.direction = 1

        # Ограничение по скорости мотора
        self.max_motor_speed = 13.0  # рад/с

        # Базовая скорость вращения моторов
        # Для RHex обычно лучше начинать с 5..8 рад/с, а не с максимума
        self.base_motor_speed = 7.0

        # Коэффициенты управления
        self.k_path = 1.4        # насколько сильно радиальная ошибка меняет желаемый курс
        self.k_heading = 4.0     # насколько сильно ошибка курса меняет поворот
        self.turn_gain = 1.6     # преобразование "поворота корпуса" в разницу скоростей лево/право

        # Ограничение на величину дифференциальной поправки
        self.max_turn_correction = 3.5  # рад/с

        # Мягкий старт, чтобы робот не дёргался в момент запуска
        self.ramp_time_sec = 3.0
        self.start_time = self.get_clock().now()

        # Фильтр сглаживания команд
        self.command_alpha = 0.2
        self.last_motor_cmd = [0.0] * 6

        # Publisher команд скоростей
        self.publisher = self.create_publisher(
            Float64MultiArray,
            f'{self.ns}/set_velocity_targets',
            10
        )

        # Joint states — для мониторинга
        self.joint_sub = self.create_subscription(
            JointState,
            f'{self.ns}/joint_states',
            self.joint_callback,
            10
        )
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_efforts = []

        # Odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            f'{self.ns}/odom',
            self.odom_callback,
            10
        )
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Таймер управления
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Гц

    def joint_callback(self, msg: JointState):
        self.joint_positions = list(msg.position)
        self.joint_velocities = list(msg.velocity)
        self.joint_efforts = list(msg.effort)

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        # Вектор от центра окружности к роботу
        dx = self.x - self.center_x
        dy = self.y - self.center_y
        r_current = math.hypot(dx, dy)

        if r_current < 1e-6:
            return

        # Угол на робот относительно центра
        alpha = math.atan2(dy, dx)

        # Радиальная ошибка
        error_r = r_current - self.radius_target

        # Векторное поле для окружности:
        # желаемый курс = касательная + мягкая поправка от радиальной ошибки
        # atan(k_path * error_r) ограничивает поворот и делает поведение мягким
        psi_ref = alpha + self.direction * (math.pi / 2.0 + math.atan(self.k_path * error_r))

        # Ошибка курса
        heading_error = wrap_pi(psi_ref - self.theta)

        # Базовая скорость уменьшается, если робот далеко от траектории
        # Это помогает не "зарываться" корпусом в пол на резких коррекциях
        speed_scale = 1.0 - 0.20 * abs(error_r)
        speed_scale = max(0.35, min(1.0, speed_scale))

        # Мягкий старт
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        ramp = min(1.0, max(0.0, elapsed / self.ramp_time_sec))

        base = self.base_motor_speed * speed_scale * ramp

        # Управляющее воздействие на поворот
        turn_cmd = self.k_heading * heading_error
        turn_cmd = max(-self.max_turn_correction, min(self.max_turn_correction, turn_cmd))

        # Перевод в скорости левой/правой стороны
        # Для RHex это лучше, чем "жёсткий диффпривод", потому что
        # поворот получается мягким, а базовое движение остаётся поступательным
        left_speed = base - self.turn_gain * turn_cmd
        right_speed = base + self.turn_gain * turn_cmd

        # Ограничение по скорости моторов
        left_speed = max(-self.max_motor_speed, min(self.max_motor_speed, left_speed))
        right_speed = max(-self.max_motor_speed, min(self.max_motor_speed, right_speed))

        # Индексы:
        # 1 - левый передний
        # 2 - правый передний
        # 3 - левый центральный
        # 4 - правый центральный
        # 5 - левый задний
        # 6 - правый задний
        motor_speeds_raw = [
            left_speed,   # 1
            right_speed,  # 2
            left_speed,   # 3
            right_speed,  # 4
            left_speed,   # 5
            right_speed   # 6
        ]

        # Сглаживание команд
        motor_speeds = []
        for i in range(6):
            cmd = (
                self.command_alpha * motor_speeds_raw[i]
                + (1.0 - self.command_alpha) * self.last_motor_cmd[i]
            )
            motor_speeds.append(cmd)

        self.last_motor_cmd = motor_speeds

        msg = Float64MultiArray()
        msg.data = motor_speeds
        self.publisher.publish(msg)

        self.get_logger().info(
            f"r={r_current:.2f} err_r={error_r:.2f} "
            f"psi_ref={psi_ref:.2f} theta={self.theta:.2f} "
            f"left={left_speed:.2f} right={right_speed:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = HexapodCircleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()