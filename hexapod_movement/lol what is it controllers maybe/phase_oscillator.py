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


class HexapodGaitController(Node):
    def __init__(self):
        super().__init__('hexapod_gait_controller')

        self.ns = '/hexapod_0'

        # ====== ВЫБОР ТРАЕКТОРИИ ======
        self.mode = "circle"   
        # circle / ellipse / figure8 / lissajous / zigzag / square / rectangle

        # ====== ПАРАМЕТРЫ ======
        self.radius = 3.0
        self.cx = 0.0
        self.cy = 0.0

        self.omega_base = 10.0  # базовая скорость ног

        # регулятор направления (PD)
        self.k_heading = 4.5
        self.k_d = 0.5
        self.turn_gain = 2.2

        # состояние
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # время и фаза
        self.time = 0.0
        self.phase = 0.0
        self.prev_error = 0.0

        # ROS
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

    # ====== ODOM ======
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny, cosy)

    # ====== ТРАЕКТОРИИ ======
    def trajectory(self, t):

        if self.mode == "circle":
            R = self.radius
            w = 0.2
            return (
                self.cx + R * math.cos(w * t),
                self.cy + R * math.sin(w * t)
            )

        elif self.mode == "ellipse":
            a, b = 4.0, 2.0
            w = 0.2
            return (
                a * math.cos(w * t),
                b * math.sin(w * t)
            )

        elif self.mode == "figure8":
            a = 3.0
            w = 0.3
            return (
                a * math.sin(w * t),
                a * math.sin(w * t) * math.cos(w * t)
            )

        elif self.mode == "lissajous":
            A, B = 3.0, 3.0
            a, b = 3, 2
            delta = math.pi / 2
            w = 0.3
            return (
                A * math.sin(a * w * t + delta),
                B * math.sin(b * w * t)
            )

        elif self.mode == "zigzag":
            L = 4.0
            speed = 0.5
            return (
                speed * t,
                L * math.sin(0.5 * t)
            )

        elif self.mode == "square":
            L = 4.0
            T = 20.0
            t_mod = t % T

            if t_mod < T/4:
                return (L * (t_mod/(T/4)), 0)
            elif t_mod < T/2:
                return (L, L * ((t_mod - T/4)/(T/4)))
            elif t_mod < 3*T/4:
                return (L * (1 - (t_mod - T/2)/(T/4)), L)
            else:
                return (0, L * (1 - (t_mod - 3*T/4)/(T/4)))

        elif self.mode == "rectangle":
            Lx, Ly = 6.0, 3.0
            T = 20.0
            t_mod = t % T

            if t_mod < T/4:
                return (Lx * (t_mod/(T/4)), 0)
            elif t_mod < T/2:
                return (Lx, Ly * ((t_mod - T/4)/(T/4)))
            elif t_mod < 3*T/4:
                return (Lx * (1 - (t_mod - T/2)/(T/4)), Ly)
            else:
                return (0, Ly * (1 - (t_mod - 3*T/4)/(T/4)))

        return (0.0, 0.0)

    # ====== ГЕНЕРАЦИЯ ШАГА ======
    def leg_velocity(self, phi):
        phi = phi % (2 * math.pi)

        if phi < math.pi:
            return self.omega_base * 0.6   # опора
        else:
            return self.omega_base * 1.4   # перенос

    # ====== MAIN LOOP ======
    def loop(self):
        dt = 0.02

        # время
        self.time += dt
        self.phase += self.omega_base * dt

        # ====== цель ======
        x_ref, y_ref = self.trajectory(self.time)

        dx = x_ref - self.x
        dy = y_ref - self.y

        psi_ref = math.atan2(dy, dx)

        # ====== PD управление ======
        heading_error = wrap_pi(psi_ref - self.theta)

        d_error = (heading_error - self.prev_error) / dt
        self.prev_error = heading_error

        turn = self.k_heading * heading_error + self.k_d * d_error

        # ====== фазы ног ======
        phases = [
            self.phase,
            self.phase + math.pi,
            self.phase + math.pi,
            self.phase,
            self.phase,
            self.phase + math.pi
        ]

        speeds = []

        for i in range(6):
            base = self.leg_velocity(phases[i])

            if i % 2 == 0:  # левая сторона
                speed = base - self.turn_gain * turn
            else:           # правая сторона
                speed = base + self.turn_gain * turn

            speeds.append(speed)

        # ====== publish ======
        msg = Float64MultiArray()
        msg.data = speeds
        self.pub.publish(msg)

        self.get_logger().info(
            f"mode={self.mode} target=({x_ref:.2f},{y_ref:.2f}) turn={turn:.2f}"
        )


def main():
    rclpy.init()
    node = HexapodGaitController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()