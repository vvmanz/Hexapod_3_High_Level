import csv
import math
import time
from datetime import datetime


def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class HexapodLogger:
    def __init__(self, namespace='hexapod_0', Ct=6.12, log_interval=1, spawn_x=0.0, spawn_y=0.0):
        self.ns = namespace
        self.Ct = Ct
        self.log_interval = log_interval

        self.spawn_x = spawn_x
        self.spawn_y = spawn_y

        self.enabled = False
        self.dataset = None
        self.odom = None

        self.prev_x = None
        self.prev_y = None
        self.prev_phi = None

        self.timestep = 0
        self.last_log_time = 0.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_name = f"logs/{self.ns}_{timestamp}.csv"

        headers = (
            ['timestep'] +
            [f'angle_{i}' for i in range(1, 7)] +
            [f'speed_{i}' for i in range(1, 7)] +
            [f'curr_{i}' for i in range(1, 7)] +
            ['x', 'y', 'phi', 'd_x', 'd_y', 'd_phi']
        )

        with open(self.csv_name, 'w', newline='') as f:
            csv.writer(f).writerow(headers)

    def start(self):
        self.enabled = True

    def stop(self):
        self.enabled = False

    def set_dataset(self, msg):
        if len(msg.data) != 18:
            return
        self.dataset = list(msg.data)
        self.try_log()

    def set_odom(self, msg):
        self.odom = msg
        self.try_log()

    def try_log(self):
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_log_time < self.log_interval:
            return

        if self.dataset is None or self.odom is None:
            return

        d = self.dataset
        o = self.odom.pose.pose

        angles = [math.degrees(a) for a in list(d[0:6])]
        speeds = list(d[6:12])
        currents = [tau / self.Ct for tau in list(d[12:18])]

        x = o.position.x - self.spawn_x
        y = o.position.y - self.spawn_y
        phi = quat_to_yaw(
            o.orientation.x,
            o.orientation.y,
            o.orientation.z,
            o.orientation.w
        )

        if self.prev_x is None:
            d_x = 0.0
            d_y = 0.0
            d_phi = 0.0
        else:
            d_x = x - self.prev_x
            d_y = y - self.prev_y
            d_phi = phi - self.prev_phi

        self.prev_x = x
        self.prev_y = y
        self.prev_phi = phi

        row = (
            [self.timestep] +
            angles +
            speeds +
            currents +
            [x, y, phi, d_x, d_y, d_phi]
        )

        with open(self.csv_name, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        self.timestep += 1
        self.last_log_time = now