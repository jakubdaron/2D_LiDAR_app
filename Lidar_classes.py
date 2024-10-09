import serial
import serial.tools.list_ports
import time
from abc import ABC, abstractmethod
from queue import Queue
import os
import csv
import math
import struct
import copy
from serial.serialutil import SerialException


class LiDAR(ABC):
    def __init__(self, bandrate, timeout):
        self.ser = None
        self.port = None
        self.bandrate = bandrate
        self.is_active = False
        self.points_X = []
        self.points_Y = []
        self.save_point_cloud = {'X': [], 'Y': []}
        self.scan = Queue()
        self.hwid = None
        self.name = None
        self.timeout = timeout

    def save_scan_to_csv(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join('Scans', f"{self.name[0]}_scan_{timestamp}.csv")
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(self.save_point_cloud['X'])):
                writer.writerow([self.save_point_cloud['X'][i], self.save_point_cloud['Y'][i]])

    def check_serial_port(self):
        available_ports = list(serial.tools.list_ports.comports())
        for port_c, desc, hwid in available_ports:
            if self.hwid in hwid:
                self.is_active = True
                self.port = port_c
                print(f"{self.name} is connected to a computer!")
                self.connect()
                break
        if self.is_active is False:
            print(f"{self.name} is not connected to a computer!")

    def connect(self):
        if self.is_active is True:
            self.ser = serial.Serial(self.port, self.bandrate, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                                     timeout=self.timeout)

    @abstractmethod
    def deactivate(self):
        pass

    @abstractmethod
    def make_full_scan(self, *args, **kwargs):
        pass


class STL27L(LiDAR):
    def __init__(self, bandrate=921600, timeout=None):
        super().__init__(bandrate, timeout)
        self.hwid = '1C6DF6D68E44ED11BFABCEC90A86E0B4'
        self.name = 'Waveshare-STL27L'
        self.check_serial_port()

    def make_full_scan(self):
        try:
            distances = []
            angles = []
            start_time = time.time()
            while (time.time() - start_time) < 0.1:
                first_byte = ord(self.ser.read(1))
                if first_byte == 84:
                    second_byte = ord(self.ser.read(1))
                    if second_byte == 44:
                        data = self.ser.read(45)

                        start_angle = data[3] * 256 + data[2]
                        end_angle = data[41] * 256 + data[40]
                        diff = ((end_angle - start_angle) / 100) / 11

                        decimal_data = [int(byte) for byte in data[4:40]]

                        distances += [decimal_data[i * 3] + decimal_data[i * 3 + 1] * 256 for i in range(0, 12)]
                        angles += [start_angle / 100 + diff * j for j in range(0, 12)]

            self.points_X = [distances[j] * math.sin(math.radians(angles[j]))
                             for j in range(len(distances)) if distances != 0]
            self.points_Y = [distances[j] * math.cos(math.radians(angles[j]))
                             for j in range(len(distances)) if distances != 0]

            self.save_point_cloud['X'] = copy.deepcopy(self.points_X)
            self.save_point_cloud['Y'] = copy.deepcopy(self.points_Y)
            if len(self.save_point_cloud['Y']) > 2000:
                return [self.save_point_cloud['X'], self.save_point_cloud['Y']]
            else:
                return [[], []]
        except IndexError:
            pass
        except SerialException:
            self.ser.close()

    def deactivate(self):
        self.ser.close()
        self.scan.queue.clear()


class A2M8(LiDAR):
    def __init__(self, bandrate=115200, timeout=1):
        super().__init__(bandrate, timeout)
        self.hwid = '0001'
        self.name = 'Slamtec-A2M8-R5'
        self.full_scan = False
        self.sync_byte = b'\xA5'
        self.set_pwm_byte = b'\xF0'
        self.scan_type_byte = b'\x82'
        self.reset_byte = b'\x40'
        self.motor_pwm = 660
        self.check_serial_port()

    def is_scan_full(self, prev_angle, angle):
        if (prev_angle - angle) > 355:
            if len(self.points_X) > 100 and len(self.points_Y) > 100:
                self.save_point_cloud['X'] = copy.deepcopy(self.points_X)
                self.save_point_cloud['Y'] = copy.deepcopy(self.points_Y)
                self.full_scan = True
            self.points_X = []
            self.points_Y = []

    def make_full_scan(self, previous_angle, previous_start_angle):
        sign = {0: 1, 1: -1}
        while self.full_scan is False:
            package = self.ser.read(84)
            if len(package) == 84:
                start_angle = (package[2] + ((package[3] & 0b01111111) << 8)) / 64

                j = 1
                for i in range(0, 80, 5):
                    d = ((package[i + 4] >> 2) + (package[i + 5] << 6))
                    a = (((package[i + 8] & 0b00001111) + ((package[i + 4] & 0b00000001) << 4)) / 8 * sign[(
                            package[i + 4] & 0b00000010) >> 1])

                    angle = (previous_start_angle + (
                            (start_angle - previous_start_angle) % 360)
                            / 32 * j - a) % 360 + 24
                    j += 1
                    self.is_scan_full(previous_angle, angle)
                    self.points_X.append(d * math.sin(math.radians(angle)))
                    self.points_Y.append(d * math.cos(math.radians(angle)))

                    previous_angle = angle

                    d = ((package[i + 6] >> 2) + (package[i + 7] << 6))
                    a = (((package[i + 8] >> 4) + ((package[i + 6] & 0b00000001) << 4)) / 8 * sign[(
                            package[i + 6] & 0b00000010) >> 1])

                    angle = (previous_start_angle + (
                            (start_angle - previous_start_angle) % 360)
                            / 32 * j - a) % 360 + 24
                    j += 1
                    self.is_scan_full(previous_angle, angle)
                    self.points_X.append(d * math.sin(math.radians(angle)))
                    self.points_Y.append(d * math.cos(math.radians(angle)))

                    previous_angle = angle

                previous_start_angle = start_angle
        self.full_scan = False
        return [self.save_point_cloud['X'], self.save_point_cloud['Y']], previous_angle, previous_start_angle

    def deactivate(self):
        self.set_pwm(0)
        self.ser.close()
        self.scan.queue.clear()

    def set_pwm(self, pwm):
        payload = struct.pack("<H", pwm)
        self.send_payload_command(self.set_pwm_byte, payload)

    def send_payload_command(self, cmd, payload):
        size = struct.pack('B', len(payload))
        req = self.sync_byte + cmd + size + payload
        checksum = 0
        for v in struct.unpack('B' * len(req), req):
            checksum ^= v
        req += struct.pack('B', checksum)
        self.ser.write(req)

    def send_command(self, cmd):
        req = self.sync_byte + cmd
        self.ser.write(req)

    def run(self):
        self.ser.setDTR(False)
        self.set_pwm(self.motor_pwm)
        self.send_payload_command(self.scan_type_byte, b'\x00\x00\x00\x00\x00')

    def reset(self):
        self.send_command(self.reset_byte)
        time.sleep(2)
        self.ser.flushInput()
