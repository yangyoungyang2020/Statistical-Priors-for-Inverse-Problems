import numpy as np


class ConfigUtils:

    @staticmethod
    def get_sensor_positions(count, room_length, room_width, origin_location):
        # TODO: Convert to generic code for arbitrary sensor count and room size
        sensor_x = [-1.5, -1.2, -0.9, -0.6, -0.3, 0,
                    0.3, 0.6, 0.9, 1.2, 1.5, 1.5,
                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                    1.5, 1.5, 1.5, 1.2, 0.9, 0.6,
                    0.3, 0, -0.3, -0.6, -0.9, -1.2,
                    -1.5, -1.5, -1.5, -1.5, -1.5,
                    -1.5, -1.5, -1.5, -1.5, -1.5]
        sensor_y = [-1.5, -1.5, -1.5, -1.5, -1.5,
                    -1.5, -1.5, -1.5, -1.5, -1.5,
                    -1.5, -1.2, -0.9, -0.6, -0.3, 0,
                    0.3, 0.6, 0.9, 1.2, 1.5, 1.5,
                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                    1.5, 1.5, 1.5, 1.2, 0.9, 0.6,
                    0.3, 0, -0.3, -0.6, -0.9, -1.2]
        sensor_positions = np.transpose(np.array([sensor_x, sensor_y]))
        return sensor_positions

    @staticmethod
    def get_sensor_links(count, transceiver):
        # TODO: Convert to code for different transmit and receive sensors
        txrx_links = []
        if transceiver:
            for i in range(count):
                for j in range(count):
                    if i != j:
                        txrx_links.append((i, j))
        else:
            for i in range(count):
                for j in range(count):
                    txrx_links.append((i, j))
        return txrx_links