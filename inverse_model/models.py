import numpy as np

from configs.config import Config


class Model:

    def __init__(self):

        # Physical parameters
        self.frequency = Config.system["frequency"]
        self.wavelength = 3e8 / self.frequency
        self.wave_number = 2 * np.pi / self.wavelength

        # Room parameters
        self.m = Config.doi["inverse_grids"]
        self.number_of_grids = self.m ** 2

        # Sensor parameters
        self.transceiver = Config.sensors["transceivers"]
        self.number_of_rx = Config.sensors["count"]
        self.number_of_tx = Config.sensors["count"]
        self.sensor_positions = Config.sensors["positions"]
        self.sensor_links = Config.sensors["links"]

        # Other parameters
        self.nan_remove = True

    def get_model(self, *args, **kwargs):
        return

    def get_data(self, *args, **kwargs):
        return

    def remove_nan_values(self, field):
        if self.nan_remove:
            np.fill_diagonal(field, np.nan)
            k = field.reshape(field.size, order='F')
            l = [x for x in k if not np.isnan(x)]
            m = np.reshape(l, (self.number_of_tx, self.number_of_rx-1))
            m = np.transpose(m)
            return m
        if not self.nan_remove:
            field[np.isnan(field)] = 0
            return field


class PRytov(Model):

    def get_model(self, direct_field, incident_field, integral_values):
        A = np.zeros((len(self.sensor_links), self.number_of_grids), dtype=complex)
        for i, pair in enumerate(self.sensor_links):
            A[i, :] = np.real(self.wave_number ** 2
                              * np.divide(np.multiply(integral_values[pair[1], :],
                                                      np.transpose(incident_field[:, pair[0]])),
                                          direct_field[pair[1], pair[0]]))
        return A

    def get_data(self, total_power, direct_power):
        direct_power = self.remove_nan_values(direct_power)
        data = (total_power - direct_power) / (10 * np.log10(np.exp(2)))
        data = data.reshape(data.size, order='F')
        return data


class PRytovComplex(Model):

    def get_model(self, direct_field, incident_field, integral_values):
        A = np.zeros((len(self.sensor_links), self.number_of_grids), dtype=complex)
        for i, pair in enumerate(self.sensor_links):
            A[i, :] = (self.wave_number ** 2) * np.divide(np.multiply(integral_values[pair[1], :],
                                                                      np.transpose(incident_field[:, pair[0]])),
                                                          direct_field[pair[1], pair[0]])

        A_real = np.real(A)
        A_imag = np.imag(A)
        A_final = np.concatenate((A_real, -A_imag), axis=1)
        return A_final

    def get_data(self, total_power, direct_power):
        direct_power = self.remove_nan_values(direct_power)
        data = (total_power - direct_power) / (10 * np.log10(np.exp(2)))
        data = data.reshape(data.size, order='F')
        return data
