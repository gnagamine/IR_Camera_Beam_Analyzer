import numpy as np

class CalibrationCurvePolarizers:
    def __init__(self):

        angles_deg = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10, 0],
                             dtype=float)
        voltages_mV = np.array([0.0690, 0.0650, 0.0588, 0.0499, 0.0388,
                               0.0282, 0.0162, 0.0078, 0.00176, 0.00120],
                              dtype=float)

        # angle_deg = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10], dtype=float)
        # voltage_V = np.array([5.44, 5.18, 4.58, 4.07, 3.23,
        #                       2.54, 1.74, 0.744, 0.375],
        #                      dtype=float)

        angles_deg = angles_deg - 90

        self.angles_deg = angles_deg
        self.voltages_mV = voltages_mV