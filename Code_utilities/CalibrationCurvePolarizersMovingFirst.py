import numpy as np

class CalibrationCurveMovingFirstPolarizer:
    def __init__(self):

        angles_deg_0 = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10, 0],
                             dtype=float)
        voltages_mV_0 = np.array([0.0690, 0.0650, 0.0588, 0.0499, 0.0388,
                               0.0282, 0.0162, 0.0078, 0.00176, 0.00120],
                              dtype=float)

        angles_deg_5 = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10, 0], dtype=float)
        voltages_mV_5= np.array([245, 242, 210, 162, 114, 69, 32, 15.8, 9.3, 8.2],
                             dtype=float) #lock in calibration curve
        #offset = .3 #lock in offset
        voltages_mV_5=voltages_mV_5*3.08 # converting from lock in units to oscilocope units #TODO check if T connector changed the signal

        angles_deg_0 = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10, 0], dtype=float)
        voltages_mV_0 = np.array(
            [826, 814, 694, 553, 416, 310, 236, 162, 141, 109],
            dtype=float
        )#osciloscope calibration curve
        # offset = 82 #lock in offset

        #voltages_mV = voltage_mV_wo_offset-offset

        angles_deg_1 = np.array([90, 60, 40, 20, 0], dtype=float)
        voltages_mV_1= np.array([238, 157, 58, 15.4, 7],
                             dtype=float) #lock in calibration curve

        angles_deg_2 = np.array([0,20, 40, 60, 90], dtype=float)
        voltages_mV_2 = np.array([3.2, 4.6, 12.6, 29, 43.2],
                             dtype=float) #lock in calibration curve

        angles_deg_3 = np.array([90, 60, 40, 20, 0, 110], dtype=float)
        voltages_mV_3 = np.array([34, 22.6, 10.8, 4.2, 3.6, 26],
                             dtype=float) #lock in calibration curve

        angles_deg = angles_deg_5
        voltages_mV = voltages_mV_5


        angles_deg = angles_deg - 90

        self.angles_deg = angles_deg
        self.voltages_mV = voltages_mV