import numpy as np
from scipy.optimize import curve_fit
from Code_utilities.CalibrationCurvePolarizersMovingSecond import CalibrationCurveMovingSecondPolarizer
from Code_utilities.CalibrationCurvePolarizersMovingFirst import CalibrationCurveMovingFirstPolarizer



import matplotlib.pyplot as plt

class PowerExtractorFromPolarizers:
    def __init__(self,
                 known_angle=None,
                 known_voltage_at_known_angle_in_V=None,
                 desired_angle=None,
                 moved_polarizer: str = None,):
        print(f'Moved {moved_polarizer} polarizer')
        self.moved_polarizer = moved_polarizer
        if self.moved_polarizer  != 'first' and self.moved_polarizer != 'second':
            raise ValueError("moved_polarizer is not defined, enter the string either 'first' or 'second'")
        self.Voltage_to_Watt= 1.54E+5 ### from gentec calibration documentation
        if self.moved_polarizer == 'first':
            self.calibration_curve = CalibrationCurveMovingFirstPolarizer()
        if self.moved_polarizer == 'second':
            self.calibration_curve = CalibrationCurveMovingSecondPolarizer()
        self.fitting_coefficients = self.get_fitting_coefficients_calibration()
        (self.maximum_amplitude_mV, self.normalized_offset, self.fi) = self.fitting_coefficients # Fitting is P = maximum_amplitude[cos^2θ + normalized_offset]
        self.maximum_voltage = self.get_maximum_voltage(known_angle,
                                                        known_voltage_at_known_angle_in_V)
        self.voltage_at_desired_angle = self.get_voltage_at_desired_angle(desired_angle)
        uW = 1E+6
        self.power_at_angle_uW =(self.voltage_at_desired_angle /self.Voltage_to_Watt)*uW


    def get_voltage_at_desired_angle(self,
                                     desired_angle,):
        angular_difference = desired_angle - 90
        cos4thetaplusphi = np.cos(np.deg2rad(angular_difference)+np.deg2rad(self.fi))**4
        voltage_at_desired_angle = self.maximum_voltage *(cos4thetaplusphi+self.normalized_offset)
        return voltage_at_desired_angle
    def get_maximum_voltage(self,
                            known_angle,
                            known_voltage_at_known_angle):
        angular_difference = known_angle - 90
        cos4thetaplusphi = np.cos(np.deg2rad(angular_difference)+np.deg2rad(self.fi))**4
        maximum_voltage = known_voltage_at_known_angle/(cos4thetaplusphi+self.normalized_offset)
        return maximum_voltage

    def cosine_squared_plus_offset(self,
                                   angle_deg: np.ndarray,
                                   amplitude_mV: float,
                                   offset_mV: float,
                                   fi: float) -> np.ndarray:
        """A·cos²θ + C  with θ in degrees."""
        return amplitude_mV * (np.cos(np.deg2rad(angle_deg)+np.deg2rad(fi)) ** 2 + offset_mV)
    def cosine_fourth_plus_offset_plus_fi(self,
                                          angle_deg: np.ndarray,
                                          amplitude_mV: float,
                                          normalized_offset_mV: float,
                                          fi: float) -> np.ndarray:
        """A·[cos4(θ+fi) C]  with θ in degrees."""
        return amplitude_mV * (np.cos(np.deg2rad(angle_deg)+np.deg2rad(fi)) ** 4  + normalized_offset_mV)
        #return amplitude_mV * (np.cos(np.deg2rad(angle_deg+fi)) ** 2 * np.cos(np.deg2rad(angle_deg)) ** 2 + normalized_offset_mV)


    def get_fitting_coefficients_calibration(self,
                                              *,
                                              initial_guess=None):
        """
        Fit V(θ)=A·cos²θ + C to calibration data and (optionally) plot the result.

        """
        angles_deg = self.calibration_curve.angles_deg
        voltages_mV = self.calibration_curve.voltages_mV

        if self.moved_polarizer == 'second':
            if initial_guess is None:
                initial_guess = (250, 7)
            best_fit_params, _ = curve_fit(self.cosine_squared_plus_offset,
                                           angles_deg,
                                           voltages_mV,
                                           p0=initial_guess)
            amplitude_mV, normalized_offset = best_fit_params
            fi=0
        if self.moved_polarizer == 'first':
            if initial_guess is None:
                initial_guess = (250, 7, 0)
            best_fit_params, _ = curve_fit(self.cosine_fourth_plus_offset_plus_fi,
                                           angles_deg,
                                           voltages_mV,
                                           p0=initial_guess)
            amplitude_mV, normalized_offset, fi = best_fit_params

        return amplitude_mV, normalized_offset, fi

    def plot_polarizer_calibration(self,
                                              num_plot_points: int = 500):

        amplitude_mV, normalized_offset, fi = self.fitting_coefficients
        angles_deg = self.calibration_curve.angles_deg
        voltages_mV = self.calibration_curve.voltages_mV
        θ_smooth = np.linspace(angles_deg.min(),
                               angles_deg.max(),
                               num_plot_points)
        fitted_curve_mV = self.cosine_fourth_plus_offset_plus_fi(θ_smooth,
                                                                 amplitude_mV,
                                                                 normalized_offset,
                                                                 fi)

        fig, ax = plt.subplots()
        ax.scatter(angles_deg,
                    voltages_mV,
                    marker="o",
                    label="Measured data")
        if self.moved_polarizer == 'second':
            ax.plot(θ_smooth,
                     fitted_curve_mV,
                     label=f"Fit: {amplitude_mV:.4f}·cos²θ + {normalized_offset:.4f}")
            text_for_annotation = (f"P = Po[cos²θ + C] \n " 
                                   f"= {amplitude_mV:.4f}·[cos²θ + {normalized_offset:.4f}  ]")
        if self.moved_polarizer == 'first':
            ax.plot(θ_smooth,
                     fitted_curve_mV,
                     label=f"Fit: {amplitude_mV:.2f}·[cos²(θ+{fi:.2f}).cos²θ + {normalized_offset:.2f}]")

        if self.moved_polarizer == 'second':
            ax.set_xlabel("Second polarizer angle (°)")
        if self.moved_polarizer == 'first':
            ax.set_xlabel("First polarizer angle (°)")
        plt.ylabel("Lock-in voltage (mV)")
        if self.moved_polarizer == 'second':
            ax.set_title("cos² Fit to Second Polarizer Calibration Data")
        if self.moved_polarizer == 'first':
            ax.set_title("cos⁴ Fit to First Polarizer Calibration Data")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("polarizer_calibration.png")
        return fig, ax

if __name__ == "__main__":

    Extractor = PowerExtractorFromPolarizers(known_angle=90,
                                          known_voltage_at_known_angle_in_V=0.7,
                                          desired_angle=90,
                                          moved_polarizer='first')

    fig, ax = Extractor.plot_polarizer_calibration()

    fig.show()


