import numpy as np
import matplotlib
from scipy.optimize import curve_fit
from CalibrationCurvePolarizers import CalibrationCurvePolarizers

# --- Workaround for Matplotlib ≥ 3.8 & PyCharm backend bug --------------------
 # We switch to a more general backend before importing pyplot.
if ("pycharm" in matplotlib.get_backend().lower()) or ("backend_interagg" in matplotlib.get_backend().lower()):
    # Prefer a GUI backend; fall back to the non-interactive 'Agg' backend.
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt

class PowerExtractorFromPolarizers:
    def __init__(self,
                 known_angle=None,
                 known_voltage_at_known_angle_in_V=None,
                 desired_angle=None,
                 ):
        self.Voltage_to_Watt= 1.54E+5 ### from gentec calibration documentation
        self.calibration_curve = CalibrationCurvePolarizers()
        self.fitting_coefficients = self.get_fitting_coefficients_calibration()
        self.maximum_voltage = self.get_maximum_voltage(known_angle,
                                                        known_voltage_at_known_angle_in_V)
        self.voltage_at_desired_angle = self.get_voltage_at_desired_angle(desired_angle,
                                                                          self.maximum_voltage)
        uW = 10E+6
        self.power_at_angle_uW = (self.voltage_at_desired_angle /self.Voltage_to_Watt)*uW


    def get_voltage_at_desired_angle(self,
                                     desired_angle,
                                     maximum_voltage):
        angular_difference = desired_angle - 90
        voltage_at_desired_angle = maximum_voltage * np.cos(np.deg2rad(angular_difference)) ** 2
        return voltage_at_desired_angle
    def get_maximum_voltage(self,
                            known_angle,
                            known_voltage_at_known_angle):
        angular_difference = known_angle - 90
        maximum_voltage = known_voltage_at_known_angle /np.cos(np.deg2rad(angular_difference)) ** 2
        return maximum_voltage

    def cosine_squared_plus_offset(self,
                                   angle_deg: np.ndarray,
                                   amplitude_mV: float,
                                   offset_mV: float) -> np.ndarray:
        """A·cos²θ + C  with θ in degrees."""
        return amplitude_mV * np.cos(np.deg2rad(angle_deg)) ** 2 + offset_mV

    def get_fitting_coefficients_calibration(self,
                                              *,
                                              initial_guess=(0.07, 0.0)):
        """
        Fit V(θ)=A·cos²θ + C to calibration data and (optionally) plot the result.

        """
        angles_deg = self.calibration_curve.angles_deg
        voltages_mV = self.calibration_curve.voltages_mV


        best_fit_params, _ = curve_fit(self.cosine_squared_plus_offset,
                                       angles_deg,
                                       voltages_mV,
                                       p0=initial_guess)
        amplitude_mV, offset_mV = best_fit_params

        return amplitude_mV, offset_mV

    def plot_polarizer_calibration(self,
                                              num_plot_points: int = 500):

        amplitude_mV, offset_mV = self.fitting_coefficients
        angles_deg = self.calibration_curve.angles_deg
        voltages_mV = self.calibration_curve.voltages_mV
        θ_smooth = np.linspace(angles_deg.min(),
                               angles_deg.max(),
                               num_plot_points)
        fitted_curve_mV = self.cosine_squared_plus_offset(θ_smooth,
                                                     amplitude_mV,
                                                     offset_mV)

        plt.figure()
        plt.scatter(angles_deg,
                    voltages_mV,
                    marker="o",
                    label="Measured data")
        plt.plot(θ_smooth,
                 fitted_curve_mV,
                 label=f"Fit: {amplitude_mV:.4f}·cos²θ + {offset_mV:.4f}")
        plt.xlabel("Second polarizer angle (°)")
        plt.ylabel("Lock-in voltage (mV)")
        plt.title("cos² Fit to Polarizer Calibration Data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("polarizer_calibration.png")
        plt.show()

if __name__ == "__main__":

    power = PowerExtractorFromPolarizers(known_angle=90,
                                          known_voltage_at_known_angle_in_V=0.7,
                                          desired_angle=90).power_at_angle_uW


    test = 10E+6*0.7/1.54E+5
    print(test)

