from BeamAnalysis import BeamAnalysis
import os
import pandas as pd
from PowerExtractorFromPolarizer import PowerExtractorFromPolarizers
import matplotlib.pyplot as plt
import numpy as np
class SeriesAnalyzer:
    def __init__(self,
                 dir_path,
                 background_filename= None,
                 camera_name = None,
                 bool_individual_backgrounds_taken = False,
                 crop_range_x_um = None,
                 crop_range_y_um = None):

        self.dir_path = dir_path
        self.camera_name = camera_name
        self.crop_range_x_um = crop_range_x_um
        self.crop_range_y_um = crop_range_y_um
        self.bool_individual_backgrounds_taken = bool_individual_backgrounds_taken
        self.background_filename = background_filename

    def get_background_filename(self,
                                signal_filename=None,
                                filename_list=None):
        if self.camera_name == 'HIKMICRO' and self.bool_individual_backgrounds_taken is False:
            return self.background_filename
        if self.camera_name == 'HIKMICRO' and self.bool_individual_backgrounds_taken is False and self.background_filename is None:
            print(f"no background was given. Either enter a background filename or set bool_individual_backgrounds_taken to True, so the script gets the background automatically from the filename")
            return None
        if self.bool_individual_backgrounds_taken is True:
            if signal_filename is None:
                print ("Signal filename is None")
                return None

            # Construct the expected background filename
            expected_background_name = f"background {signal_filename}"

            # Check if the constructed background name exists in the list of files
            if filename_list is not None and expected_background_name in filename_list:
                return expected_background_name
            else:
                # Handle cases where the expected background file is not found
                # For example, print a warning or return None
                print(f"Warning: Background file '{expected_background_name}' not found for signal '{signal_filename}'.")
                return None
        return None  # Default return if no conditions are met

    def get_all_fitting_coefficients(self,
                                     bool_plot_2D_maps = False,
                                     bool_get_angles = False,
                                     known_angle=None,
                                     known_voltage_at_known_angle_in_V=None,
                                     moved_polarizer: str = None):
        filename_list = os.listdir(self.dir_path)
        fitting_coefficients_list = []
        for filename in filename_list:
            if filename.endswith('.csv') and filename != self.background_filename and not filename.startswith('.') and not filename.__contains__('background'):
                signal_filename = filename
                background_filename = self.get_background_filename(signal_filename = signal_filename,
                                                                  filename_list = filename_list)
                analysis = BeamAnalysis(dir_path = self.dir_path,
                                        signal_filename =signal_filename,
                                        background_filename = background_filename,
                                        camera_name=self.camera_name,
                                        crop_range_x_um=self.crop_range_x_um,
                                        crop_range_y_um=self.crop_range_y_um)

                popt, fitting_coefficients, _, _ = analysis.fit_gaussian(bool_save_plots=bool_plot_2D_maps)
                fitting_coefficients["filename"] = signal_filename
                fitting_coefficients_list.append(fitting_coefficients)
        fitting_coefficients_df = pd.DataFrame(fitting_coefficients_list)
        if bool_get_angles:
            fitting_coefficients_df = self.generate_angle_column(pd.DataFrame(fitting_coefficients_list))
            if known_angle is None or known_voltage_at_known_angle_in_V is None:
                print('Please enter the known angle and known voltage at known angle in V')
                return
            fitting_coefficients_df = self.generate_powers_column(fitting_coefficients_df,
                                                                  known_angle=known_angle,
                                                                  kown_voltage_at_known_angle_in_V=known_voltage_at_known_angle_in_V,
                                                                  moved_polarizer=moved_polarizer)
        return fitting_coefficients_df

    def generate_angle_column(self,
                              fitting_coefficients_df ):
        """
        Extract the numeric angle from the *filename* column and append it as a
        new column called ``angle``.

        The expected filename pattern is something like
        ``IR_00045_50degrees.csv``.  A regular expression captures the digits
        immediately before the literal word ``degrees``.  Any filename that
        does not match yields ``NaN`` in the *angle* column.
        """
        if "filename" not in fitting_coefficients_df.columns:
            raise KeyError(
                "'filename' column not found — cannot extract angle values."
            )

        regex_pattern = r"(\d+)\s?degrees"

        angle = (
            fitting_coefficients_df["filename"]
            .str.extract(regex_pattern,
                         expand=False)
            .astype(float)  # convert to numeric; NaN for non‑matches
        )

        fitting_coefficients_df = fitting_coefficients_df.assign(angle=angle)
        return fitting_coefficients_df

    def generate_powers_column(self,
                               fitting_coefficients_df,
                               known_angle=None,
                               kown_voltage_at_known_angle_in_V=None,
                               moved_polarizer: str = None):
        fitting_coefficients_df["Power (uW)"] = PowerExtractorFromPolarizers(known_angle=known_angle,
                                          known_voltage_at_known_angle_in_V=kown_voltage_at_known_angle_in_V,
                                          desired_angle=fitting_coefficients_df["angle"].values,
                                         moved_polarizer=moved_polarizer).power_at_angle_uW
        return fitting_coefficients_df

    def plot_temperature_vs_power(self,
                                                 fitting_coefficients_df,
                                                 y_label=None,
                                              title = None,
                                              bool_fixed_intercept = False):
        """
        Scatter‑plot the fitted peak temperature (amplitude) versus optical
        power and overlay a least‑squares linear fit. If *bool_fixed_intercept*
        is True, the intercept is fixed at zero; otherwise the intercept is
        fitted as a free parameter.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes containing the scatter and fit.
        fit_coeffs : tuple
            (slope, intercept) of the linear model *y = slope·x + intercept*.
        """
        # Extract x (power) and y (peak temperature) as NumPy arrays
        x = fitting_coefficients_df["Power (uW)"].to_numpy(dtype=float)
        y = fitting_coefficients_df["amplitude"].to_numpy(dtype=float)

        # ----- Linear fit --------------------------------------------------
        if bool_fixed_intercept:
            # Intercept forced to zero
            x_reshaped = x[:, np.newaxis]
            slope_tuple = np.linalg.lstsq(x_reshaped,
                                          y,
                                          rcond=None)
            slope = slope_tuple[0][0]
            intercept = 0.0
        else:
            # Intercept is a free parameter
            A = np.vstack([x, np.ones_like(x)]).T  # design matrix [x, 1]
            coeffs, _, _, _ = np.linalg.lstsq(A,
                                              y,
                                              rcond=None)
            slope, intercept = coeffs

        x_line = np.linspace(x.min(),
                             x.max(),
                             200)
        y_line = slope * x_line + intercept

        # Plot scatter and fit line
        fig, ax = plt.subplots()
        ax.scatter(x,
                   y,
                   label="Data")
        if intercept == 0:
            fit_label = f"Fit: y = {slope:.2f}x"
        else:
            fit_label = f"Fit: y = {slope:.2f}x + {intercept:.2f}"
        ax.plot(x_line,
                y_line,
                color="red",
                label=fit_label)
        ax.set_xlabel("THz Power (uW)")
        if y_label is None:
            ax.set_ylabel("Temperature Amplitude (C)")
        else:
            ax.set_ylabel(y_label)
        # The line below was redundant as it's covered by the if/else block above.
        # ax.set_ylabel("Temperature Amplitude (C)")
        ax.legend()
        if title is not None:
            ax.set_title(title)
            fig.savefig(f"{title}.pdf")  # Consider a different filename
        else:
            fig.savefig("Temperature_vs_Power_Zero_Intercept.pdf")  # Consider a different filename

        return fig, ax, (slope, intercept)

if __name__ == "__main__":
    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Power_series'
    background_filename = 'IR_00051_background.csv'
    series_analyzer = SeriesAnalyzer(dir_path, background_filename)
    fitting_coefficients_df = series_analyzer.get_all_fitting_coefficients(bool_get_angles=True)
    fig, ax = series_analyzer.plot_temperature_vs_power(fitting_coefficients_df)


