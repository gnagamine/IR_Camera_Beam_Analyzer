from BeamAnalysis import AnalysisIRCamera
import os
import pandas as pd
from PowerExtractorFromPolarizer import PowerExtractorFromPolarizers
import matplotlib.pyplot as plt
import numpy as np
class SeriesAnalyzer:
    def __init__(self,
                 dir_path,
                 background_filename):

        self.dir_path = dir_path
        self.background_filename = background_filename

    def get_all_fitting_coefficients(self,
                                     bool_plot_2D_maps = False,
                                     bool_get_angles = False):
        filename_list = os.listdir(self.dir_path)
        fitting_coefficients_list = []
        for filename in filename_list:
            if filename.endswith('.csv') and filename != self.background_filename and not filename.startswith('.'):
                signal_filename = filename
                analysis = AnalysisIRCamera(dir_path = self.dir_path,
                                    signal_filename =signal_filename,
                                    background_filename = self.background_filename)
                popt, fitting_coefficients, _, _ = analysis.fit_gaussian(bool_save_plots=bool_plot_2D_maps)
                fitting_coefficients["filename"] = signal_filename
                fitting_coefficients_list.append(fitting_coefficients)
        if bool_get_angles:
            fitting_coefficients_df = self.generate_angle_column(pd.DataFrame(fitting_coefficients_list))
            fitting_coefficients_df = self.generate_powers_column(fitting_coefficients_df)
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

        angle = (
            fitting_coefficients_df["filename"]
            .str.extract(r"_(\d+)degrees", expand=False)
            .astype(float)        # convert to numeric; NaN for non‑matches
        )

        fitting_coefficients_df = fitting_coefficients_df.assign(angle=angle)
        return fitting_coefficients_df

    def generate_powers_column(self,
                               fitting_coefficients_df):
        fitting_coefficients_df["Power (uW)"] = PowerExtractorFromPolarizers(known_angle=90,
                                          known_voltage_at_known_angle_in_V=0.73,
                                          desired_angle=fitting_coefficients_df["angle"].values).power_at_angle_uW
        return fitting_coefficients_df

    def plot_temperature_vs_power(self,
                                  fitting_coefficients_df):
        """
        Scatter‑plot the fitted peak temperature (amplitude) versus optical
        power and overlay a least‑squares linear fit.

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

        # Perform linear least‑squares fit
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept

        # Plot scatter and fit line
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="Data")
        ax.plot(x_line, y_line, color="red",
                label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
        ax.set_xlabel("THz Power (uW)")
        ax.set_ylabel("Temperature Amplitude (C)")
        ax.legend()
        fig.savefig("Temperature_vs_Power.pdf")

        return fig, ax, (slope, intercept)

if __name__ == "__main__":
    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Power_series'
    background_filename = 'IR_00051_background.csv'
    series_analyzer = SeriesAnalyzer(dir_path, background_filename)
    fitting_coefficients_df = series_analyzer.get_all_fitting_coefficients(bool_get_angles=True)
    fig, ax = series_analyzer.plot_temperature_vs_power(fitting_coefficients_df)


