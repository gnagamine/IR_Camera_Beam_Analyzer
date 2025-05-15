from BeamAnalysis import AnalysisIRCamera
import os
from SeriesAnalyzer import SeriesAnalyzer



if __name__ == "__main__":

    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Power_series'
    background_filename = 'IR_00051_background.csv'
    series_analyzer = SeriesAnalyzer(dir_path, background_filename)
    fitting_coefficients_df = series_analyzer.get_all_fitting_coefficients(bool_plot_2D_maps = False,
                                                                           bool_get_angles=True)
    fig, ax, (slope, intercept) = series_analyzer.plot_temperature_vs_power(fitting_coefficients_df)
