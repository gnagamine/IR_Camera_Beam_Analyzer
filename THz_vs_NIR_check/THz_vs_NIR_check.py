from BeamAnalysis import AnalysisIRCamera
import os
from SeriesAnalyzer import SeriesAnalyzer



if __name__ == "__main__":

    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250512/Test_if_beam_is_thz'
    background_filename = 'background.csv'
    series_analyzer = SeriesAnalyzer(dir_path, background_filename)
    fitting_coefficients_df = series_analyzer.get_all_fitting_coefficients(bool_plot_2D_maps = True,
                                                                           bool_get_angles=True)
