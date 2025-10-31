
from Code_utilities.SeriesAnalyzer import SeriesAnalyzer



if __name__ == "__main__":

    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/NEC_camera/Power Dependence/Single Measurements'
    series_analyzer = SeriesAnalyzer(dir_path,
                                     camera_name = 'NEC')
    fitting_coefficients_df = series_analyzer.get_all_fitting_coefficients(bool_plot_2D_maps = False,
                                                                           bool_get_angles=True,
                                                                           known_voltage_at_known_angle_in_V = 0.041,
                                                                           known_angle = 90
                                                                           )
    fig, ax, (slope, intercept) = series_analyzer.plot_temperature_vs_power(fitting_coefficients_df,
                                                                            y_label='Gaussian amplitude (counts)',
                                                                            bool_fixed_intercept=False,
                                                                            title='NEC, fixing intercept')
