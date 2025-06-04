from BeamAnalysis import BeamAnalysis
import os
from SeriesAnalyzer import SeriesAnalyzer



if __name__ == "__main__":

    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Power_series'
    background_filename = 'IR_00051_background.csv'
    series_analyzer = SeriesAnalyzer(dir_path, background_filename)
    fitting_coefficients_df = series_analyzer.get_all_fitting_coefficients(bool_plot_2D_maps = True,
                                                                           bool_get_angles=True,
                                                                           known_voltage_at_known_angle_in_V = 0.730,
                                                                           known_angle = 90
                                                                           )
    # fig, ax, (slope, intercept) = series_analyzer.plot_temperature_vs_power(fitting_coefficients_df)

    fig_fixed_intercept, ax_fixed_intercept, (slope_fixed_intercept, intercept_fixed_intercept) = series_analyzer.plot_temperature_vs_power(fitting_coefficients_df,
                                                                            bool_fixed_intercept=True,
                                                                            title = 'HIKMICRO New, fixing intercept')
    fig_fixed_intercept.show()

    fig_no_fixed_intercept, ax_no_fixed_intercept, (slope_no_fixed_intercept, intercept_no_fixed_intercept) = series_analyzer.plot_temperature_vs_power(fitting_coefficients_df,
                                                                            bool_fixed_intercept=False,
                                                                            title = 'HIKMICRO New, not fixing intercept')
    fig_no_fixed_intercept.show()