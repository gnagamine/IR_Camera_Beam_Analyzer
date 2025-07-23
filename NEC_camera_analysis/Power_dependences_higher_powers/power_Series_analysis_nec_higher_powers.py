
from SeriesAnalyzer import SeriesAnalyzer
import pandas as pd




if __name__ == "__main__":

    dir_path_higherpowers  = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250519/all_angles'
    data_center = (2995, 5158)
    plot_range= 1000
    series_analyzer_higherpowers  = SeriesAnalyzer(dir_path_higherpowers,
                                                   camera_name = 'NEC',
                                                   crop_range_x_um = (data_center[0] - plot_range, data_center[0] + plot_range),
                                                   crop_range_y_um = (data_center[1] - plot_range, data_center[1] + plot_range))

    fitting_coefficients_df_higherpowers = series_analyzer_higherpowers .get_all_fitting_coefficients(bool_plot_2D_maps = False,
                                                                           bool_get_angles=True,
                                                                           known_voltage_at_known_angle_in_V = 0.454,
                                                                           known_angle = 90
                                                                           )
    # fig, ax, (slope, intercept) = series_analyzer_higherpowers .plot_temperature_vs_power(fitting_coefficients_df_higherpowers ,
    #                                                                         y_label='Gaussian amplitude (counts)',
    #                                                                         bool_fixed_intercept=True,
    #                                                                         title='NEC,  fixing intercept')


    dir_path_lowerpowers = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/NEC_camera/Power Dependence/Single Measurements'
    series_analyzer_lowerpowers  = SeriesAnalyzer(dir_path_lowerpowers,
                                                  camera_name = 'NEC')
    fitting_coefficients_df_lowerpowers = series_analyzer_lowerpowers .get_all_fitting_coefficients(bool_plot_2D_maps = False,
                                                                           bool_get_angles=True,
                                                                           known_voltage_at_known_angle_in_V = 0.041,
                                                                           known_angle = 90
                                                                           )


    all_coeffs_df = pd.concat([fitting_coefficients_df_higherpowers, fitting_coefficients_df_lowerpowers])
    fig, ax, (slope, intercept) = series_analyzer_lowerpowers.plot_temperature_vs_power(all_coeffs_df,
                                                                            y_label='Gaussian amplitude (counts)',
                                                                            bool_fixed_intercept=False,
                                                                            title='NEC, fixing intercept all together')




    #%%
    # from BeamAnalysis import BeamAnalysis
    # filename = '10 degrees.csv'
    #
    # exampe_analysis = BeamAnalysis(dir_path = dir_path,
    #                                signal_filename = filename,
    #                                camera_name = 'NEC',
    #                                crop_range_x_um = (data_center[0] - plot_range, data_center[0] + plot_range),
    #                                crop_range_y_um = (data_center[1] - plot_range, data_center[1] + plot_range))
    # exampe_analysis.plot_fit_comparison()

