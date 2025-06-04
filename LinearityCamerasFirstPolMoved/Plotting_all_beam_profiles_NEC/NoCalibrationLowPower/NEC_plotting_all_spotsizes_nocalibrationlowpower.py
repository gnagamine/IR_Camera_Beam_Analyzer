from SeriesAnalyzer import SeriesAnalyzer

if __name__ == "__main__":
    dir_path_higherpowers = (
        '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz '
        'Camera/20250523/NEC/LowPower_all_same_calibration_done28May')
    data_center = (2410, 6024)
    plot_range = 300
    series_analyzer_higherpowers = SeriesAnalyzer(dir_path_higherpowers,
                                                  camera_name='NEC',
                                                  crop_range_x_um=(
                                                      data_center[0] - plot_range, data_center[0] + plot_range),
                                                  crop_range_y_um=(
                                                      data_center[1] - plot_range, data_center[1] + plot_range))

    fitting_coefficients_df_higherpowers = series_analyzer_higherpowers.get_all_fitting_coefficients(
        bool_plot_2D_maps=True,
        bool_get_angles=True,
        known_voltage_at_known_angle_in_V=0.17,
        known_angle=90,
        moved_polarizer='first'
    )
