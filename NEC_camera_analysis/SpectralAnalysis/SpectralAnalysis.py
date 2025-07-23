from SeriesAnalyzer import SeriesAnalyzer
import matplotlib.pyplot as plt


if __name__ == "__main__":

    dir_path_HIKMICRO = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/HIKMICRO_camera/SpectralSeries'

    data_center_HIKMICRO = (2043, 1273)
    data_range_um = 500
    crop_range_X_um_HIKMICRO = (data_center_HIKMICRO[0] - data_range_um, data_center_HIKMICRO[0] + data_range_um,)
    crop_range_Y_um_HIKMICRO = (data_center_HIKMICRO[1] - data_range_um, data_center_HIKMICRO[1] + data_range_um,)

    series_analyzer_HIKMICRO = SeriesAnalyzer(dir_path=dir_path_HIKMICRO,
                                              camera_name='HIKMICRO',
                                              crop_range_x_um=crop_range_X_um_HIKMICRO,
                                              crop_range_y_um=crop_range_Y_um_HIKMICRO)


    fitting_coefficients_df_HIKMICRO = series_analyzer_HIKMICRO.get_all_fitting_coefficients(bool_plot_2D_maps = True,)

    dir_path_NEC = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/NEC_camera/Spectral Dependece/All_Measurements'

    data_center_NEC = (3421, 4461)
    data_range_um = 500
    crop_range_X_um_NEC = (data_center_NEC[0] - data_range_um, data_center_NEC[0] + data_range_um,)
    crop_range_Y_um_NEC = (data_center_NEC[1] - data_range_um, data_center_NEC[1] + data_range_um,)

    series_analyzer_NEC = SeriesAnalyzer(dir_path=dir_path_NEC,
                                         camera_name='NEC',
                                         crop_range_x_um=crop_range_X_um_NEC,
                                         crop_range_y_um=crop_range_Y_um_NEC)

    fitting_coefficients_df_NEC = series_analyzer_NEC.get_all_fitting_coefficients(bool_plot_2D_maps = True,)

