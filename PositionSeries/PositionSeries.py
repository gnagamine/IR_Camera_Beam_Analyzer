import pandas as pd

from Code_utilities.SeriesAnalyzer import SeriesAnalyzer

if __name__ == "__main__":

    dir_path_HIKMICRO = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250516/position series_HIKMICRO'
    series_analyzer_HIKMICRO = SeriesAnalyzer(dir_path_HIKMICRO,
                                              bool_individual_backgrounds_taken=True,
                                              camera_name = 'HIKMICRO')
    fitting_coefficients_df_HIKMICRO = series_analyzer_HIKMICRO.get_all_fitting_coefficients(bool_plot_2D_maps = False)


    dir_path_NEC = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250516/position series_NEC/each_delay'
    series_analyzer_NEC = SeriesAnalyzer(dir_path_NEC,
                                         camera_name = 'NEC')
    fitting_coefficients_df_NEC = series_analyzer_NEC.get_all_fitting_coefficients(bool_plot_2D_maps = False)

    FWHM_side_by_side_df=pd.DataFrame()
    FWHM_side_by_side_df['FWHM_HIKMICRO_X'] = fitting_coefficients_df_HIKMICRO['fwhm_x']
    FWHM_side_by_side_df['FWHM_HIKMICRO_Y'] = fitting_coefficients_df_HIKMICRO['fwhm_y']
    FWHM_side_by_side_df['FWHM_NEC_X'] = fitting_coefficients_df_NEC['fwhm_x']
    FWHM_side_by_side_df['FWHM_NEC_Y'] = fitting_coefficients_df_NEC['fwhm_y']

