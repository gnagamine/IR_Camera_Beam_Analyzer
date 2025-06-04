import matplotlib
matplotlib.use('macosx')
from PixelPlotter import PixelPlotter



if __name__ == '__main__':
    #%%

    NEC_dir_path_highpowers = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250523/NEC/HighPowerRange/AllMeasurements')

    pixel_plotter_NEC_highpowers = PixelPlotter(maps_dir_path=NEC_dir_path_highpowers,
                                     camera_name='NEC',
                                     known_angle=90,
                                     known_voltage_at_known_angle_in_V=.17)

    map_array_NEC = pixel_plotter_NEC_highpowers.get_single_map(map_filename='60 degrees.csv')
    fig_NEC_map, ax_NEC_map = pixel_plotter_NEC_highpowers.plot_map(map_data=map_array_NEC)
    fig_NEC_map.show()
    x_pos = 57
    y_pos = 152
    # power_vs_pixel_df = pixel_plotter_NEC_highpowers.get_single_pixel_intensity_vs_power_df(x_pos=x_pos,
    #                                                                              y_pos=y_pos,
    #                                                                              moved_polarizer='first')
    fig_intensity_vs_power_highpowers, ax_intensity_vs_power_highpowers = pixel_plotter_NEC_highpowers.plot_intensity_vs_power(x_pos=x_pos,
                                                        y_pos=y_pos,
                                                        moved_polarizer='first')
    ax_intensity_vs_power_highpowers.set_title('NEC')
    fig_intensity_vs_power_highpowers.show()

    #%%


    # fig_adjusted, ax_adjusted, adjusted_map_array = pixel_plotter_NEC.plot_adjusted_map(map_filename='90 degrees.csv',
    #                                                                                     x_pos_calibration=129,
    #                                                                                     y_pos_calibration=145)
    #
    # fig_adjusted.show()

    adjusted_map_array = pixel_plotter_NEC_highpowers.get_power_adjusted_map(map_array=map_array_NEC,
                                                                  x_pos=x_pos,
                                                                  y_pos=y_pos,
                                                                  moved_polarizer='first')
    background_ref_Y_position = 100*23
    fig_cross_section_plot, ax_cross_section_plot = pixel_plotter_NEC_highpowers.plot_x_cross_section_um(map_array=adjusted_map_array,
                                                                                              x_pos=x_pos,
                                                                                              label='NEC Power Adjusted '
                                                                                                'Intensity',
                                                                                              background_ref_Y_position_um=background_ref_Y_position, )
    fig_cross_section_plot, ax_cross_section_plot = pixel_plotter_NEC_highpowers.plot_x_cross_section_um(map_array=map_array_NEC,
                                                                                              x_pos=x_pos,
                                                                                              label='NEC Raw Intensity',
                                                                                              fig=fig_cross_section_plot,
                                                                                              ax=ax_cross_section_plot,
                                                                                              background_ref_Y_position_um=background_ref_Y_position)

    #%%

    #
    dir_path_HIKMICRO = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250523/HIKMICRO')
    pixel_plotter_HIKMICRO =PixelPlotter(maps_dir_path=dir_path_HIKMICRO,
                                         camera_name='HIKMICRO',
                                         known_angle=90,
                                         known_voltage_at_known_angle_in_V=0.17)
    HIKMICRO_map_array = pixel_plotter_HIKMICRO.get_single_map(map_filename='90 degrees.csv')
    fig_HIKMICRO_map, ax_HIKMICRO_map = pixel_plotter_HIKMICRO.plot_map(map_data=HIKMICRO_map_array)
    fig_HIKMICRO_map.show()
    x_pos_HIKMICRO = 46
    y_pos_HIKMICRO = 184
    background_ref_Y_position_HIKMICRO = 100*12
    x_axis_shift_in_um = +0

    fig_cross_section_plot, ax_cross_section_plot = pixel_plotter_HIKMICRO.plot_x_cross_section_um(map_array=HIKMICRO_map_array,
                                                                                                  x_pos=x_pos_HIKMICRO,
                                                                                                  label='HIKMICRO Raw Intensity',
                                                                                                  fig=fig_cross_section_plot,
                                                                                                  ax=ax_cross_section_plot,
                                                                                                  background_ref_Y_position_um=background_ref_Y_position_HIKMICRO,
                                                                                                   x_axis_shift_in_um=x_axis_shift_in_um)


    ax_cross_section_plot.set_xlim(-500, 500)
    fig_cross_section_plot.show()

    # fig_intensity_vs_power__HIKMICRO, ax_intensity_vs_power__HIKMICRO = pixel_plotter_HIKMICRO.plot_intensity_vs_power(x_pos=x_pos_HIKMICRO,
    #                                                                                                                    y_pos=y_pos_HIKMICRO,
    #                                                                                                                    moved_polarizer='first')
    # ax_intensity_vs_power__HIKMICRO.set_title('HIKMICRO')
    #
    # fig_intensity_vs_power__HIKMICRO.show()


    #%%


    NEC_dir_path_lowpowers = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250523/NEC/LowPowerRange/AllMeasurements')

    pixel_plotter_NEC_lowpowers = PixelPlotter(maps_dir_path=NEC_dir_path_lowpowers,
                                     camera_name='NEC',
                                     known_angle=90,
                                     known_voltage_at_known_angle_in_V=.0172)

    map_array_NEC = pixel_plotter_NEC_lowpowers.get_single_map(map_filename='60 degrees.csv')
    fig_NEC_map, ax_NEC_map = pixel_plotter_NEC_lowpowers.plot_map(map_data=map_array_NEC)
    fig_NEC_map.show()
    x_pos = 57
    y_pos = 152
    power_vs_pixel_df = pixel_plotter_NEC_lowpowers.get_single_pixel_intensity_vs_power_df(x_pos=x_pos,
                                                                                 y_pos=y_pos,
                                                                                 moved_polarizer='first')
    fig_intensity_vs_power_lowpowers, ax_intensity_vs_power_lowpowers = pixel_plotter_NEC_lowpowers.plot_intensity_vs_power(x_pos=x_pos,
                                                        y_pos=y_pos,
                                                        moved_polarizer='first',
                                                        fig=fig_intensity_vs_power_highpowers,
                                                        ax=ax_intensity_vs_power_highpowers)

    fig_intensity_vs_power_lowpowers.show()







    #
    # fig_HIKMICRO_pixel_power_dependence, ax_HIKMICRO_pixel_power_dependence = pixel_plotter_HIKMICRO.plot_intensity_vs_power(x_pos=x_pos_HIKMICRO,
    #                                                                                                                               y_pos=y_pos_HIKMICRO)
    # fig_HIKMICRO_pixel_power_dependence.show()