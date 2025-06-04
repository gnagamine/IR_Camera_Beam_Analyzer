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