import matplotlib
matplotlib.use('macosx')
from PixelPlotter import PixelPlotter

if __name__ == '__main__':
    #%%

    NEC_dir_path_highpowers = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250523/NEC/HighPowersNoCalibration')
    measured_voltage_converted_from_lock_in_to_osciloscope_V = .100*3.08
    angle_pol_voltage_measurement= 60

    pixel_plotter_NEC_highpowers = PixelPlotter(maps_dir_path=NEC_dir_path_highpowers,
                                     camera_name='NEC',
                                     known_angle=angle_pol_voltage_measurement,
                                     known_voltage_at_known_angle_in_V=measured_voltage_converted_from_lock_in_to_osciloscope_V)

    Y_ref_postion_background_column = 90

    map_array_NEC = pixel_plotter_NEC_highpowers.get_single_map(map_filename='20 degrees.csv')
    fig_NEC_map, ax_NEC_map = pixel_plotter_NEC_highpowers.plot_map(map_data=map_array_NEC)
    fig_NEC_map.show()
    x_pos = 108
    y_pos = 104

    fig_intensity_vs_power_highpowers, ax_intensity_vs_power_highpowers = pixel_plotter_NEC_highpowers.plot_intensity_vs_power(x_pos=x_pos,
                                                        y_pos=y_pos,
                                                        moved_polarizer='first',
                                                        Y_ref_postion_background_column=Y_ref_postion_background_column,
                                                        bool_subtract_background=True)
    title = 'NEC High Power with NO calibration'
    ax_intensity_vs_power_highpowers.set_title(title)
    fig_intensity_vs_power_highpowers.show()
    fig_intensity_vs_power_highpowers.savefig(title)