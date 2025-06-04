import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('macosx')
from PixelPlotter import PixelPlotter

if __name__ == '__main__':
    #%%
    import matplotlib

    print(matplotlib.get_backend())
    NEC_dir_path_highpowers = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250523/NEC/HighPowerRange/AllMeasurements')
    measured_voltage_converted_from_lock_in_to_osciloscope_V = .17*3.08
    angle_pol_voltage_measurement= 90

    pixel_plotter_NEC_highpowers = PixelPlotter(maps_dir_path=NEC_dir_path_highpowers,
                                     camera_name='NEC',
                                     known_angle=angle_pol_voltage_measurement,
                                     known_voltage_at_known_angle_in_V=measured_voltage_converted_from_lock_in_to_osciloscope_V)

    Y_ref_postion_background_column = 130

    map_array_NEC = pixel_plotter_NEC_highpowers.get_single_map(map_filename='20 degrees.csv')
    fig_NEC_map, ax_NEC_map = pixel_plotter_NEC_highpowers.plot_map(map_data=map_array_NEC)
    fig_NEC_map.show()
    x_pos = 57
    y_pos = 152

    fig_intensity_vs_power_highpowers, ax_intensity_vs_power_highpowers = pixel_plotter_NEC_highpowers.plot_intensity_vs_power(x_pos=x_pos,
                                                        y_pos=y_pos,
                                                        moved_polarizer='first',
                                                        Y_ref_postion_background_column=Y_ref_postion_background_column,
                                                        bool_subtract_background=True)
    title = 'NEC High Power with calibration'
    ax_intensity_vs_power_highpowers.set_title(title)
    fig_intensity_vs_power_highpowers.show()
    fig_intensity_vs_power_highpowers.savefig(title)

    #%%

    NEC_dir_path_highpowers = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250523/NEC/HighPowerRange/AllMeasurements')
    measured_voltage_converted_from_lock_in_to_osciloscope_V = .17*3.08
    angle_pol_voltage_measurement= 90

    pixel_plotter_NEC_highpowers = PixelPlotter(maps_dir_path=NEC_dir_path_highpowers,
                                     camera_name='NEC',
                                     known_angle=angle_pol_voltage_measurement,
                                     known_voltage_at_known_angle_in_V=measured_voltage_converted_from_lock_in_to_osciloscope_V)

    Y_ref_postion_background_column = 130

    map_array_NEC = pixel_plotter_NEC_highpowers.get_single_map(map_filename='20 degrees.csv')
    fig_NEC_map, ax_NEC_map = pixel_plotter_NEC_highpowers.plot_map(map_data=map_array_NEC)
    fig_NEC_map.show()
    x_pos = 57
    y_pos = 152

    fig_intensity_vs_power_highpowers, ax_intensity_vs_power_highpowers = pixel_plotter_NEC_highpowers.plot_intensity_vs_power(x_pos=x_pos,
                                                        y_pos=y_pos,
                                                        moved_polarizer='first',
                                                        Y_ref_postion_background_column=Y_ref_postion_background_column,
                                                        bool_subtract_background=True,
                                                                                                                               fig=fig_intensity_vs_power_highpowers,
                                                                                                                               ax=ax_intensity_vs_power_highpowers)
    title = 'NEC High Power with calibration'
    ax_intensity_vs_power_highpowers.set_title(title)
    fig_intensity_vs_power_highpowers.show()
    fig_intensity_vs_power_highpowers.savefig(title)

    df_intensity_vs_power_are= pixel_plotter_NEC_highpowers.get_area_intensity_vs_power_df(x_pos=x_pos,
                                                        y_pos=y_pos,
                                                        moved_polarizer='first',
                                                        Y_ref_postion_background_column=Y_ref_postion_background_column,
                                                        bool_subtract_background=True,
                                                                                           integration_width=20,
                                                                                           bool_plot_maps=False)

    fig_area_plot, ax_area_plot = pixel_plotter_NEC_highpowers.plot_intensity_vs_power_and_linear_fit(df_x=df_intensity_vs_power_are['Power (uW)'],
                                                                                                       df_y=df_intensity_vs_power_are['Intensity'],
                                                                                                       title='NEC',
                                                                                                       x_label='Power (uW)',
                                                                                                       y_label='Intensity (arb. units)',)

    fitting_coeff_df = pixel_plotter_NEC_highpowers.get_beam_fitting_coeff_vs_power_df(x_pos=x_pos,
                                                        y_pos=y_pos,
                                                        moved_polarizer='first',
                                                        Y_ref_postion_background_column=Y_ref_postion_background_column,
                                                        bool_subtract_background=True,
                                                                                           integration_width=20,
                                                                                           bool_plot_maps=True)

    fig_area_plot.show()
    fig_area_plot.savefig(title)

    fig_spotsizes_vs_power, ax_spotsizes_vs_power = plt.subplots()
    ax_spotsizes_vs_power.plot(fitting_coeff_df['Power (uW)'], fitting_coeff_df['fwhm_x'], "o")
    ax_spotsizes_vs_power.set_xlim(1.6, 23.5)
    ax_spotsizes_vs_power.set_ylim(105-10, 125-10)
    ax_spotsizes_vs_power.set_title(title)
    ax_spotsizes_vs_power.set_xlabel('Power (uW)')
    ax_spotsizes_vs_power.set_ylabel('X FWHM (um)')
    fig_spotsizes_vs_power.savefig(title)
    fig_spotsizes_vs_power.show()
    fig_spotsizes_vs_power.savefig("spot_sizes_"+title+".pdf")