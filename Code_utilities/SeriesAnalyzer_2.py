import pandas as pd
import os
import numpy as np
import re

from Code_utilities.BeamAnalysis import BeamAnalysis
from Code_utilities.BeamCharacteristicsExtractor import BeamCharacteristicsExtractor
import matplotlib.pyplot as plt
from PowerExtractorFromPolarizer import PowerExtractorFromPolarizers


class SeriesAnalyzer_2:
    def __init__(self,
                 dir_path: str = None,
                 camera_name: str = None,
                 Y_ref_position_for_background_subtraction: int = None,
                 crop_range_x_pixels: int = None,
                 crop_range_y_pixels: int = None):

        self.crop_range_x_pixels = crop_range_x_pixels
        self.crop_range_y_pixels = crop_range_y_pixels
        self.dir_path = dir_path
        self.camera_name = camera_name
        self.Y_ref_position_for_background_subtraction = Y_ref_position_for_background_subtraction

        if self.camera_name == 'HIKMICRO':
            self.pixel_size = 12
        if self.camera_name == 'NEC':
            self.pixel_size = 23.5
        if self.camera_name == 'gentec':
            self.pixel_size = 5.5

        self._check_if_arguments_exist()

    def plot_map_in_pixels(self,
                           map_array: np.ndarray):
        """
        Plot the map array
        """

        # Calculate the extent of the image in micrometers (data shape: [rows, cols])
        rows, cols = map_array.shape
        extent = [0, cols, 0, rows]

        # Figure for Signal
        fig, ax = plt.subplots()
        im0 = ax.imshow(map_array,
                        cmap='viridis',
                        extent=extent)
        ax.set_title("Map")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        plt.colorbar(im0,
                     ax=ax)

        return fig, ax

    def plot_map_in_pixels_for_paper(self,
                           map_array: np.ndarray):
        """
        Plot the map array
        """

        # Calculate the extent of the image in micrometers (data shape: [rows, cols])
        rows, cols = map_array.shape
        extent = [0, cols, 0, rows]

        # Figure for Signal
        fig, ax = plt.subplots()
        im0 = ax.imshow(map_array,
                        cmap='viridis',
                        extent=extent)
        ax.set_title("Map")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        plt.colorbar(im0,
                     ax=ax)

        return fig, ax

    def plot_map_in_um(self,
                       map_array: np.ndarray,
                       ax=None):
        """
        Plot the map array in micrometers.
        """

        # Calculate the extent of the image in micrometers (data shape: [rows, cols])
        rows, cols = map_array.shape

        # Convert pixel dimensions to micrometers
        width_um = cols * self.pixel_size
        height_um = rows * self.pixel_size

        # Define the extent for imshow: [xmin, xmax, ymin, ymax]
        # By setting origin='lower', the (0,0) pixel of the array will be at the bottom-left
        # of the plot, and the y-axis will increase upwards, which is typical for scientific plots.
        extent_um = [0, width_um, 0, height_um]

        # Figure for Signal
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        im0 = ax.imshow(map_array,
                        cmap='viridis',
                        extent=extent_um,  # Use the calculated micrometer extent
                        origin='lower')  # Set origin to lower for standard Cartesian y-axis
        ax.set_title("Map")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        plt.colorbar(im0,
                     ax=ax)

        return fig, ax

    def _allowed_extensions(self):
        if self.camera_name == 'HIKMICRO':
            return '.csv'
        if self.camera_name == 'NEC':
            return '.csv'
        if self.camera_name == 'gentec':
            return '.txt'

    def get_beam_characterization_df(self,
                                     save_data_plot_bool: bool = False):
        filenames_list = os.listdir(self.dir_path)
        list_of_FWHM_x = []
        list_of_FWHM_y = []
        filenames_list_for_df = []
        total_intensity_list = []
        fig_obj = None
        for filename in filenames_list:
            if filename.endswith(self._allowed_extensions()) and not filename.startswith('.') and not filename.__contains__('background'):
                map_array = self._load_map_array(filename)
                fhwm_extractor = BeamCharacteristicsExtractor(map_array=map_array,
                                                              camera_name=self.camera_name,
                                                              Y_ref_position_for_background_subtraction=self.Y_ref_position_for_background_subtraction,
                                                              label=filename[:-4])

                if save_data_plot_bool:
                    fig_obj, axs_obj = fhwm_extractor.plot_analysis(bool_savel_plots=True,
                                                                    show_plot=False)
                    plt.close(fig_obj)

                FWHM_x = fhwm_extractor.fwhm_x
                FWHM_y = fhwm_extractor.fwhm_y
                total_intensity = fhwm_extractor.total_intensity_in_array

                filenames_list_for_df.append(filename[:-4])
                list_of_FWHM_x.append(FWHM_x)
                list_of_FWHM_y.append(FWHM_y)
                total_intensity_list.append(total_intensity)

        beam_characterization_df = pd.DataFrame(
            {'FWHM_x (pixels)': list_of_FWHM_x,
             'FWHM_y (pixels)': list_of_FWHM_y,
             'total_intensity': total_intensity_list,
             'filenames': filenames_list_for_df}
        )
        self._add_columns_FWHM_in_um(beam_characterization_df)

        return beam_characterization_df

    def plot_all_maps(self,
                      save_data_plot_bool: bool = True,
                      normalization_factor=None, ):

        filenames_list = os.listdir(self.dir_path)

        for filename in filenames_list:
            if filename.endswith('.csv') and not filename.startswith('.') and not filename.__contains__('background'):
                map_array = self._load_map_array(filename)
                if normalization_factor is not None:
                    map_array = map_array / normalization_factor
                fig, ax = self.plot_map_in_pixels(map_array)
                if save_data_plot_bool:
                    if normalization_factor is not None:
                        fig.savefig(f"{filename[:-4]}_normalized.pdf")
                    else:
                        fig.savefig(f"{filename[:-4]}.pdf")
                    plt.close(fig)

    def plot_all_maps_for_paper(self,
                      save_data_plot_bool: bool = True,
                      normalize=None, ):

        filenames_list = os.listdir(self.dir_path)

        for filename in filenames_list:
            if filename.endswith('.csv') and not filename.startswith('.') and not filename.__contains__('background'):
                map_array = self._load_map_array(filename)
                if normalize is True:
                    map_array = map_array / np.max(map_array)
                fig, ax = self.plot_map_in_pixels_for_paper(map_array)
                if save_data_plot_bool:
                    if normalize is not None:
                        fig.savefig(f"{filename[:-4]}_normalized.pdf")
                    else:
                        fig.savefig(f"{filename[:-4]}.pdf")
                plt.close(fig)

    def plot_all_map_analysis(self,
                      save_data_plot_bool: bool = True):

        filenames_list = os.listdir(self.dir_path)

        for filename in filenames_list:
            if filename.endswith('.csv') and not filename.startswith('.') and not filename.__contains__('background'):
                map_array = self._load_map_array(filename)
                map_characteristics = BeamCharacteristicsExtractor(map_array=map_array,
                                                              camera_name=self.camera_name,
                                                              Y_ref_position_for_background_subtraction=self.Y_ref_position_for_background_subtraction,
                                                              label=filename[:-4])
                fig, ax = map_characteristics.plot_analysis(bool_savel_plots=False,
                                                                show_plot=False)
                if save_data_plot_bool:
                    fig.savefig(f"{filename[:-4]}_BeamAnalysis.pdf")
                    plt.close(fig)

    def plot_all_map_side_by_side(self):
        filenames_list = os.listdir(self.dir_path)
        # Filter filenames
        filtered_filenames = [
            f for f in filenames_list
            if f.endswith('.csv') and not f.startswith('.') and 'background' not in f
        ]
        n = len(filtered_filenames)
        print(f'filtered_filenames: {filtered_filenames}')

        if n == 0:
            print("No maps to plot.")
            return

        # Create a single figure with n subplots in one row
        fig, axes = plt.subplots(1,
                                 n,
                                 figsize=(n * 5, 5))  # Adjust figsize as needed
        # Ensure axes is a list even if n=1
        if n == 1:
            axes = [axes]

        # Plot each map on its corresponding axis
        for i, filename in enumerate(filtered_filenames):
            map_array = self._load_map_array(filename)
            self.plot_map_in_um(map_array,
                                ax=axes[i])

        plt.show()
        return fig, axes




    def plot_all_normalized_maps(self,
                                 save_data_plot_bool: bool = True,
                                 normalization_factor_df=None):

        if normalization_factor_df is None or not isinstance(normalization_factor_df,
                                                             pd.DataFrame):
            print("Error: normalization_factor_df must be a pandas DataFrame.")
            return
        if not {'Signal_Column', 'Integral_Value'}.issubset(normalization_factor_df.columns):
            print("Error: normalization_factor_df must contain 'Signal_Column' and 'Integral_Value' columns.")
            return

        filenames_list = os.listdir(self.dir_path)
        print(f'Available normalization signals (unique): {normalization_factor_df["Signal_Column"].unique().tolist()}')

        for filename_with_ext in filenames_list:
            if filename_with_ext.endswith('.csv') and \
                    not filename_with_ext.startswith('.') and \
                    not filename_with_ext.__contains__('background'):

                base_filename = filename_with_ext[:-4]  # Remove .csv extension
                base_filename=base_filename.replace(',', '.')


                # Find the normalization factor for the current file
                norm_row = normalization_factor_df[normalization_factor_df['Signal_Column'] == base_filename]
                print(f'Base filename: {base_filename}')
                print(f'the filename in df is{norm_row["Signal_Column"]}')
                print(f'the normalization factor for this file: {norm_row}')

                if norm_row.empty:
                    print(f"Warning: No normalization factor found for {base_filename}. Skipping.")
                    continue

                normalization_factor = norm_row['Integral_Value'].iloc[0]

                if normalization_factor == 0:
                    print(f"Warning: Normalization factor for {base_filename} is 0. Skipping to avoid division by "
                          f"zero.")
                    continue

                map_array = self._load_map_array(filename_with_ext)
                normalized_map_array = map_array / normalization_factor

                map_characteristics = BeamCharacteristicsExtractor(map_array=normalized_map_array,
                                                              camera_name=self.camera_name,
                                                              Y_ref_position_for_background_subtraction=self.Y_ref_position_for_background_subtraction,
                                                              label=filename_with_ext[:-4])

                fig, axs = map_characteristics.plot_analysis(bool_savel_plots=False,
                                                            show_plot=False)

                if save_data_plot_bool:
                    fig.savefig(f"{base_filename}_normalized.pdf")

                # Decide if you want to show plots interactively or just save them
                # If running in a script and saving many plots, you might not want to call plt.show()
                # or ensure plots are closed after saving to prevent memory issues.
                # For now, let's assume we close them after saving as in plot_all_maps
                plt.close(fig)

    def _get_lists_of_temperature_and_powers_maps(self,
                                                  save_data_plot_bool: bool = False,
                                                  known_angle=None,
                                                  known_voltage_at_known_angle_in_V=None,
                                                  moved_polarizer=None):

        filenames_list = os.listdir(self.dir_path)
        list_of_power_map_arrays = []
        list_of_temperature_maps = []
        for filename in filenames_list:
            if filename.endswith('.csv') and not filename.startswith('.') and not filename.__contains__('background'):
                power = self._extract_power_from_filename(filename=filename,
                                                          known_angle=known_angle,
                                                          known_voltage_at_known_angle_in_V=known_voltage_at_known_angle_in_V,
                                                          moved_polarizer=moved_polarizer)
                map_array = self._load_map_array(filename)
                map_array_Y_bg_subtracted = BeamCharacteristicsExtractor(map_array=map_array,
                                                                         camera_name=self.camera_name,
                                                                         Y_ref_position_for_background_subtraction=self.Y_ref_position_for_background_subtraction,
                                                                         label=filename[:-4]).map_array
                power_map_array = self._get_power_map_array(array_map=map_array_Y_bg_subtracted,
                                                            power=power)
                power_map_analysis = BeamCharacteristicsExtractor(map_array=power_map_array,
                                                                  camera_name=self.camera_name,
                                                                  Y_ref_position_for_background_subtraction=self.Y_ref_position_for_background_subtraction,
                                                                  label=filename[:-4])
                list_of_power_map_arrays.append(power_map_array)
                list_of_temperature_maps.append(map_array_Y_bg_subtracted)

                if save_data_plot_bool:
                    fig_obj, axs_obj = power_map_analysis.plot_analysis(bool_savel_plots=True,
                                                                        show_plot=False)
                    plt.close(fig_obj)

        return list_of_power_map_arrays, list_of_temperature_maps

    def compute_slope_map(self,
                          save_data_plot_bool: bool = False,
                          known_angle=None,
                          known_voltage_at_known_angle_in_V=None,
                          moved_polarizer=None):
        """
        Compute the slope of temperature vs. power for each pixel.

        Parameters:
        list_of_power_map_arrays : list of 2D numpy arrays
            Each array represents the incident power per pixel for a measurement.
        list_of_temperature_maps : list of 2D numpy arrays
            Each array represents the temperature (intensity) per pixel for a measurement.

        Returns:
        slope_map : 2D numpy array
            The slope of temperature vs. power for each pixel.
        """
        list_of_power_map_arrays, list_of_temperature_maps = self._get_lists_of_temperature_and_powers_maps(
            save_data_plot_bool=save_data_plot_bool,
            known_angle=known_angle,
            known_voltage_at_known_angle_in_V=known_voltage_at_known_angle_in_V,
            moved_polarizer=moved_polarizer)

        # Check that both lists have the same number of arrays
        if len(list_of_power_map_arrays) != len(list_of_temperature_maps):
            raise ValueError("Both lists must have the same number of arrays.")

        # Check that all arrays have the same shape
        shapes = [arr.shape for arr in list_of_power_map_arrays + list_of_temperature_maps]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All arrays must have the same shape.")

        N = len(list_of_power_map_arrays)
        power_stack = np.stack(list_of_power_map_arrays,
                               axis=0)
        temp_stack = np.stack(list_of_temperature_maps,
                              axis=0)

        sum_power = np.sum(power_stack,
                           axis=0)
        sum_temp = np.sum(temp_stack,
                          axis=0)
        sum_power_temp = np.sum(power_stack * temp_stack,
                                axis=0)
        sum_power2 = np.sum(power_stack ** 2,
                            axis=0)

        numerator = N * sum_power_temp - sum_power * sum_temp
        denominator = N * sum_power2 - sum_power ** 2

        # Avoid division by zero by setting slope to NaN where denominator is zero
        slope_map = np.where(denominator != 0,
                             numerator / denominator,
                             np.nan)

        return slope_map

    def get_beam_char_df_w_powers(self,
                                  save_data_plot_bool: bool = False,
                                  known_angle=None,
                                  known_voltage_at_known_angle_in_V=None,
                                  moved_polarizer=None,
                                  beam1_over_e2_width = None):

        beam_characterization_df = self.get_beam_characterization_df(save_data_plot_bool=save_data_plot_bool)
        beam_characterization_df_with_angles = self._generate_angle_column(beam_characterization_df)

        beam_characterization_df_with_powers = self._generate_powers_column(beam_characterization_df_with_angles,
                                                                            known_angle=known_angle,
                                                                            known_voltage_at_known_angle_in_V=known_voltage_at_known_angle_in_V,
                                                                            moved_polarizer=moved_polarizer)
        if beam1_over_e2_width is not None:
            uW_to_mW = 1e-3
            converter_factor = uW_to_mW*2/(np.pi*(beam1_over_e2_width*10**-4)**2)
            beam_characterization_df_with_powers['fluence (mW/cm^2)'] =beam_characterization_df_with_powers['Power (uW)']*converter_factor

        return beam_characterization_df_with_powers

    def get_FWHM_df_w_positions(self,
                                save_data_plot_bool: bool = False):

        FWHM_df = self.get_beam_characterization_df(save_data_plot_bool=save_data_plot_bool)

        FWHM_df["position (mm)"] = (
            FWHM_df["filenames"]
            .str.replace(",",
                         ".",
                         regex=False)
            .str.removesuffix(" mm")
            .astype(float)
        )
        return FWHM_df

    def plot_FWHM_vs_position(self):
        FWHM_df = self.get_FWHM_df_w_positions()
        fig, ax = plt.subplots()
        ax.plot(FWHM_df["position (mm)"],
                FWHM_df["FWHM_x (um)"],
                label="FWHM_x",
                marker="o",
                linestyle="None")
        ax.plot(FWHM_df["position (mm)"],
                FWHM_df["FWHM_y (um)"],
                label="FWHM_y",
                marker="o",
                linestyle="None")
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("FWHM (um)")
        ax.legend()
        return fig, ax

    def _get_power_map_array(self,
                             array_map,
                             power):
        total_temperature = np.sum(array_map)
        print (total_temperature)
        power_map_array = array_map * power / total_temperature
        return power_map_array

    def _load_map_array(self,
                        filename):

        beam_analysis = BeamAnalysis(dir_path=self.dir_path,
                                     signal_filename=filename,
                                     camera_name=self.camera_name,
                                     crop_range_x_pixels=self.crop_range_x_pixels,
                                     crop_range_y_pixels=self.crop_range_y_pixels)
        map_array = beam_analysis.map_array

        return map_array

    def _check_if_arguments_exist(self):
        if any(arg is None for arg in
               [self.dir_path,
                self.camera_name,
                self.crop_range_x_pixels,
                self.crop_range_y_pixels]):
            raise ValueError("All arguments (dir_path, camera_name, "
                             "crop_range_x_pixels, crop_range_y_pixels) must be provided and not be None.")

    def _add_columns_FWHM_in_um(self,
                                FWHM_df):

        FWHM_df["FWHM_x (um)"] = FWHM_df["FWHM_x (pixels)"] * self.pixel_size
        FWHM_df["FWHM_y (um)"] = FWHM_df["FWHM_y (pixels)"] * self.pixel_size
        return FWHM_df

    def _generate_angle_column(self,
                               fitting_coefficients_df):
        """
        Extract the numeric angle from the *filename* column and append it as a
        new column called ``angle``.

        The expected filename pattern is something like
        ``IR_00045_50degrees.csv``.  A regular expression captures the digits
        immediately before the literal word ``degrees``.  Any filename that
        does not match yields ``NaN`` in the *angle* column.
        """
        if "filenames" not in fitting_coefficients_df.columns:
            raise KeyError(
                "'filenames' column not found — cannot extract angle values."
            )

        regex_pattern = r"(\d+)\s?degrees"

        angle = (
            fitting_coefficients_df["filenames"]
            .str.extract(regex_pattern,
                         expand=False)
            .astype(float)  # convert to numeric; NaN for non‑matches
        )

        fitting_coefficients_df = fitting_coefficients_df.assign(angle=angle)
        return fitting_coefficients_df

    def _generate_powers_column(self,
                                fitting_coefficients_df,
                                known_angle=None,
                                known_voltage_at_known_angle_in_V=None,
                                moved_polarizer: str = None):

        fitting_coefficients_df["Power (uW)"] = PowerExtractorFromPolarizers(known_angle=known_angle,
                                                                             known_voltage_at_known_angle_in_V=known_voltage_at_known_angle_in_V,
                                                                             desired_angle=fitting_coefficients_df[
                                                                                 "angle"].values,
                                                                             moved_polarizer=moved_polarizer).power_at_angle_uW

        return fitting_coefficients_df

    def _extract_power_from_filename(self,
                                     filename: str = None,
                                     known_angle=None,
                                     known_voltage_at_known_angle_in_V=None,
                                     moved_polarizer=None):

        match = re.search(r"(\d+(?:\.\d+)?)\s*degrees",
                          filename,
                          flags=re.IGNORECASE)
        angle = float(match.group(1)) if match else np.nan

        power = PowerExtractorFromPolarizers(known_angle=known_angle,
                                             known_voltage_at_known_angle_in_V=known_voltage_at_known_angle_in_V,
                                             desired_angle=angle,
                                             moved_polarizer=moved_polarizer).power_at_angle_uW

        return power

    def _linear_model(self,
                      x,
                      a,
                      b):
        """
        A linear model function: y = a*x + b.

        Args:
            x (np.ndarray or float): Independent variable.
            a (float): Slope.
            b (float): Intercept.

        Returns:
            np.ndarray or float: Dependent variable y.
        """
        y = a * x + b
        return y
