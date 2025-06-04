import matplotlib
matplotlib.use('macosx')
import pandas as pd
import os
from PowerExtractorFromPolarizer import PowerExtractorFromPolarizers
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from BeamAnalysis import BeamAnalysis
from TwoDArrayAnalysis import TwoDArrayAnalysis


class PixelPlotter:
    def __init__(self,
                 maps_dir_path: str,
                 camera_name: str = None,
                 known_angle: float = None,
                 known_voltage_at_known_angle_in_V: float = None):
        self.maps_dir_path = maps_dir_path
        self.camera_name = camera_name
        self.known_angle = known_angle
        self.known_voltage_at_known_angle_in_V = known_voltage_at_known_angle_in_V
        if self.camera_name == 'HIKMICRO':
            self.pixel_size_um = 13
        if self.camera_name == 'NEC':
            self.pixel_size_um = 23.5

    def get_single_map(self,
                       map_filename: str, ):

        map_array = BeamAnalysis(dir_path=self.maps_dir_path,
                           signal_filename=map_filename,
                           camera_name=self.camera_name).map_array


        return map_array

    def get_single_pixel_intensity(self,
                                   x_pos: int,
                                   y_pos: int,
                                   map_filename: str, ):

        map_array = BeamAnalysis(dir_path=self.maps_dir_path,
                           signal_filename=map_filename,
                           camera_name=self.camera_name).map_array
        intensity = map_array[x_pos, y_pos]

        return intensity

    def get_single_pixel_intensity_vs_power_df(self,
                                               x_pos: int,
                                               y_pos: int,
                                               moved_polarizer: str,
                                               bool_subtract_background: bool = False,
                                               Y_ref_postion_background_column: float = None):

        list_of_paths = os.listdir(self.maps_dir_path)
        list_of_intensities = []
        list_of_powers = []
        self.x_pos = x_pos
        self.y_pos = y_pos

        for filename in list_of_paths:
            if filename.endswith('.csv') and not filename.startswith('.') and not filename.__contains__('background'):
                map_array = BeamAnalysis(dir_path=self.maps_dir_path,
                                         signal_filename=filename,
                                         camera_name=self.camera_name).map_array
                if bool_subtract_background:
                    map_array = self.subtract_background(Y_ref_postion_background_column=Y_ref_postion_background_column,
                                                        map_array=map_array)

                intensity = map_array[x_pos, y_pos]
                angle = self.get_angle(filename)
                power = PowerExtractorFromPolarizers(known_angle=self.known_angle,
                                                     known_voltage_at_known_angle_in_V=self.known_voltage_at_known_angle_in_V,
                                                     desired_angle=angle,
                                                     moved_polarizer=moved_polarizer).power_at_angle_uW

                list_of_intensities.append(intensity)
                list_of_powers.append(power)

        power_vs_pixel_df = pd.DataFrame({"Intensity": list_of_intensities,
                                          "Power (uW)": list_of_powers})

        return power_vs_pixel_df

    def get_area_intensity_vs_power_df(self,
                                               x_pos: int,
                                               y_pos: int,
                                               moved_polarizer: str,
                                               bool_subtract_background: bool = False,
                                               Y_ref_postion_background_column: float = None,
                                                integration_width: int = None,
                                       bool_plot_maps: bool = False):
        if integration_width is None:
            raise ValueError("integration_width is required")
        list_of_paths = os.listdir(self.maps_dir_path)
        list_of_intensities = []
        list_of_powers = []
        self.x_pos = x_pos
        self.y_pos = y_pos

        for filename in list_of_paths:
            if filename.endswith('.csv') and not filename.startswith('.') and not filename.__contains__('background'):
                map_array = BeamAnalysis(dir_path=self.maps_dir_path,
                                         signal_filename=filename,
                                         camera_name=self.camera_name).map_array
                if bool_subtract_background:
                    map_array = self.subtract_background(Y_ref_postion_background_column=Y_ref_postion_background_column,
                                                        map_array=map_array)


                intensity = self.get_intensity_in_area(x_pos=x_pos,
                                                      y_pos=y_pos,
                                                      map_array=map_array,
                                                      width=integration_width,
                                                       bool_plot_maps=bool_plot_maps,
                                                       title=filename[:-4])
                angle = self.get_angle(filename)
                power = PowerExtractorFromPolarizers(known_angle=self.known_angle,
                                                     known_voltage_at_known_angle_in_V=self.known_voltage_at_known_angle_in_V,
                                                     desired_angle=angle,
                                                     moved_polarizer=moved_polarizer).power_at_angle_uW

                list_of_intensities.append(intensity)
                list_of_powers.append(power)

        power_vs_pixel_df = pd.DataFrame({"Intensity": list_of_intensities,
                                          "Power (uW)": list_of_powers})

        return power_vs_pixel_df

    def get_beam_fitting_coeff_vs_power_df(self,
                                           x_pos: int,
                                           y_pos: int,
                                           moved_polarizer: str,
                                           bool_subtract_background: bool = False,
                                           Y_ref_postion_background_column: float = None,
                                           integration_width: int = None,
                                           bool_plot_maps: bool = False):
        if integration_width is None:
            raise ValueError("integration_width is required")

        list_of_paths = os.listdir(self.maps_dir_path)
        list_of_intensities = []
        list_of_powers = []
        list_of_fitting_coeff = []
        self.x_pos = x_pos
        self.y_pos = y_pos

        for filename in list_of_paths:
            if filename.endswith('.csv') and not filename.startswith('.') and not filename.__contains__('background'):
                map_array = BeamAnalysis(dir_path=self.maps_dir_path,
                                         signal_filename=filename,
                                         camera_name=self.camera_name).map_array
                if bool_subtract_background:
                    map_array = self.subtract_background(Y_ref_postion_background_column=Y_ref_postion_background_column,
                                                         map_array=map_array)

                intensity = self.get_intensity_in_area(x_pos=x_pos,
                                                       y_pos=y_pos,
                                                       map_array=map_array,
                                                       width=integration_width,
                                                       bool_plot_maps=bool_plot_maps,
                                                       title=filename[:-4])
                fitting_coeff = self.get_2D_fit_coefficients(x_pos=x_pos,
                                                             y_pos=y_pos,
                                                             map_array=map_array,
                                                             width=integration_width,
                                                             bool_plot_maps=bool_plot_maps,
                                                             title=filename[:-4])

                angle = self.get_angle(filename)
                power = PowerExtractorFromPolarizers(known_angle=self.known_angle,
                                                     known_voltage_at_known_angle_in_V=self.known_voltage_at_known_angle_in_V,
                                                     desired_angle=angle,
                                                     moved_polarizer=moved_polarizer).power_at_angle_uW

                list_of_fitting_coeff.append(fitting_coeff)
                list_of_intensities.append(intensity)
                list_of_powers.append(power)

        # Check for alignment between lists
        if len(list_of_fitting_coeff) != len(list_of_intensities):
            raise ValueError("Mismatch in number of fitting coefficients and intensities")

        # Create initial DataFrame with intensities and powers
        power_vs_pixel_df = pd.DataFrame({"Intensity": list_of_intensities,
                                          "Power (uW)": list_of_powers})

        # Create DataFrame from list of fitting coefficients
        fitting_coeff_df = pd.DataFrame(list_of_fitting_coeff)

        # Concatenate the two DataFrames along columns
        power_vs_pixel_df = pd.concat([power_vs_pixel_df, fitting_coeff_df],
                                      axis=1)

        # Define a specific column order for clarity
        columns_order = ["Power (uW)", "Intensity", "amplitude", "xo", "yo",
                         "sigma_x", "sigma_y", "fwhm_x", "fwhm_y", "offset"]

        # Return the DataFrame with ordered columns
        return power_vs_pixel_df[columns_order]

    def get_2D_fit_coefficients(self,
                              x_pos: int,
                              y_pos: int,
                              map_array: np.ndarray,
                              width: int,
                              bool_plot_maps: bool = False,
                              title: str = None):# Or np.integer, np.floating, depending on map_array.dtype

        if width <= 0:
            # Or raise ValueError("Width must be a positive integer")
            return 0.0

        # Use integer division to find half the width
        half_w = width // 2

        # Calculate start and end indices for the slice
        # This method ensures (x_pos, y_pos) is the center for odd widths
        # and appropriately positioned for even widths.
        # The slice will have 'width' elements.
        start_x = x_pos - half_w
        end_x = start_x + width  # end_x will be exclusive in slicing

        start_y = y_pos - half_w
        end_y = start_y + width  # end_y will be exclusive in slicing

        # Get array dimensions
        array_height, array_width = map_array.shape

        # Clip the calculated slice boundaries to the actual array dimensions
        # This handles cases where the desired square is partially or fully outside the array
        actual_start_x = max(0, start_x)
        actual_end_x = min(array_height, end_x) # Use array_height (shape[0])
        actual_start_y = max(0, start_y)
        actual_end_y = min(array_width, end_y)   # Use array_width (shape[1])

        # If the clipped region has no size (e.g., it's entirely outside), return 0
        if actual_start_x >= actual_end_x or actual_start_y >= actual_end_y:
            return 0.0
        cropped_map = map_array[actual_start_x:actual_end_x, actual_start_y:actual_end_y]
        if bool_plot_maps:
            fig, ax = self.plot_map(map_data=cropped_map,
                          title=title)
            fig.savefig(title + '.pdf')

        _, fitting_coefficients, _, _ = TwoDArrayAnalysis(cropped_map,
                                                      camera_name=self.camera_name).fit_gaussian()

        return fitting_coefficients # Ensure float return for consistency
    def get_intensity_in_area(self,
                              x_pos: int,
                              y_pos: int,
                              map_array: np.ndarray,
                              width: int,
                              bool_plot_maps: bool = False,
                              title: str = None) -> float: # Or np.integer, np.floating, depending on map_array.dtype
        """
        Calculates the sum of elements in a square region of map_array.

        The square region has a side length of 'width' and is intended to be
        centered at the pixel (x_pos, y_pos).
        - If 'width' is odd, (x_pos, y_pos) will be the exact center pixel of the square.
        - If 'width' is even, (x_pos, y_pos) will be the top-left pixel of the
          bottom-right quadrant of the 2x2 central pixels (i.e., one of the four
          pixels closest to the geometric center).

        Args:
            x_pos: The row index for the center of the square.
            y_pos: The column index for the center of the square.
            map_array: The 2D numpy array.
            width: The side length of the square region (must be positive).

        Returns:
            The sum of the elements within the defined square region.
            Returns 0.0 if width is non-positive or if the region
            is entirely outside the map_array.
        """
        if width <= 0:
            # Or raise ValueError("Width must be a positive integer")
            return 0.0

        # Use integer division to find half the width
        half_w = width // 2

        # Calculate start and end indices for the slice
        # This method ensures (x_pos, y_pos) is the center for odd widths
        # and appropriately positioned for even widths.
        # The slice will have 'width' elements.
        start_x = x_pos - half_w
        end_x = start_x + width  # end_x will be exclusive in slicing

        start_y = y_pos - half_w
        end_y = start_y + width  # end_y will be exclusive in slicing

        # Get array dimensions
        array_height, array_width = map_array.shape

        # Clip the calculated slice boundaries to the actual array dimensions
        # This handles cases where the desired square is partially or fully outside the array
        actual_start_x = max(0, start_x)
        actual_end_x = min(array_height, end_x) # Use array_height (shape[0])
        actual_start_y = max(0, start_y)
        actual_end_y = min(array_width, end_y)   # Use array_width (shape[1])

        # If the clipped region has no size (e.g., it's entirely outside), return 0
        if actual_start_x >= actual_end_x or actual_start_y >= actual_end_y:
            return 0.0
        cropped_map = map_array[actual_start_x:actual_end_x, actual_start_y:actual_end_y]
        if bool_plot_maps:
            fig, ax = self.plot_map(map_data=cropped_map,
                          title=title)
            fig.savefig(title + '.pdf')

        # Extract the region of interest (ROI)
        roi = map_array[actual_start_x:actual_end_x, actual_start_y:actual_end_y]

        # Sum the elements in the ROI
        intensity = np.sum(roi)

        return float(intensity) # Ensure float return for consistency
    def subtract_background(self,
                            Y_ref_postion_background_column: float = None,
                            map_array: np.array= None,):

        column_position_for_average = int(Y_ref_postion_background_column)
        x=self.x_pos
        map_array = map_array - np.average(map_array[x, 0:column_position_for_average+1])

        return map_array

    def plot_intensity_vs_power_and_linear_fit(self,
                                               df_x,
                                               df_y,
                                               title,
                                               x_label: str = 'Power (uW)',
                                               y_label: str = 'Intensity (arb. units)',
                                               fig=None,
                                               ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        # Plot the data points
        ax.plot(df_x, df_y, 'o', label='Data')

        # Fit a linear function (y = a * x + b)
        popt, _ = curve_fit(self.linear_model, df_x, df_y, p0=[0, 0])
        a, b = popt
        print(f"Fit coefficients: a = {a:.4e}, b = {b:.4e}")

        # Generate points for the linear fit line
        x_fit = np.linspace(df_x.min(), df_x.max(), 100)
        y_fit = self.linear_model(x_fit, a, b)

        # Plot the linear fit
        ax.plot(x_fit, y_fit, 'r-', label='fit')

        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()

        return fig, ax

    @staticmethod
    def linear_model(x, a, b):
        return a * x + b


    def get_angle(self,
                  filename):
        """
        Extract the numeric angle from the *filename* column and append it as a
        new column called ``angle``.

        The expected filename pattern is something like
        ``IR_00045_50degrees.csv``.  A regular expression captures the digits
        immediately before the literal word ``degrees``.  Any filename that
        does not match yields ``NaN`` in the *angle* column.
        """
        # regex_pattern = r"(\d+)\s?degrees"
        #
        # angle = (
        #     filename.str.extract(regex_pattern,
        #                  expand=False)
        #     .astype(float)  # convert to numeric; NaN for non‑matches
        # )
        match = re.search(r"(\d+(?:\.\d+)?)\s*degrees",
                          filename,
                          flags=re.IGNORECASE)

        return float(match.group(1)) if match else np.nan

    @staticmethod
    def quadratic_no_linear(x,
                            a,
                            c):
        """
        Custom quadratic function without the linear term: y = a*x^2 + c
        """
        return a * x ** 2 + c

    def get_quadratic_fit_coefficients(self,
                                       x: np.ndarray,
                                       y: np.ndarray):
        """
        Compute coefficients for the custom quadratic fit y = a*x^2 + c.

        Args:
            x (np.ndarray): Input x values (e.g., power in uW).
            y (np.ndarray): Input y values (e.g., intensity in counts).

        Returns:
            tuple or None: Coefficients (a, c) for y = a*x^2 + c, or None if fit fails.
        """
        if len(x) >= 2 and len(y) >= 2:  # Need at least 2 points for 2 parameters
            try:
                # Fit the custom quadratic model to the data
                popt, _ = curve_fit(self.quadratic_no_linear,
                                    x,
                                    y)
                a, c = popt
                print(f"Fit coefficients: a = {a:.4e}, c = {c:.4e}")
                return a, c
            except Exception as e:
                print(f"Fit failed: {e}")
                return None
        else:
            print("Not enough data points (need at least 2).")
            return None

    def plot_intensity_vs_power(self,
                                x_pos: int,
                                y_pos: int,
                                moved_polarizer: str,
                                fig=None,
                                ax=None,
                                bool_subtract_background: bool = False,
                                Y_ref_postion_background_column: float = None):
        """
        Plot intensity vs power for a given pixel with the fit y = a*x^2 + c.

        Args:
            x_pos (int): X-coordinate of the pixel.
            y_pos (int): Y-coordinate of the pixel.

        Returns:
            tuple: Matplotlib figure and axes objects (fig, ax).
        """
        # Assuming this method retrieves data from a DataFrame
        power_vs_pixel_df = self.get_single_pixel_intensity_vs_power_df(x_pos=x_pos,
                                                                        y_pos=y_pos,
                                                                        moved_polarizer=moved_polarizer,
                                                                        bool_subtract_background=bool_subtract_background,
                                                                        Y_ref_postion_background_column=Y_ref_postion_background_column)
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # Extract power and intensity as NumPy arrays
        power = power_vs_pixel_df['Power (uW)'].values
        intensity = power_vs_pixel_df['Intensity'].values

        # Get the custom quadratic fit coefficients
        coefficients = self.get_quadratic_fit_coefficients(power,
                                                           intensity)
        coefficients=None
        #coefficients =self.get_two_lines_fitting_coefficients(power,intensity, threshold)

        # Plot the fit if coefficients are available
        if coefficients is not None:
            a, c = coefficients
            power_fit = np.linspace(power.min(),
                                    power.max(),
                                    100)  # Smooth curve
            intensity_fit = self.quadratic_no_linear(power_fit,
                                                     a,
                                                     c)
            ax.plot(power_fit,
                    intensity_fit,
                    linestyle='-',
                    color='red',
                    label=f'Fit ($y = {a:.2e}x^2 + {c:.2e}$)')

        # Plot the data points
        ax.plot(power,
                intensity,
                marker='o',
                linestyle='None',
                label='Data')

        # Add labels and legend
        ax.set_title(f'Intensity vs Power for Pixel ({x_pos}, {y_pos})')
        ax.set_xlabel('Power (uW)')
        ax.set_ylabel('Intensity (counts)')
        ax.legend()

        return fig, ax

    def get_power_adjusted_map(self,
                               map_array: np.array,
                               x_pos: int,
                               y_pos: int,
                               moved_polarizer: str):
        power_vs_pixel_df = self.get_single_pixel_intensity_vs_power_df(x_pos=x_pos,
                                                                        y_pos=y_pos,
                                                                        moved_polarizer=moved_polarizer)

        # Extract power and intensity as NumPy arrays
        power = power_vs_pixel_df['Power (uW)'].values
        intensity = power_vs_pixel_df['Intensity'].values

        # Get quadratic fit coefficients
        a, c = self.get_quadratic_fit_coefficients(power,
                                                   intensity)  # Coefficients (a, b, c) for y = ax^2 + c, or None if


        # Guard against divide‑by‑zero and negative values inside sqrt
        with np.errstate(divide="ignore",
                         invalid="ignore"):
            adjusted_map = np.sqrt((map_array - c) / a)

        return adjusted_map

    def plot_adjusted_map(self,
                          map_filename: str,
                          x_pos_calibration: int,
                          y_pos_calibration: int):

        map_array = self.get_single_map(map_filename=map_filename)
        adjusted_map = self.get_power_adjusted_map(map_array=map_array,
                                                   x_pos=x_pos_calibration,
                                                   y_pos=y_pos_calibration)

        fig, ax = self.plot_map(map_data=adjusted_map)
        ax.set_title('Power Adjusted Intensity Map')

        return fig, ax, adjusted_map

    def plot_map(self,
                 map_data,
                 fig=None,
                 ax=None,
                 title: str = 'Intensity Map',
                 bool_save_map: bool = False,):
        """
        Plot the intensity map using matplotlib.
        The color scale is automatically adjusted using percentiles to enhance visibility.
        """

        if map_data is None:
            load_method_suggestion = f"load_data() for camera '{self.camera_name}'"
            raise ValueError(f"Data not loaded. Call {load_method_suggestion} first.")

        if not isinstance(map_data,
                          np.ndarray) or not np.issubdtype(map_data.dtype,
                                                           np.number):
            try:
                map_data = np.nan_to_num(np.array(map_data,
                                                  dtype=float),
                                         nan=0.0)
            except ValueError as e:
                raise ValueError(f"Data is not purely numeric and could not be converted. Error: {e}. "
                                 "Please check the loaded data content.")

        if map_data.size == 0:  # Handle empty data array after loading/conversion
            print("Warning: Data array is empty. Cannot plot intensity map.")
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
            ax.set_title(title if title != 'Intensity Map' else f'{self.camera_name} Intensity Map (No Data)')
            ax.text(0.5,
                    0.5,
                    "No data to display",
                    ha="center",
                    va="center")

            return fig, ax

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()  # Get the figure if ax is provided

        im = ax.imshow(map_data,
                       cmap='viridis',
                       interpolation='nearest')
        # vmin=vmin,
        # # Set explicit min for color scale
        # vmax=vmax)  # Set explicit max for color scale

        ax.set_title(title if title != 'Intensity Map' else f'{self.camera_name} Intensity Map')

        plt.colorbar(im,
                     # Use plt.colorbar directly
                     ax=ax,
                     label='Intensity')
        if bool_save_map:
            plt.savefig(f'{title}.png')
        return fig, ax

    def get_x_cross_section(self,
                            map: np.array,
                            x_pos: int):

        return map[x_pos, :]

    def plot_x_cross_section_um(self,
                                map_array: np.array,
                                x_pos: int,
                                fig=None,
                                ax=None,
                                title: str = 'Normalized cross section plot',
                                label: str = None,
                                background_ref_Y_position_um: int = None,
                                x_axis_shift_in_um: float = 0.0):
        if ax is None:
            fig, ax = plt.subplots()

        x_cross_section = self.get_x_cross_section(map=map_array,
                                                   x_pos=x_pos)

        fig_2, ax_2 = plt.subplots()
        ax_2.plot(np.arange(len(x_cross_section)), x_cross_section)
        fig_2.show()

        if background_ref_Y_position_um is not None:
            column_position_for_average = int(background_ref_Y_position_um / self.pixel_size_um)
            print(f"Background column position: {column_position_for_average}")
            background = np.average(x_cross_section[:column_position_for_average])
            x_cross_section_background_subtracted = x_cross_section - background
            x_cross_section_normalized = x_cross_section_background_subtracted/np.max(x_cross_section_background_subtracted)
        else:
            x_cross_section_normalized = x_cross_section / np.max(x_cross_section)

        # Build the original spatial axis (in µm)
        x_plot_axis = np.arange(len(x_cross_section_normalized)) * self.pixel_size_um
        # ------------------------------------------------------------------
        # Shift the axis so the peak (maximum) is at 0 µm
        # ------------------------------------------------------------------
        peak_index = int(np.argmax(x_cross_section_normalized))
        peak_position_um = x_plot_axis[peak_index]
        x_plot_axis = x_plot_axis - peak_position_um

        x_plot_axis = x_plot_axis + x_axis_shift_in_um

        ax.plot(x_plot_axis,
                x_cross_section_normalized,
                label=label)
        ax.set_xlabel('Y Position (um)')
        ax.set_title(title)
        ax.legend()

        return fig, ax


if __name__ == "__main__":
    dir_path = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz '
        'Camera/20250523/NEC/HighPowerRange/AllMeasurements')

    pixel_plotter_NEC = PixelPlotter(maps_dir_path=dir_path,
                                     camera_name='NEC',
                                     known_angle=90,
                                     known_voltage_at_known_angle_in_V=0.17)

    map_array_NEC = pixel_plotter_NEC.get_single_map(map_filename='60 degrees.csv')
    fig, ax = pixel_plotter_NEC.plot_map(map_data=map_array_NEC)
    x_pos = 129
    y_pos = 145
    power_vs_pixel_df = pixel_plotter_NEC.get_single_pixel_intensity_vs_power_df(x_pos=x_pos,
                                                                                 y_pos=y_pos,
                                                                                 moved_polarizer='second')
    fig, ax = pixel_plotter_NEC.plot_intensity_vs_power(x_pos=x_pos,
                                                        y_pos=y_pos,
                                                        moved_polarizer='second')
    fig.show()

    # fig_adjusted, ax_adjusted, adjusted_map_array = pixel_plotter_NEC.plot_adjusted_map(map_filename='90 degrees.csv',
    #                                                                                     x_pos_calibration=129,
    #                                                                                     y_pos_calibration=145)
    #
    # fig_adjusted.show()

    adjusted_map_array = pixel_plotter_NEC.get_power_adjusted_map(map_array=map_array_NEC,
                                                                  x_pos=x_pos,
                                                                  y_pos=y_pos,
                                                                  moved_polarizer='second')
    background_ref_Y_position = 2000
    fig_cross_section_plot, ax_cross_section_plot = pixel_plotter_NEC.plot_x_cross_section_um(map_array=adjusted_map_array,
                                                                                              x_pos=x_pos,
                                                                                              label='NEC Power Adjusted '
                                                                                                'Intensity',
                                                                                              background_ref_Y_position_um=background_ref_Y_position, )
    fig_cross_section_plot, ax_cross_section_plot = pixel_plotter_NEC.plot_x_cross_section_um(map_array=map_array_NEC,
                                                                                              x_pos=x_pos,
                                                                                              label='NEC Raw Intensity',
                                                                                              fig=fig_cross_section_plot,
                                                                                              ax=ax_cross_section_plot,
                                                                                              background_ref_Y_position_um=background_ref_Y_position)


    dir_path_HIKMICRO = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/HIKMICRO_camera/PowerSeries')
    pixel_plotter_HIKMICRO =PixelPlotter(maps_dir_path=dir_path_HIKMICRO,
                                         camera_name='HIKMICRO',
                                         known_angle=90,
                                         known_voltage_at_known_angle_in_V=0.604)
    HIKMICRO_map_array = pixel_plotter_HIKMICRO.get_single_map(map_filename='90 degrees Humidity 0,8.csv')
    fig_HIKMICRO_map, ax_HIKMICRO_map = pixel_plotter_HIKMICRO.plot_map(map_data=HIKMICRO_map_array)
    # fig_HIKMICRO_map.show()
    x_pos_HIKMICRO = 93
    y_pos_HIKMICRO = 157
    background_ref_Y_position_HIKMICRO = 1500
    x_axis_shift_in_um = +15

    fig_cross_section_plot, ax_cross_section_plot = pixel_plotter_HIKMICRO.plot_x_cross_section_um(map_array=HIKMICRO_map_array,
                                                                                                  x_pos=x_pos_HIKMICRO,
                                                                                                  label='HIKMICRO Raw Intensity',
                                                                                                  fig=fig_cross_section_plot,
                                                                                                  ax=ax_cross_section_plot,
                                                                                                  background_ref_Y_position_um=background_ref_Y_position_HIKMICRO,
                                                                                                   x_axis_shift_in_um=x_axis_shift_in_um)


    ax_cross_section_plot.set_xlim(-200, 200)
    fig_cross_section_plot.show()

    fig_HIKMICRO_pixel_power_dependence, ax_HIKMICRO_pixel_power_dependence = pixel_plotter_HIKMICRO.plot_intensity_vs_power(x_pos=x_pos_HIKMICRO,
                                                                                                                                  y_pos=y_pos_HIKMICRO,
                                                                                                                             moved_polarizer='second')
    fig_HIKMICRO_pixel_power_dependence.show()

