import numpy as np
import matplotlib.pyplot as plt


class BeamCharacteristicsExtractor:
    """
    A class to analyze a 2D laser beam intensity profile, extract FWHM in x and y,
    and visualize the results.
    """

    def __init__(self,
                 map_array: np.ndarray,
                 camera_name: str = None,
                 label: str = None,
                 Y_ref_position_for_background_subtraction: int = None
                 ):
        """
        Initializes the FWHMExtractor with a 2D beam profile.

        Args:
            map_array (np.ndarray): 2D NumPy array representing the beam intensity.
                                          Each value corresponds to light intensity at a pixel.

        Raises:
            ValueError: If the input is not a valid 2D NumPy array or is empty.
        """
        if camera_name is None:
            raise ValueError("Camera name must be provided.")
        if not isinstance(map_array,
                          np.ndarray) or map_array.ndim != 2:
            raise ValueError("Input beam_profile_2d must be a 2D NumPy array.")
        if map_array.size == 0:
            raise ValueError("Input beam_profile_2d cannot be empty.")

        self.Y_ref_position_for_background_subtraction = Y_ref_position_for_background_subtraction
        self.map_array = map_array
        if self.Y_ref_position_for_background_subtraction is not None:
            background = np.average(self.map_array[:, 0:self.Y_ref_position_for_background_subtraction])
            self.map_array = self.map_array - background

        self.max_row, self.max_col = self._find_max_intensity_coords()
        self.max_intensity_value = self.map_array[self.max_row, self.max_col]
        self.camera_name = camera_name
        self.label = label

        self.profile_x, self.profile_y = self._get_profile_cuts()

        self.fwhm_x, self.fwhm_x_left_coord, self.fwhm_x_right_coord = self._calculate_fwhm_for_profile(self.profile_x)
        self.fwhm_y, self.fwhm_y_left_coord, self.fwhm_y_right_coord = self._calculate_fwhm_for_profile(self.profile_y)

        self.total_intensity_in_array = np.sum(self.map_array)

    def _find_max_intensity_coords(self) -> tuple[int, int]:
        """
        Finds the coordinates (row, column) of the maximum intensity in the 2D beam profile.
        Internal helper method.

        Returns:
            tuple[int, int]: (row, col) of the maximum intensity.
        """
        max_row, max_col = np.unravel_index(np.argmax(self.map_array),
                                            self.map_array.shape)
        return int(max_row), int(max_col)

    def _get_profile_cuts(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts 1D profiles (cuts) along x and y directions passing through the maximum intensity point.
        Internal helper method.

        Returns:
            tuple[np.ndarray, np.ndarray]: (profile_x, profile_y)
                                           profile_x is the horizontal cut along self.max_row.
                                           profile_y is the vertical cut along self.max_col.
        """
        profile_x = self.map_array[self.max_row, :]
        profile_y = self.map_array[:, self.max_col]
        return profile_x, profile_y

    def _calculate_fwhm_for_profile(self,
                                    profile: np.ndarray) -> tuple[float, float | None, float | None]:
        """
        Calculates the Full Width at Half Maximum (FWHM) of a 1D profile numerically.
        Also returns the coordinates of the points where intensity is half-maximum.
        Internal helper method.

        Args:
            profile (np.ndarray): 1D array representing the intensity profile.

        Returns:
            tuple[float, float | None, float | None]:
                - FWHM value in pixels.
                - Left coordinate of FWHM.
                - Right coordinate of FWHM.
                Returns (0.0, None, None) if FWHM cannot be determined.
        """
        profile = np.asarray(profile,
                             dtype=float)
        if len(profile) < 2:
            return 0.0, None, None

        max_val = np.max(profile)
        if max_val == 0:
            return 0.0, None, None

        peak_idx = np.argmax(profile)
        half_max = max_val / 2.0

        left_cross, right_cross = None, None

        # Find Left Crossing
        if profile[0] >= half_max:
            left_cross = 0.0
        else:
            for i in range(peak_idx,
                           0,
                           -1):
                p_prev, p_curr = profile[i - 1], profile[i]
                if p_prev <= half_max < p_curr:
                    left_cross = (i - 1) + (half_max - p_prev) / (p_curr - p_prev) if p_curr != p_prev else float(i - 1)
                    break

        # Find Right Crossing
        if profile[-1] >= half_max:
            right_cross = float(len(profile) - 1)
        else:
            for i in range(peak_idx,
                           len(profile) - 1):
                p_curr, p_next = profile[i], profile[i + 1]
                if p_curr > half_max >= p_next:
                    right_cross = i + (p_curr - half_max) / (p_curr - p_next) if p_curr != p_next else float(i + 1)
                    break

        if left_cross is not None and right_cross is not None and right_cross >= left_cross:
            return right_cross - left_cross, left_cross, right_cross

        return 0.0, None, None

    def get_fwhm_x_data(self) -> dict:
        """
        Returns the FWHM data for the X profile.

        Returns:
            dict: {'fwhm': float, 'left_coord': float | None, 'right_coord': float | None}
        """
        return {
            'fwhm': self.fwhm_x,
            'left_coord': self.fwhm_x_left_coord,
            'right_coord': self.fwhm_x_right_coord
        }

    def get_fwhm_y_data(self) -> dict:
        """
        Returns the FWHM data for the Y profile.

        Returns:
            dict: {'fwhm': float, 'left_coord': float | None, 'right_coord': float | None}
        """
        return {
            'fwhm': self.fwhm_y,
            'left_coord': self.fwhm_y_left_coord,
            'right_coord': self.fwhm_y_right_coord
        }

    def get_max_intensity_details(self) -> dict:
        """
        Returns the coordinates and value of the maximum intensity.

        Returns:
            dict: {'coords': tuple[int, int], 'value': float}
        """
        return {
            'coords': (self.max_row, self.max_col),
            'value': self.max_intensity_value
        }

    def plot_analysis(self,
                      show_plot: bool = True,
                      bool_savel_plots: bool = False,):
        """
        Plots the 2D beam profile map, and the 1D X and Y profiles with FWHM indications.

        Args:
            show_plot (bool): If True, displays the plot. Otherwise, creates the plot
                              but does not call plt.show(). Useful for further modifications
                              or saving the figure externally.
        Returns:
            matplotlib.figure.Figure: The figure object containing the plots.
        """
        num_plots = 3
        fig_width = 18

        fig, axs = plt.subplots(1,
                                num_plots,
                                figsize=(fig_width, 5))
        plot_idx = 0

        # Plot 2D Beam Profile
        im = axs[plot_idx].imshow(self.map_array,
                                  aspect='auto',
                                  cmap='viridis',
                                  origin='upper')
        if self.label is not None:
            axs[plot_idx].set_title(self.label + '\n total intensity = ' + str(np.sum(self.map_array)))
        else:
            axs[plot_idx].set_title('2D Beam Profile\n total intensity = ' + str(np.sum(self.map_array)))
        axs[plot_idx].set_xlabel('Pixel Column')
        axs[plot_idx].set_ylabel('Pixel Row')
        axs[plot_idx].axhline(self.max_row,
                              color='r',
                              linestyle='--',
                              alpha=0.7,
                              label=f'Y-cut (row {self.max_row})')
        axs[plot_idx].axvline(self.max_col,
                              color='lime',
                              linestyle='--',
                              alpha=0.7,
                              label=f'X-cut (col {self.max_col})')
        axs[plot_idx].plot(self.max_col,
                           self.max_row,
                           'wo',
                           markersize=6,
                           markeredgecolor='k',
                           label='Max Intensity')
        axs[plot_idx].legend(fontsize='small')
        fig.colorbar(im,
                     ax=axs[plot_idx],
                     orientation='vertical',
                     fraction=0.046,
                     pad=0.04,
                     label='Intensity')
        plot_idx += 1

        # Plot X profile
        axs[plot_idx].plot(self.profile_x,
                           label=f'X-profile (row {self.max_row})',
                           color='lime')
        axs[plot_idx].set_title(f'Beam Profile along X\nFWHM: {self.fwhm_x:.2f} pixels')
        axs[plot_idx].set_xlabel('Pixel Column')
        axs[plot_idx].set_ylabel('Intensity')
        if self.fwhm_x_left_coord is not None and self.fwhm_x_right_coord is not None:
            half_max_val_x = np.max(self.profile_x) / 2.0
            axs[plot_idx].plot([self.fwhm_x_left_coord, self.fwhm_x_right_coord],
                               [half_max_val_x, half_max_val_x],
                               'r--',
                               label='FWHM Level')
            axs[plot_idx].axvline(self.fwhm_x_left_coord,
                                  color='r',
                                  linestyle=':',
                                  alpha=0.7)
            axs[plot_idx].axvline(self.fwhm_x_right_coord,
                                  color='r',
                                  linestyle=':',
                                  alpha=0.7)
            axs[plot_idx].text(self.fwhm_x_left_coord,
                               half_max_val_x,
                               f' {self.fwhm_x_left_coord:.1f}',
                               color='r',
                               ha='right',
                               va='bottom')
            axs[plot_idx].text(self.fwhm_x_right_coord,
                               half_max_val_x,
                               f' {self.fwhm_x_right_coord:.1f}',
                               color='r',
                               ha='left',
                               va='bottom')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True,
                           linestyle=':',
                           alpha=0.6)
        plot_idx += 1

        # Plot Y profile
        axs[plot_idx].plot(self.profile_y,
                           label=f'Y-profile (col {self.max_col})',
                           color='red')
        axs[plot_idx].set_title(f'Beam Profile along Y\nFWHM: {self.fwhm_y:.2f} pixels')
        axs[plot_idx].set_xlabel('Pixel Row')
        axs[plot_idx].set_ylabel('Intensity')
        if self.fwhm_y_left_coord is not None and self.fwhm_y_right_coord is not None:
            half_max_val_y = np.max(self.profile_y) / 2.0
            axs[plot_idx].plot([self.fwhm_y_left_coord, self.fwhm_y_right_coord],
                               [half_max_val_y, half_max_val_y],
                               'lime',
                               linestyle='--',
                               label='FWHM Level')
            axs[plot_idx].axvline(self.fwhm_y_left_coord,
                                  color='lime',
                                  linestyle=':',
                                  alpha=0.7)
            axs[plot_idx].axvline(self.fwhm_y_right_coord,
                                  color='lime',
                                  linestyle=':',
                                  alpha=0.7)
            axs[plot_idx].text(self.fwhm_y_left_coord,
                               half_max_val_y,
                               f' {self.fwhm_y_left_coord:.1f}',
                               color='lime',
                               ha='right',
                               va='bottom')
            axs[plot_idx].text(self.fwhm_y_right_coord,
                               half_max_val_y,
                               f' {self.fwhm_y_right_coord:.1f}',
                               color='lime',
                               ha='left',
                               va='bottom')
        axs[plot_idx].legend()
        axs[plot_idx].grid(True,
                           linestyle=':',
                           alpha=0.6)

        plt.tight_layout()
        if show_plot:
            plt.show()
        if bool_savel_plots:
            if self.label is None:
                plt.savefig(f'{self.camera_name}_BeamAnalysis.pdf')
            else:
                plt.savefig(f'{self.camera_name}_{self.label}_BeamAnalysis.pdf')
        return fig, axs

    def get_all_results(self) -> dict:
        """
        Returns a dictionary containing all major analysis results.

        Returns:
            dict: A dictionary with keys:
                  'max_intensity_coords': (row, col),
                  'max_intensity_value': float,
                  'fwhm_x': float,
                  'fwhm_x_coords': (left, right) or (None, None),
                  'fwhm_y': float,
                  'fwhm_y_coords': (left, right) or (None, None),
                  'profile_x': np.ndarray,
                  'profile_y': np.ndarray
        """
        return {
            'max_intensity_coords': (self.max_row, self.max_col),
            'max_intensity_value': self.max_intensity_value,
            'fwhm_x': self.fwhm_x,
            'fwhm_x_coords': (self.fwhm_x_left_coord, self.fwhm_x_right_coord),
            'fwhm_y': self.fwhm_y,
            'fwhm_y_coords': (self.fwhm_y_left_coord, self.fwhm_y_right_coord),
            'profile_x': self.profile_x,
            'profile_y': self.profile_y
        }


# --- Example Usage ---
if __name__ == "__main__":
    # Create a sample 2D Gaussian beam profile
    array_size = 200
    grid_range = 10.0
    x_coords = np.linspace(-grid_range / 2,
                           grid_range / 2,
                           array_size)
    y_coords = np.linspace(-grid_range / 2,
                           grid_range / 2,
                           array_size)
    X, Y = np.meshgrid(x_coords,
                       y_coords)

    amplitude = 200.0
    x0, y0 = 0.5, -0.2
    sigma_x_units, sigma_y_units = 1.0, 1.5

    gaussian_beam = amplitude * np.exp(
        -(((X - x0) ** 2 / (2 * sigma_x_units ** 2)) + ((Y - y0) ** 2 / (2 * sigma_y_units ** 2)))
    )

    noise_level = 10.0
    noise = np.random.normal(0,
                             noise_level,
                             gaussian_beam.shape)
    noisy_beam = gaussian_beam + noise
    noisy_beam = np.clip(noisy_beam,
                         0,
                         None)

    pixels_per_unit = array_size / grid_range
    sigma_x_pix = sigma_x_units * pixels_per_unit
    sigma_y_pix = sigma_y_units * pixels_per_unit
    theoretical_fwhm_x = 2.35482 * sigma_x_pix
    theoretical_fwhm_y = 2.35482 * sigma_y_pix

    print(f"--- Theoretical Values (for Gaussian part) ---")
    print(f"Sigma X: {sigma_x_units:.2f} units ({sigma_x_pix:.2f} pixels)")
    print(f"Sigma Y: {sigma_y_units:.2f} units ({sigma_y_pix:.2f} pixels)")
    print(f"Theoretical FWHM_x: {theoretical_fwhm_x:.2f} pixels")
    print(f"Theoretical FWHM_y: {theoretical_fwhm_y:.2f} pixels")
    print("-" * 40)

    try:
        print("\n--- Analyzing Noisy Gaussian Beam ---")
        analyzer_noisy = BeamCharacteristicsExtractor(noisy_beam)

        max_details = analyzer_noisy.get_max_intensity_details()
        print(f"Max Intensity: Value={max_details['value']:.2f} at Coords={max_details['coords']}")

        fwhm_x_data = analyzer_noisy.get_fwhm_x_data()
        print(f"FWHM X: {fwhm_x_data['fwhm']:.2f} pixels (Coords: [{fwhm_x_data['left_coord']:.2f if fwhm_x_data['left_coord'] is not None else 'N/A'}, {fwhm_x_data['right_coord']:.2f if fwhm_x_data['right_coord'] is not None else 'N/A'}])")

        fwhm_y_data = analyzer_noisy.get_fwhm_y_data()
        print(f"FWHM Y: {fwhm_y_data['fwhm']:.2f} pixels (Coords: [{fwhm_y_data['left_coord']:.2f if fwhm_y_data['left_coord'] is not None else 'N/A'}, {fwhm_y_data['right_coord']:.2f if fwhm_y_data['right_coord'] is not None else 'N/A'}])")

        analyzer_noisy.plot_analysis()

        all_results_noisy = analyzer_noisy.get_all_results()
        # print("\nAll results dictionary keys:", all_results_noisy.keys())

        print("\n--- Analyzing Perfect Gaussian Beam ---")
        perfect_beam = amplitude * np.exp(
            -(((X - 0) ** 2 / (2 * sigma_x_units ** 2)) + ((Y - 0) ** 2 / (2 * sigma_y_units ** 2)))
        )
        analyzer_perfect = BeamCharacteristicsExtractor(perfect_beam)
        analyzer_perfect.plot_analysis()
        print(f"Perfect Beam FWHM X: {analyzer_perfect.get_fwhm_x_data()['fwhm']:.2f} pixels")
        print(f"Perfect Beam FWHM Y: {analyzer_perfect.get_fwhm_y_data()['fwhm']:.2f} pixels")

        print("\n--- Analyzing Flat Beam ---")
        flat_beam = np.ones((50, 50)) * 10
        analyzer_flat = BeamCharacteristicsExtractor(flat_beam)
        analyzer_flat.plot_analysis()  # Should show FWHM as width of beam or handle gracefully
        print(f"Flat Beam FWHM X: {analyzer_flat.get_fwhm_x_data()['fwhm']:.2f} pixels")
        print(f"Flat Beam FWHM Y: {analyzer_flat.get_fwhm_y_data()['fwhm']:.2f} pixels")

        print("\n--- Analyzing Narrow Beam (should be ~1 pixel FWHM if peak is sharp enough) ---")
        narrow_beam_data = np.zeros((50, 50))
        narrow_beam_data[25, 24] = 50
        narrow_beam_data[25, 25] = 100  # Peak
        narrow_beam_data[25, 26] = 50
        analyzer_narrow = BeamCharacteristicsExtractor(narrow_beam_data)
        analyzer_narrow.plot_analysis()
        print(f"Narrow Beam FWHM X: {analyzer_narrow.get_fwhm_x_data()['fwhm']:.2f} pixels")
        print(f"Narrow Beam FWHM Y: {analyzer_narrow.get_fwhm_y_data()['fwhm']:.2f} pixels")  # Y profile will be mostly 0

    except ValueError as e:
        print(f"Error during analysis: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")