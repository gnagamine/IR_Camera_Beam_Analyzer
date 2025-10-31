import numpy as np

import matplotlib.pyplot as plt
from Code_utilities.IntensityMapIRCamera import IntensityMap
import os
from scipy.optimize import curve_fit

class BeamAnalysis:
    def __init__(self,
                 dir_path,
                 signal_filename,
                 background_filename=None,
                 camera_name = None,
                 crop_range_x_um = None,
                 crop_range_y_um = None,
                 crop_range_x_pixels = None,
                 crop_range_y_pixels = None,
                 width_in_pixels = None,
                 beam_center_x_in_pixels = None,
                 beam_center_y_in_pixels = None,):
        """
        Initialize with file paths for the signal and background CSV files.
        """
        if camera_name is None:
            raise ValueError("Camera name must be specified.")
        self.signal_filename = signal_filename
        signal_path =  os.path.join(dir_path, signal_filename)

        self.crop_range_x_um = crop_range_x_um
        self.crop_range_y_um = crop_range_y_um
        self.crop_range_x_pixels = crop_range_x_pixels
        self.crop_range_y_pixels = crop_range_y_pixels

        self.camera_name = camera_name


        self.dir_path = dir_path
        if camera_name == 'HIKMICRO':
            self.pixel_size_um = 12
            if background_filename is not None:
                background_path = os.path.join(dir_path,
                                               background_filename)
            if background_filename is None:
                background_filename = self.get_background_filename_from_signal_filename()
                background_path = os.path.join(dir_path,
                                               background_filename)
            self.background_filename = background_filename
            self.background = IntensityMap(background_path,
                                           camera_name=camera_name)
            self.raw_signal = IntensityMap(signal_path,
                                           camera_name = camera_name)

        if camera_name == 'NEC':
            self.pixel_size_um = 23.5
            self.background = None
            self.raw_signal = IntensityMap(signal_path,
                                           camera_name = camera_name)
        if camera_name == 'gentec':
            self.pixel_size_um = 5.5
            self.background = None
            self.raw_signal = IntensityMap(signal_path,
                                           camera_name = camera_name)
        self.load_data()
        self.processed_signal = self.subtract_background()
        self.map_array = self.processed_signal

    def get_background_filename_from_signal_filename(self):
        """
        Scan ``self.dir_path`` for a file that can serve as the background
        map for *self.signal_filename*.

        A file qualifies if **both** of the following are true (case‑insensitive):

        1. The filename contains the substring ``"background"``.
        2. The filename also contains the entire *signal* filename
           (``self.signal_filename``).

        Exactly one such file must be found; otherwise an error is raised.

        Returns
        -------
        str
            The background filename that matches the above criteria.

        Raises
        ------
        FileNotFoundError
            If no candidate file is found.
        ValueError
            If more than one candidate file is found.
        """
        # List all entries in the directory (case‑insensitive comparison)
        filenames = os.listdir(self.dir_path)
        signal_key = self.signal_filename.lower()
        if signal_key != '0 degrees.csv' and signal_key!='5 degrees.csv':
            candidates = [
                fname
                for fname in filenames
                if "background" in fname.lower() and signal_key in fname.lower()
            ]
        if signal_key == '0 degrees.csv':
            candidates = ['background 0 degrees.csv']
        if signal_key=='5 degrees.csv':
            candidates=['background 5 degrees.csv']

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No background file matching both 'background' and "
                f"'{self.signal_filename}' found in: {self.dir_path}"
            )

        if len(candidates) > 1:
            raise ValueError(
                "Multiple background files satisfy the criteria: "
                f"filename: {signal_key}"
                f"candidates: {candidates}.  Please specify the desired file explicitly."
            )

        return candidates[0]

    def load_data(self):
        """
        Load data for both the signal and background.
        """
        self.raw_signal.load_data()
        if self.crop_range_x_pixels is not None and self.crop_range_x_um is not None:
            raise ValueError("crop_range_x_pixels and crop_range_x_um cannot be used together. Please use crop_range_x_um and crop_range_y_um instead.")

        self.raw_signal.crop_data_um(x_range_um=self.crop_range_x_um,
                                     y_range_um=self.crop_range_y_um)
        self.raw_signal.crop_data_pixels(x_range_pixels=self.crop_range_x_pixels,
                                     y_range_pixels=self.crop_range_y_pixels)
        if self.background is not None:
            self.background.load_data()
            self.background.crop_data_um(x_range_um=self.crop_range_x_um,
                                     y_range_um=self.crop_range_y_um)
            self.background.crop_data_pixels(x_range_pixels=self.crop_range_x_pixels,
                                             y_range_pixels=self.crop_range_y_pixels)
    def subtract_background(self):
        """
        Subtract the background data from the signal data.
        """
        if self.raw_signal.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Ensure that the arrays are of the same shape
        if self.background is not None:
            if self.raw_signal.data.shape != self.background.data.shape:
                signal = self.raw_signal.data
                background = self.background.data
                print(f"Signal shape: {signal.shape}")
                print(f"Background shape: {background.shape}")
                raise ValueError("Signal and background data must have the same shape.")
        if self.background is not None :
            self.processed_signal = self.raw_signal.data - self.background.data
        if self.background is None: self.processed_signal = self.raw_signal.data

        if self.background is None and self.camera_name == 'HIKMICRO':
            raise ValueError("Background data not provided by HIKMICRO. Either comment this error out or provide background data.")

        return self.processed_signal


    def plot_maps(self):
        """
        Plot the signal, background, and background-subtracted maps side by side.
        """
        if self.raw_signal.data is None or self.background.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Calculate the subtracted map if not already done
        if self.processed_signal is None:
            self.subtract_background()

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1,
                                3,
                                figsize=(15, 5))

        # Plot signal
        self.raw_signal.plot(ax=axs[0],
                             title="Signal")

        # Plot background
        self.background.plot(ax=axs[1],
                             title="Background")

        # Plot signal minus background
        im = axs[2].imshow(self.processed_signal,
                           cmap='viridis')
        axs[2].set_title("Signal - Background")
        plt.colorbar(im,
                     ax=axs[2])

        plt.tight_layout()
        plt.show()

    def plot_maps_micrometer(self):
        """
        Plot the signal, background, and background-subtracted maps side by side with axes in micrometers.
        """
        if self.raw_signal.data is None or self.background.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Calculate the subtracted map if not already done
        if self.processed_signal is None:
            self.subtract_background()
        # Calculate the extent of the image in micrometers
        rows, cols = self.raw_signal.data.shape
        extent = [0, self.pixel_size_um * cols, 0, self.pixel_size_um * rows]

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1,
                                3,
                                figsize=(15, 5))

        # Plot signal
        im0 = axs[0].imshow(self.raw_signal.data,
                            cmap='viridis',
                            extent=extent)
        axs[0].set_title("Signal")
        axs[0].set_xlabel("x (um)")
        axs[0].set_ylabel("y (um)")
        plt.colorbar(im0,
                     ax=axs[0])

        # Plot background
        im1 = axs[1].imshow(self.background.data,
                            cmap='viridis',
                            extent=extent)
        axs[1].set_title("Background")
        axs[1].set_xlabel("x (um)")
        axs[1].set_ylabel("y (um)")
        plt.colorbar(im1,
                     ax=axs[1])

        # Plot signal minus background
        im2 = axs[2].imshow(self.processed_signal,
                            cmap='viridis',
                            extent=extent)
        axs[2].set_title("Signal - Background")
        axs[2].set_xlabel("x (um)")
        axs[2].set_ylabel("y (um)")
        plt.colorbar(im2,
                     ax=axs[2])

        plt.tight_layout()
        plt.show()


    def plot_maps_micrometer(self):
        """
        Plot the signal, background, and background-subtracted maps in three separate figures with axes in micrometers.
        The colorbar for the difference plot is labeled as "temperature difference in celcius".
        """
        if self.raw_signal.data is None or self.background.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Calculate the subtracted map if not already done
        if self.processed_signal is None:
            self.subtract_background()

        # Calculate the extent of the image in micrometers (data shape: [rows, cols])
        rows, cols = self.raw_signal.data.shape
        extent = [0, self.pixel_size_um * cols, 0, self.pixel_size_um * rows]

        # Figure for Signal
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        im0 = ax1.imshow(self.raw_signal.data, cmap='viridis', extent=extent)
        ax1.set_title("Signal")
        ax1.set_xlabel("x (um)")
        ax1.set_ylabel("y (um)")
        plt.colorbar(im0, ax=ax1)

        # Figure for Background
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        im1 = ax2.imshow(self.background.data, cmap='viridis', extent=extent)
        ax2.set_title("Background")
        ax2.set_xlabel("x (um)")
        ax2.set_ylabel("y (um)")
        plt.colorbar(im1, ax=ax2)

        # Figure for Signal - Background
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        im2 = ax3.imshow(self.processed_signal, cmap='viridis', extent=extent)
        ax3.set_title("Signal - Background")
        ax3.set_xlabel("x (um)")
        ax3.set_ylabel("y (um)")
        cbar = plt.colorbar(im2, ax=ax3)
        cbar.set_label("temperature difference in celcius")

        plt.show()

    def plot_map_in_pixels(self):
        """
        Plot the map array
        """

        # Calculate the extent of the image in micrometers (data shape: [rows, cols])
        rows, cols = self.map_array.shape
        extent = [0, cols, 0,  rows]

        # Figure for Signal
        fig, ax = plt.subplots()
        im0 = ax.imshow(self.map_array, cmap='viridis', extent=extent)
        ax.set_title("Map")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        plt.colorbar(im0, ax=ax)

        return fig, ax

    def plot_background_in_pixels(self):
        """
        Plot the map array
        """

        # Calculate the extent of the image in micrometers (data shape: [rows, cols])
        rows, cols = self.background.shape
        extent = [0, cols, 0,  rows]

        # Figure for Signal
        fig, ax = plt.subplots()
        im0 = ax.imshow(self.background.data, cmap='viridis', extent=extent)
        ax.set_title("Map")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        plt.colorbar(im0, ax=ax)

        return fig, ax


    def twoD_Gaussian(self,
                      coords,
                      amplitude,
                      xo,
                      yo,
                      sigma_x,
                      sigma_y,
                      offset):

        x, y = coords
        g = offset + amplitude * np.exp(-(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2)))
        return g.ravel()

    def generate_initial_guess(self,
                               data,
                               rows,
                               cols):
        """
        Generate a robust data‑driven initial guess for the 2‑D Gaussian fit.

        Parameters
        ----------
        data : np.ndarray
            2‑D array containing the background‑subtracted beam image.
        rows, cols : int
            Dimensions of ``data`` (passed in to avoid re‑computing them).

        Returns
        -------
        tuple
            (amplitude, xo, yo, sigma_x, sigma_y, offset) suitable for
            :pyfunc:`scipy.optimize.curve_fit`.
        """

        # ----- Estimate background (offset) and amplitude ------------------
        offset = np.percentile(data, 1)          # ignore brightest pixels
        amplitude = data.max() - offset
        amplitude = max(amplitude, 1e-6)          # guard against zero peak


        # ----- Peak position for the beam centre (xo, yo) ------------------
        # Use the pixel with maximum intensity as the initial centre.
        row_peak, col_peak = np.unravel_index(np.argmax(data), data.shape)
        xo = float(col_peak)   # x‑coordinate corresponds to column index
        yo = float(row_peak)   # y‑coordinate corresponds to row index

        # Pre-compute meshgrid for later sigma estimation
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y, indexing="xy")

        total_intensity = data.sum() or 1.0  # prevent divide‑by‑zero

        # ----- Second moments for initial sigmas --------------------------
        sigma_x = np.sqrt(((X - xo) ** 2 * data).sum() / total_intensity)/1.5
        sigma_y = np.sqrt(((Y - yo) ** 2 * data).sum() / total_intensity)/1.5

        # Fallbacks if the image is too flat or noisy
        if not np.isfinite(sigma_x) or sigma_x == 0:
            sigma_x = cols / 9
        if not np.isfinite(sigma_y) or sigma_y == 0:
            sigma_y = rows / 9

        # ----- Clamp values to stay inside image bounds and pos‑offset -----
        xo = min(max(xo, 0), cols - 1)
        yo = min(max(yo, 0), rows - 1)
        # Offset must not violate the lower bound we will set (≥ ‑inf is OK,
        # but here we clamp to at least the 1 % percentile to avoid overshoot)
        if offset < -1e3:   # arbitrary large negative guard
            offset = -1e3
        return amplitude, xo, yo, sigma_x, sigma_y, offset

    def generate_upper_bound(self,
                              data,
                              rows,
                              cols):
        """
        Build a conservative upper‑bounds vector for
        (amplitude, xo, yo, sigma_x, sigma_y, offset).

        * amplitude capped at twice the full dynamic range
        * xo ≤ cols‑1, yo ≤ rows‑1
        * sigma_x, sigma_y ≤ max(rows, cols)  (i.e., image dimension)
        * offset allowed to float high (+∞)
        """
        dynamic_range   = data.max() - data.min()
        upper_amplitude = dynamic_range * 2 if np.isfinite(dynamic_range) else np.inf
        upper_xo        = cols - 1
        upper_yo        = rows - 1
        upper_sigma     = max(rows, cols)  # TODO The fittings still fails if the signal is too low
        upper_offset    = np.inf

        return [
            upper_amplitude,
            upper_xo,
            upper_yo,
            upper_sigma,
            upper_sigma,
            upper_offset,
        ]

    def generate_lower_bound(self,
                              data,
                              rows,
                              cols):
        """
        Build a conservative lower‑bounds vector for
        (amplitude, xo, yo, sigma_x, sigma_y, offset).

        * amplitude ≥ 0
        * xo, yo ≥ 0 (top‑left corner of the image)
        * sigma_x, sigma_y ≥ 1 pixel (avoid zero widths)
        * offset allowed to be arbitrarily negative
        """
        lower_amplitude = 0.0
        lower_xo        = 0.0
        lower_yo        = 0.0
        lower_sigma_x   = 1.0
        lower_sigma_y   = 1.0
        lower_offset    = -np.inf    # allow negative baseline

        return [
            lower_amplitude,
            lower_xo,
            lower_yo,
            lower_sigma_x,
            lower_sigma_y,
            lower_offset,
        ]

    #(data.max(), cols / 2, rows / 2, cols / 4, rows / 4, np.abs(np.average(data)))
    def fit_gaussian(self,
                     bool_save_plots=False):
        """
        Fit a 2D Gaussian to the beam data (signal minus background) and extract the beam widths in x and y.
        The beam widths are returned in micrometers by multiplying the fitted sigma values (in pixels)
        by the pixel size.
        """
        if self.processed_signal is None:
            self.subtract_background()
        data = self.processed_signal
        rows, cols = data.shape
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        # Initial guess: amplitude, xo, yo, sigma_x, sigma_y, offset
        initial_guess = self.generate_initial_guess(data,
                                                    rows,
                                                    cols)

        # Allow negative baseline (offset) while keeping physical constraints
        lower_bounds = self.generate_lower_bound(data, rows, cols)
        upper_bounds = self.generate_upper_bound(data, rows, cols)

        # lower_bounds = [0, 0, 0, 0, 0, -np.inf]
        # upper_bounds = [np.inf, cols - 1, rows - 1, np.inf, np.inf, np.inf]

        popt, pcov = curve_fit(self.twoD_Gaussian,
                               (X, Y),
                               data.ravel(),
                               p0=initial_guess,
                               bounds=(lower_bounds, upper_bounds))

        amplitude, xo, yo, sigma_x, sigma_y, offset = popt
        beam_width_x_um = sigma_x * self.pixel_size_um
        beam_width_y_um = sigma_y * self.pixel_size_um

        # Convert 1‑e² Gaussian sigma to FWHM (Full Width at Half Maximum)
        fwhm_factor = 2 * np.sqrt(2 * np.log(2))  # ≈ 2.3548
        fwhm_x_um = beam_width_x_um * fwhm_factor
        fwhm_y_um = beam_width_y_um * fwhm_factor
        beam_width_average = (beam_width_x_um+beam_width_y_um)/2

        fitting_coefficients = {
            "amplitude": amplitude,
            "xo": xo,
            "yo": yo,
            "sigma_x": beam_width_x_um,
            "sigma_y": beam_width_y_um,
            "fwhm_x": fwhm_x_um,
            "fwhm_y": fwhm_y_um,
            "offset": offset,
            'beam_width_average': beam_width_average
        }
        fig, ax = None, None
        if bool_save_plots:
            fig, ax = self.plot_fit_comparison()
        return popt, fitting_coefficients, fig, ax

    def plot_fit_comparison(self):
        """
        Plot a comparison between the actual beam data (signal minus background) and the 2D Gaussian fit model.
        The fitted model is generated using the parameters from fit_gaussian().
        """
        # Ensure that the subtracted data is available
        if self.processed_signal is None:
            self.subtract_background()

        # Get the fit parameters; this will print beam widths as well
        popt, _, _, _ = self.fit_gaussian()
        amplitude, xo, yo, sigma_x, sigma_y, offset = popt

        # Prepare grid for evaluation
        data = self.processed_signal
        rows, cols = data.shape
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x,
                           y)

        # Evaluate the 2D Gaussian using the fitted parameters
        fitted_map = offset + amplitude * np.exp(-(
                    ((X - xo) ** 2) / (2 * sigma_x ** 2) + ((Y - yo) ** 2) / (2 * sigma_y ** 2)))

        # Calculate the extent in micrometers
        extent = [0, self.pixel_size_um * cols, 0, self.pixel_size_um * rows]

        # Create a figure with two subplots: one for the actual data and one for the fitted model
        fig, axs = plt.subplots(1,
                                2,
                                figsize=(12, 5))

        # Plot the actual subtracted data
        im1 = axs[0].imshow(data,
                            cmap='viridis',
                            extent=extent)
        axs[0].set_title(self.signal_filename[:-4])
        axs[0].set_xlabel("x (um)")
        axs[0].set_ylabel("y (um)")
        plt.colorbar(im1,
                     ax=axs[0])

        # Plot the fitted Gaussian map
        im2 = axs[1].imshow(fitted_map,
                            cmap='viridis',
                            extent=extent)
        axs[1].set_title("2D Gaussian Fit")
        axs[1].set_xlabel("x (um)")
        axs[1].set_ylabel("y (um)")
        plt.colorbar(im2,
                     ax=axs[1])

        plt.tight_layout()
        fig.savefig(self.signal_filename[:-4] + "_fit_comparison.pdf")
        print('fit comparison saved in ' + os.getcwd())
        return fig, axs


if __name__ == "__main__":
    # Replace these file paths with your CSV file locations.
    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Strong THz Focusing/IR_Camera_Beam_Analysis'
    signal_filename = 'Image_THzBeam_after_1inch_parmirror.csv'
    background_filename = 'Image_background_after_1inch_parmirror.csv'

    analysis = BeamAnalysis(dir_path = dir_path,
                            signal_filename =signal_filename,
                            background_filename = background_filename)
    analysis.load_data()
    analysis.subtract_background()
    analysis.plot_maps_micrometer()
    popt, (beam_width_x_um, beam_width_y_um) = analysis.fit_gaussian()
    analysis.plot_fit_comparison()

    """Second parabola:"""
    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Strong THz Focusing/IR_Camera_Beam_Analysis'
    signal_filename = 'Image_THzBeam_after_2inch_parmirror.csv'
    background_filename = 'Image_background_after_1inch_parmirror.csv'

    analysis_2inch = BeamAnalysis(dir_path = dir_path,
                                  signal_filename =signal_filename,
                                  background_filename = background_filename)
    analysis_2inch.load_data()
    analysis_2inch.subtract_background()
    analysis_2inch.plot_maps_micrometer()
    popt, (beam_width_x_um, beam_width_y_um) = analysis_2inch.fit_gaussian()
    analysis_2inch.plot_fit_comparison()