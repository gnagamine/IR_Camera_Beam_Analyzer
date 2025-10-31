import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import trim_mean


class AmplitudeSpectralDensityCalculator:
    def __init__(self,
                 frames_array,
                 frame_rate_Hz,
                 pixel_block_size: int = 10,
                 visualize_region: bool = False,  # <-- Added
                 pdf_path: str = "cropped_region_analysis.pdf",
                 shift_crop_region_x=0,
                 shift_crop_region_y=0):  # <-- Added

        self.frames_array = frames_array
        self.pixel_block_size = pixel_block_size
        self.frame_rate = frame_rate_Hz
        self.shift_crop_region_x = shift_crop_region_x
        self.shift_crop_region_y = shift_crop_region_y

        # Pass the visualization flag to the calculation method
        self.freqs, self.mean_asd = self.calculate_averaged_asd(
            visualize_region=visualize_region,
            pdf_path=pdf_path
        )

    # ------------------------------------------------------------------ #
    # NEW: Method to create and save maps of the cropped region
    # ------------------------------------------------------------------ #
    def _visualize_cropped_region(self,
                                  block_data,
                                  pdf_path="cropped_region_analysis.pdf"):
        """
        Calculates and saves maps of the cropped region's mean and std. dev.
        to a multi-page PDF.
        """
        print(f"Visualizing cropped region, saving maps to {pdf_path} ...")

        # Calculate statistics (maps)
        # Mean over time (axis=0)
        mean_map = np.nanmean(block_data,
                              axis=0)
        # Std Dev over time (axis=0)
        std_map = np.nanstd(block_data,
                            axis=0)

        try:
            # Use PdfPages to save multiple figures to one PDF
            with PdfPages(pdf_path) as pdf:
                # --- Figure 1: Mean Map (Fixed Pattern) ---
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                im1 = ax1.imshow(mean_map,
                                 cmap='viridis',
                                 interpolation='nearest')
                fig1.colorbar(im1,
                              ax=ax1,
                              label='Mean Signal (°C)')
                ax1.set_title(f'Mean Pixel Value (Time-Averaged)\nBlock Size: {self.pixel_block_size}x{self.pixel_block_size}')
                ax1.set_xlabel('Pixel X (relative)')
                ax1.set_ylabel('Pixel Y (relative)')

                pdf.savefig(fig1)  # Save the current figure to the PDF
                plt.close(fig1)  # Close the figure to free memory

                # --- Figure 2: Standard Deviation Map (Total Noise) ---
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                im2 = ax2.imshow(std_map,
                                 cmap='inferno',
                                 interpolation='nearest')
                fig2.colorbar(im2,
                              ax=ax2,
                              label='Std. Dev. (°C)')
                ax2.set_title(f'Pixel Standard Deviation (Total Noise)\nBlock Size: {self.pixel_block_size}x{self.pixel_block_size}')
                ax2.set_xlabel('Pixel X (relative)')
                ax2.set_ylabel('Pixel Y (relative)')

                pdf.savefig(fig2)  # Save the current figure to the PDF
                plt.close(fig2)  # Close the figure

            print(f"Successfully saved region maps to {pdf_path}")

        except Exception as e:
            print(f"Error saving visualization PDF: {e}")
            # Ensure any open plots are closed on error
            plt.close('all')

    def calculate_averaged_asd(self,
                                 visualize_region: bool = False,
                                 pdf_path: str = "cropped_region_analysis.pdf" ):
        """
        Calculates and plots the spatially averaged ASD from a central block of pixels.

        This method is highly efficient as it uses vectorized NumPy operations
        and the 'axis' parameter of scipy.signal.welch.

        Parameters
        ----------
        pixel_block_size : int
            The side length of a square block of pixels to average (e.g., 10 for a 10x10 block).

        Returns
        -------
        freqs : np.ndarray
            Array of frequencies (Hz).
        mean_asd : np.ndarray
            The spatially-averaged Amplitude Spectral Density (in Signal Units / sqrt(Hz)).
        """
        pixel_block_size = self.pixel_block_size
        # Step 1: Load the full data cube (T, H, W)
        data = self.frames_array
        n_frames = data.shape[0]

        # Step 2: Define the central block of pixels
        center_x =self.shift_crop_region_x+ (data.shape[2]) // 2
        center_y = self.shift_crop_region_y+ (data.shape[1]) // 2

        start_y = center_y - (pixel_block_size // 2)
        start_x = center_x - (pixel_block_size // 2)

        end_y = start_y + pixel_block_size
        end_x = start_x + pixel_block_size

        # Step 3: Slice the entire block at once -> (T, block_size, block_size)
        block_data = data[:, start_y:end_y, start_x:end_x]
        # ------------------------------------------------- #
        # Step 3b: Visualize the cropped region if requested
        # ------------------------------------------------- #
        if visualize_region:
            self._visualize_cropped_region(block_data,
                                           pdf_path)

        # Reshape to (T, n_pixels) where n_pixels = block_size * block_size
        # This gives us a 2D array where each column is a pixel's time series
        pixel_block_timeseries = block_data.reshape(n_frames,
                                                    -1)
        num_pixels = pixel_block_timeseries.shape[1]

        print(f"Calculating ASD for {num_pixels} pixels in a {pixel_block_size}x{pixel_block_size} block...")

        # Step 4: Pre-process the data
        # Remove the DC offset (mean) from *each pixel's* time series (each column)
        # We use nanmean to be robust to any NaN values
        block_no_dc = pixel_block_timeseries - np.nanmean(pixel_block_timeseries,
                                                          axis=0)

        # Step 5: Compute PSD for ALL pixels at once
        # 'nperseg' is the segment length. A power of 2 is good.
        # This balances frequency resolution vs. number of averages.
        n_per_seg = min(128,
                        n_frames)

        # By passing 'axis=0', welch calculates the PSD along the time axis
        # for *every single column* (pixel) in the block_no_dc array.
        freqs, psd_block = welch(
            block_no_dc,
            fs=self.frame_rate,
            nperseg=n_per_seg,
            scaling='density',
            axis=0  # <-- This is the key optimization
        )
        # psd_block now has shape (n_freqs, n_pixels)

        # Step 6: Average the PSDs
        # Average all the calculated PSDs together (across the pixel axis=1)
        # We use nanmean to safely ignore any "bad pixels" that were all NaNs
        # mean_psd = np.nanmean(psd_block,
        #                       axis=1)

        proportion_to_cut = 0.1

        # Apply trim_mean to each row (frequency bin)
        # Note: This is slower than a single vectorized call,
        # but is the most robust way to handle NaNs correctly.
        mean_psd = np.apply_along_axis(
            lambda row: trim_mean(row[np.isfinite(row)],
                                  proportion_to_cut),
            axis=1,
            arr=psd_block
        )

        # Step 7: Convert final average PSD to average ASD
        mean_asd = np.sqrt(mean_psd)

        return freqs, mean_asd
