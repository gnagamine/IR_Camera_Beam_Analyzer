import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PumpAndProbe_DataAnalysis import (FFTDataClass,
                                       PumpProbeDataClass)


class GlazPumpProbeData:
    def __init__(self,
                 file_path,
                 param_file_path=None):
        """Initialize with data file path and optional parameter file path."""
        self.file_path = file_path
        # Determine the data type from filename for plot labeling
        self.data_type = self._get_data_type()

        # Read the tab-delimited data file with no header
        self.data = pd.read_csv(file_path,
                                delimiter='\t',
                                header=None)

        # Derive parameter file path if not provided
        if param_file_path is None:
            base, fname = os.path.split(file_path)
            name, ext = os.path.splitext(fname)
            parts = name.split('_')
            # Derive dataset prefix (date_time) and file index
            if len(parts) >= 4:
                dataset_prefix = '_'.join(parts[:2])
                index = parts[-1]
                param_name = f"{dataset_prefix}_param_{index}"
            else:
                # fallback for unexpected filename
                param_name = name.replace(parts[-2],
                                          'param')
            param_file_path = os.path.join(base,
                                           f"{param_name}{ext}")

        # Read the parameter file (with header) and extract delays
        param_df = pd.read_csv(param_file_path,
                               delimiter='\t',
                               header=0)
        self.delays = param_df.iloc[:, 0].values
        # Filter out any rows where data contains NaN, keeping delays in sync
        mask = ~np.isnan(self.data.values).any(axis=1)
        self.data = self.data.loc[mask]
        self.delays = self.delays[mask]

        # Update delay parameters for compatibility
        self.delay_start = float(self.delays.min())
        self.delay_end = float(self.delays.max())
        if self.delays.size > 1:
            self.delay_step = float(np.mean(np.diff(self.delays)))
        else:
            self.delay_step = None

        # Validate that data rows match number of delays
        if self.data.shape[0] != self.delays.size:
            raise ValueError(f"Number of data rows {self.data.shape[0]} does not match number of delays "
                             f"{self.delays.size}")

        # Assign delays as the DataFrame index
        self.data.index = self.delays

    def _get_data_type(self):
        """Return a descriptive data type string based on the filename."""
        fname = os.path.basename(self.file_path)
        if 'wPump' in fname:
            return 'Pumped Spectrum'
        elif 'woPump' in fname:
            return 'Un-pumped Spectrum'
        elif 'SpectralDifference' in fname or 'param' in fname:
            return 'Differential Spectrum'
        else:
            return 'Spectrum'

    def plot_spectral_timetrace(self,
                                vmin_percentile=None,
                                vmax_percentile=None,
                                bool_plot_normalized_time_trace=False):
        """Create a 2D plot of spectral difference vs delay."""

        if bool_plot_normalized_time_trace:
            data = self.normalize_spectra()
        else:
            data = self.data

        # Determine color scale limits based on normalized or raw data
        if bool_plot_normalized_time_trace and vmin_percentile is not None and vmax_percentile is not None:
            # For normalized data, interpret percentiles as fractional cutoffs
            vmin = vmin_percentile / 100.0
            vmax = vmax_percentile / 100.0
        elif vmin_percentile is not None and vmax_percentile is not None:
            # For raw data, use distribution percentiles
            vmin = np.percentile(data.values,
                                 vmin_percentile)
            vmax = np.percentile(data.values,
                                 vmax_percentile)
        else:
            # Default full data range
            vmin = np.min(data.values)
            vmax = np.max(data.values)

        # Build figure and axes, return them for further customization
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data.values,
                       aspect='auto',
                       origin='lower',
                       extent=[0, data.shape[1], self.delay_start, self.delay_end],
                       cmap='viridis',
                       vmin=vmin,
                       vmax=vmax)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Delay (ps)')
        fig.colorbar(im,
                     ax=ax,
                     label='counts')
        ax.set_title(self.data_type)
        fig.savefig(self.data_type + ' Time Trace.png')
        return fig, ax

    def plot_cross_section(self,
                           pixel):
        """Plot the spectral difference vs delay for a specific pixel.

        Args:
            pixel (int): The pixel index (zero-based) to plot the cross-section for.
        """
        if not isinstance(pixel,
                          int):
            raise TypeError("Pixel must be an integer")
        if pixel < 0 or pixel >= self.data.shape[1]:
            raise ValueError(f"Pixel {pixel} is out of range. Must be between 0 and {self.data.shape[1] - 1}")

        # Extract the data for the specified pixel
        single_pixel_timetrace = self.data.iloc[:, pixel]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.delays,
                 single_pixel_timetrace)
        plt.xlabel('Delay (ps)')
        plt.ylabel(f'Spectral Difference at Pixel {pixel}')
        plt.title(f'Cross-Section at Pixel {pixel}')
        plt.grid(True)
        plt.savefig(f'cross_section_pixel_{pixel}.png')
        plt.show()

    def plot_fft(self,
                 pixel,
                 window_start=None,
                 window_end=None):
        """Calculate and plot the FFT of the time trace for a specific pixel within a selected time window.

        Args:
            pixel (int): The pixel index (zero-based) to perform FFT on.
            window_start (float, optional): The start of the delay window (ps). Defaults to delay_start.
            window_end (float, optional): The end of the delay window (ps). Defaults to delay_end.
        """
        if not isinstance(pixel,
                          int):
            raise TypeError("Pixel must be an integer")
        if pixel < 0 or pixel >= self.data.shape[1]:
            raise ValueError(f"Pixel {pixel} is out of range. Must be between 0 and {self.data.shape[1] - 1}")

        # Set default window to full range if not specified
        window_start = self.delay_start if window_start is None else window_start
        window_end = self.delay_end if window_end is None else window_end

        # Validate window range
        if window_start < self.delay_start or window_end > self.delay_end or window_start >= window_end:
            raise ValueError(f"Invalid window range: {window_start} to {window_end} ps. Must be within {self.delay_start} to {self.delay_end} ps and start < end.")

        # Extract the data for the specified pixel and window
        mask = (self.delays >= window_start) & (self.delays <= window_end)
        delays_window = self.delays[mask]
        cross_section_window = self.data.iloc[mask, pixel]

        # Calculate FFT
        N = len(delays_window)
        T = self.delay_step  # Time step in ps
        fft_values = np.fft.fft(cross_section_window)
        freq = np.fft.fftfreq(N,
                              T)

        # Only take positive frequencies
        positive_mask = freq > 0
        freq = freq[positive_mask]
        fft_magnitude = np.abs(fft_values)[positive_mask]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(freq,
                 fft_magnitude)
        plt.xlim(0,
                 11)
        plt.xlabel('Frequency THz')
        plt.ylabel('FFT Magnitude')
        plt.title(f'FFT of Cross-Section at Pixel {pixel} (Window: {window_start} to {window_end} ps)')
        plt.grid(True)
        plt.savefig(f'fft_pixel_{pixel}_window_{window_start}_{window_end}.png')
        plt.show()

    def plot_cross_section_at_delay(self,
                                    delay,
                                    fig=None,
                                    ax=None):
        """Plot the spectral difference vs pixel for the closest available delay.

        Args:
            delay (float): The target delay time (ps). The closest available delay will be used.
        """

        # Create new figure/axes only if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Find the index of the closest delay
        idx = np.argmin(np.abs(self.delays - delay))
        selected_delay = self.delays[idx]
        print(f"Using closest delay: {selected_delay} ps")

        # Extract the spectrum at that delay
        spectrum = self.data.iloc[idx, :]

        # Plot on the provided or new axes
        ax.plot(range(self.data.shape[1]),
                spectrum)
        ax.set_xlabel('Pixel Index')
        ax.set_ylabel('Spectral Difference')
        ax.set_title(f'Cross-Section at Delay {selected_delay} ps')
        ax.grid(True)
        # Save the plot
        filename = f'cross_section_delay_{selected_delay:.2f}.png'
        fig.savefig(filename)
        print(f"Plot saved as {filename}")
        fig.show()

        return fig, ax

    def normalize_spectra(self):
        """Compute normalized_data where each spectrum (row) is scaled by its maximum amplitude."""
        # Determine the maximum value for each delay (row)
        row_max = self.data.max(axis=1)
        # Divide each row by its maximum, aligning on the index
        self.normalized_data = self.data.div(row_max,
                                             axis=0)
        return self.normalized_data


if __name__ == "__main__":
    # Usage
    file_path = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Phononic Cross Phase '
                 'Modulation/20250416/20250416_150138_SpectralDifference_0.txt')
    data = GlazPumpProbeData(file_path)
    data.plot_spectral_timetrace()
    data.plot_cross_section(600)
    data.plot_fft(600,
                  301.5,
                  307)

    data.plot_cross_section_at_delay(301.9)
