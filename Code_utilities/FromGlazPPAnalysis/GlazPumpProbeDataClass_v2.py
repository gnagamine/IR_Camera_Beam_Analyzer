import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PumpAndProbe_DataAnalysis import (FFTDataClass,
                                       FFTPlotClass)
from Analyses.Plotting_tools.StandardFigureClass import StandardFigure


class GlazPumpProbeData_v2:
    def __init__(self,
                 file_path,
                 param_file_path=None,
                 bool_average=False):
        """Initialize with data file path and optional parameter file path."""
        self.file_path = file_path
        # Determine the data type from filename for plot labeling
        self.data_type = self._get_data_type()

        # Read the tab-delimited data file with no header
        if bool_average is False:
            self.pump_and_probe_spectrum_timetrace = pd.read_csv(file_path,
                                                                 delimiter='\t',
                                                                 header=None)
        else:
            self.pump_and_probe_spectrum_timetrace = self._get_averaged_timetrace()

        self.unpumped_timetrace_map = self._get_unpumped_data()

        # Derive parameter file path if not provided
        if param_file_path is None:
            param_file_path = self.get_param_filepath(file_path)

        # Save for later comment parsing
        self.param_file_path = param_file_path

        # Read the parameter file (with header) and extract delays
        if param_file_path.endswith('_averaged.txt'):
            param_file_path = param_file_path[:-len('_averaged.txt')] + '_0.txt'
        param_df = pd.read_csv(param_file_path,
                               delimiter='\t',
                               header=0)
        self.delays = param_df.iloc[:, 0].values
        self.filter_NaN_rows_from_data()
        self.comments = self.get_comments()

        # Update delay parameters for compatibility
        self.delay_start = float(self.delays.min())
        self.delay_end = float(self.delays.max())
        if self.delays.size > 1:
            self.delay_step = float(np.mean(np.diff(self.delays)))
        else:
            self.delay_step = None

        self.series_parameter = self.extract_series_parameter()

    @staticmethod
    def _is_float(token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False

    def _get_averaged_timetrace(self):
        # Check if the path is a valid directory
        path = os.path.dirname(self.file_path)
        keyword = os.path.basename(self.file_path)
        if not os.path.isdir(path):
            raise ValueError(f"'{path}' is not a valid directory")

        # Find all .txt files with the keyword in their name
        files = [f for f in os.listdir(path) if keyword in f and f.endswith('.txt')]
        if not files:
            raise ValueError(f"No .txt files found with keyword '{keyword}' in '{path}'")

        print(f"Found {len(files)} files with keyword '{keyword}'")

        # Detect delimiter from the first file
        first_file_path = os.path.join(path,
                                       files[0])
        with open(first_file_path,
                  'r') as f:
            first_line = f.readline()
            delimiter = '\t' if '\t' in first_line else ' '

        print(f"Detected delimiter: '{delimiter}'")

        # Read matrices from files
        matrices = []
        for file in files:
            file_path = os.path.join(path,
                                     file)
            try:
                matrix = np.loadtxt(file_path)
                # Ensure the matrix is 2D
                if matrix.ndim != 2:
                    raise ValueError(f"File '{file}' does not contain a 2D matrix")
                matrices.append(matrix)
            except Exception as e:
                raise ValueError(f"Error reading '{file}': {str(e)}")

        # Verify all matrices have the same shape
        shapes = [matrix.shape for matrix in matrices]
        if len(set(shapes)) > 1:
            raise ValueError("Matrices have different dimensions: " + ", ".join([str(s) for s in shapes]))

        print(f"All matrices have shape: {shapes[0]}")

        # Compute element-wise average
        stacked = np.stack(matrices,
                           axis=0)
        averaged = np.mean(stacked,
                           axis=0)

        # Generate output filename from the first file
        first_file = files[0]
        base_name = '_'.join(first_file.split('_')[:-1])
        output_file = os.path.join(path,
                                   base_name + '_averaged.txt')

        # Save the averaged matrix with the detected delimiter
        np.savetxt(output_file,
                   averaged,
                   fmt='%.6f',
                   delimiter=delimiter)

        print(f"Averaged matrix saved to '{output_file}'")
        averaged_timetrace_df = pd.read_csv(output_file,
                                            delimiter='\t',
                                            header=0)

        return averaged_timetrace_df

    def _get_unpumped_data(self):
        """
        Find the corresponding 'unpumped' file within the same directory.

        Criteria:
        - The first 15 characters of the filename are identical to the current file name.
        - The filename contains the substring 'woPump'.
        - The last 6 characters (e.g., '.csv' or similar) are identical.
        """

        parent_directory = os.path.dirname(self.file_path)
        current_filename = os.path.basename(self.file_path)
        all_filenames_in_directory = os.listdir(parent_directory)

        matching_files = []
        for candidate_filename in all_filenames_in_directory:
            # Ensure the filename is long enough to compare safely
            if len(candidate_filename) >= 21:  # 15 + 6 minimum
                same_prefix = candidate_filename[:15] == current_filename[:15]
                same_suffix = candidate_filename[-6:] == current_filename[-6:]
                contains_woPump = "woPump" in candidate_filename

                if same_prefix and same_suffix and contains_woPump:
                    matching_files.append(candidate_filename)

        if not matching_files:
            print(f"No matching 'unpumped' file found for {current_filename}")
            return None

        if len(matching_files) > 1:
            print(f"Multiple matching files found: {matching_files}")
            # You can decide what to do here — for example, pick the first
            return os.path.join(parent_directory,
                                matching_files[0])

        unpumped_filepath = os.path.join(parent_directory,
                                         matching_files[0])
        print(f"Found unpumped file: {unpumped_filepath}")
        unpumped_timetrace_map_df = pd.read_csv(unpumped_filepath,
                                                delimiter='\t',
                                                header=None)
        return unpumped_timetrace_map_df

    def get_comments(self):
        """
        Parse metadata comments from the parameter file header.
        Returns a dict where keys are the header fields (e.g. 'Scancount', 'Comments')
        and values are the corresponding text (multiline for 'Comments').
        """
        header_lines = []
        path = self.get_commets_filepath()
        with open(path,
                  "r") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue

                # stop only when the *whole line* looks numeric
                tokens = stripped.split()
                if all(self._is_float(t) for t in tokens):
                    break

                header_lines.append(stripped)
        text = "\n".join(header_lines)
        return text

    def extract_series_parameter(self,
                                 series_parameter_keyword="series keyword:"):  # Default to "series keyword:"
        """
        Extract the text following a specified keyword (e.g., 'series keyword:')
        from the 'Comments' metadata, regardless of its position in the line.

        Args:
            series_parameter_keyword (str, optional): The keyword to search for.
                Defaults to "series keyword:".

        Returns:
            str or None: The extracted string if the keyword is found, otherwise None.
        """
        if not series_parameter_keyword:  # Handle empty keyword if necessary
            return None

        keyword_to_find = series_parameter_keyword.lower()

        for line in self.comments.splitlines():
            line_lower = line.lower()
            if keyword_to_find in line_lower:
                # Find the starting position of the keyword in the original line (case-insensitive)
                start_index = line_lower.find(keyword_to_find)
                # Extract the substring after the keyword
                # Add len(keyword_to_find) to get the part *after* the keyword
                value = line[start_index + len(keyword_to_find):].strip()
                # If the keyword itself ends with a colon, an additional strip(':') might be useful
                # or ensure the keyword includes the colon if that's the delimiter.
                # For "series keyword:", the current strip() is likely sufficient.
                return value
        return None

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

    def filter_NaN_rows_from_data(self):
        mask = ~np.isnan(self.pump_and_probe_spectrum_timetrace.values).any(axis=1)
        self.pump_and_probe_spectrum_timetrace = self.pump_and_probe_spectrum_timetrace.loc[mask]
        self.delays = self.delays[mask]

    def get_param_filepath(self,
                           file_path):
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

        return param_file_path

    def get_commets_filepath(self):
        file_path = self.file_path
        base, fname = os.path.split(file_path)
        name, ext = os.path.splitext(fname)
        parts = name.split('_')
        # Derive dataset prefix (date_time) and file index
        if len(parts) >= 4:
            dataset_prefix = '_'.join(parts[:2])
            index = parts[-1]
            param_name = f"{dataset_prefix}_Param"
        else:
            # fallback for unexpected filename
            param_name = name.replace(parts[-2],
                                      'param')
        comments_file_path = os.path.join(base,
                                          f"{param_name}{ext}")

        return comments_file_path

    def normalize_spectra(self):
        """Compute normalized_data where each spectrum (row) is scaled by its maximum amplitude."""
        # Determine the maximum value for each delay (row)
        row_max = self.pump_and_probe_spectrum_timetrace.max(axis=1)
        # Divide each row by its maximum, aligning on the index
        self.normalized_pump_and_probe_spectrum_timetrace = self.pump_and_probe_spectrum_timetrace.div(row_max,
                                                                                                       axis=0)
        return self.normalized_pump_and_probe_spectrum_timetrace

    def plot_spectral_timetrace(self,
                                vmin_percentile=None,
                                vmax_percentile=None,
                                bool_plot_normalized_time_trace=False,
                                name_save_tag=None):
        """Create a 2D plot of spectral difference vs delay."""

        if bool_plot_normalized_time_trace:
            data = self.normalize_spectra()
        else:
            data = self.pump_and_probe_spectrum_timetrace

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
        if self.series_parameter is not None:
            ax.set_title(self.data_type + '_' + self.series_parameter)
        else:
            ax.set_title(self.data_type)

        if name_save_tag is None:
            if self.series_parameter is not None:
                fig.savefig(self.data_type + '_' + self.series_parameter + '_Time Trace.png')
            else:
                fig.savefig(self.data_type + '_Time Trace.png')
        else:
            if self.series_parameter is not None:
                fig.savefig(name_save_tag + self.data_type + '_' + self.series_parameter + '_Time Trace.png')
            else:
                fig.savefig(name_save_tag + self.data_type + '_Time Trace.png')

        return fig, ax

    def get_timetrace_for_given_pixel_df(self,
                                         pixel: int,
                                         number_of_pixels_for_average: int = 1):

        if not isinstance(pixel,
                          int):
            raise TypeError("Pixel must be an integer")
        if not isinstance(number_of_pixels_for_average,
                          int) or number_of_pixels_for_average < 1:
            raise ValueError("number_of_pixels_for_average must be a positive integer")

        n_cols = self.pump_and_probe_spectrum_timetrace.shape[1]
        if pixel < 0 or pixel >= n_cols:
            raise ValueError(f"Pixel {pixel} out of range (0–{n_cols - 1})")

        # --- determine column window ---------------------------------------
        half_window = number_of_pixels_for_average // 2
        start_col = max(0,
                        pixel - half_window)
        end_col = min(n_cols,
                      start_col + number_of_pixels_for_average)

        # If we're at the right edge adjust start_col to keep window size
        start_col = max(0,
                        end_col - number_of_pixels_for_average)

        # --- compute average time‑trace ------------------------------------
        averaged_timetrace = self.pump_and_probe_spectrum_timetrace.iloc[:, start_col:end_col].mean(axis=1)
        timetrace_for_given_pixel_df = pd.DataFrame({'time (ps)': self.delays,
                                                     'pp_data': averaged_timetrace})

        return timetrace_for_given_pixel_df

    def _get_unpumped_timetrace_for_given_pixel_df(self,
                                         pixel: int,
                                         number_of_pixels_for_average: int = 1):

        if not isinstance(pixel,
                          int):
            raise TypeError("Pixel must be an integer")
        if not isinstance(number_of_pixels_for_average,
                          int) or number_of_pixels_for_average < 1:
            raise ValueError("number_of_pixels_for_average must be a positive integer")

        n_cols = self.unpumped_timetrace_map.shape[1]
        if pixel < 0 or pixel >= n_cols:
            raise ValueError(f"Pixel {pixel} out of range (0–{n_cols - 1})")

        # --- determine column window ---------------------------------------
        half_window = number_of_pixels_for_average // 2
        start_col = max(0,
                        pixel - half_window)
        end_col = min(n_cols,
                      start_col + number_of_pixels_for_average)

        # If we're at the right edge adjust start_col to keep window size
        start_col = max(0,
                        end_col - number_of_pixels_for_average)

        # --- compute average time‑trace ------------------------------------
        averaged_timetrace = self.unpumped_timetrace_map.iloc[:, start_col:end_col].mean(axis=1)
        unpumped_timetrace_for_given_pixel_df = pd.DataFrame({'time (ps)': self.delays,
                                                     'pp_data': averaged_timetrace})

        return unpumped_timetrace_for_given_pixel_df

    def get_timetrace_subtracting_peaks_df(self,
                                           pixel_1: int,
                                           pixel_2: int,
                                           number_of_pixels_for_sum: int = 1):

        if not isinstance(pixel_1,
                          int):
            raise TypeError("pixel_1 must be an integer")
        if not isinstance(number_of_pixels_for_sum,
                          int) or number_of_pixels_for_sum < 1:
            raise ValueError("number_of_pixels_for_average must be a positive integer")

        n_cols = self.pump_and_probe_spectrum_timetrace.shape[1]
        if pixel_1 < 0 or pixel_1 >= n_cols:
            raise ValueError(f"pixel_1 {pixel_1} out of range (0–{n_cols - 1})")

        # --- determine column window 1 ---------------------------------------
        half_window_1 = number_of_pixels_for_sum // 2
        start_col_1 = max(0,
                          pixel_1 - half_window_1)
        end_col_1 = min(n_cols,
                        start_col_1 + number_of_pixels_for_sum)

        # If we're at the right edge adjust start_col to keep window size
        start_col_1 = max(0,
                          end_col_1 - number_of_pixels_for_sum)

        sum_timetrace_pixel_1 = self.pump_and_probe_spectrum_timetrace.iloc[:, start_col_1:end_col_1].sum(axis=1)

        # --- determine column window 2 ---------------------------------------
        half_window_2 = number_of_pixels_for_sum // 2
        start_col_2 = max(0,
                          pixel_2 - half_window_2)
        end_col_2 = min(n_cols,
                        start_col_2 + number_of_pixels_for_sum)

        # If we're at the right edge adjust start_col to keep window size
        start_col_2 = max(0,
                          end_col_2 - number_of_pixels_for_sum)

        sum_timetrace_pixel_2 = self.pump_and_probe_spectrum_timetrace.iloc[:, start_col_2:end_col_2].sum(axis=1)

        # --- compute time‑trace ------------------------------------

        time_trace = sum_timetrace_pixel_1 - sum_timetrace_pixel_2

        timetrace_subtracting_peaks_df = pd.DataFrame({'time (ps)': self.delays,
                                                       'pp_data': time_trace})

        return timetrace_subtracting_peaks_df

    def _get_timetrace_unpumped_summing_two_peaks_df(self,
                                                     pixel_1: int,
                                                     pixel_2: int,
                                                     number_of_pixels_for_sum: int = 1):

        if not isinstance(pixel_1,
                          int):
            raise TypeError("pixel_1 must be an integer")
        if not isinstance(number_of_pixels_for_sum,
                          int) or number_of_pixels_for_sum < 1:
            raise ValueError("number_of_pixels_for_average must be a positive integer")

        n_cols = self.unpumped_timetrace_map.shape[1]
        if pixel_1 < 0 or pixel_1 >= n_cols:
            raise ValueError(f"pixel_1 {pixel_1} out of range (0–{n_cols - 1})")

        # --- determine column window 1 ---------------------------------------
        half_window_1 = number_of_pixels_for_sum // 2
        start_col_1 = max(0,
                          pixel_1 - half_window_1)
        end_col_1 = min(n_cols,
                        start_col_1 + number_of_pixels_for_sum)

        # If we're at the right edge adjust start_col to keep window size
        start_col_1 = max(0,
                          end_col_1 - number_of_pixels_for_sum)

        sum_timetrace_pixel_1 = self.unpumped_timetrace_map.iloc[:, start_col_1:end_col_1].sum(axis=1)

        # --- determine column window 2 ---------------------------------------
        half_window_2 = number_of_pixels_for_sum // 2
        start_col_2 = max(0,
                          pixel_2 - half_window_2)
        end_col_2 = min(n_cols,
                        start_col_2 + number_of_pixels_for_sum)

        # If we're at the right edge adjust start_col to keep window size
        start_col_2 = max(0,
                          end_col_2 - number_of_pixels_for_sum)

        sum_timetrace_pixel_2 = self.unpumped_timetrace_map.iloc[:, start_col_2:end_col_2].sum(axis=1)

        # --- compute time‑trace ------------------------------------

        time_trace = sum_timetrace_pixel_1 + sum_timetrace_pixel_2

        timetrace_summing_peaks_df = pd.DataFrame({'time (ps)': self.delays,
                                                   'pp_data': time_trace})

        return timetrace_summing_peaks_df

    def get_differential_normalized_timetrace_subtracting_peaks_df(self,
                                                                   pixel_1: int,
                                                                   pixel_2: int,
                                                                   number_of_pixels_for_sum: int = 1):

        if not isinstance(pixel_1,
                          int):
            raise TypeError("pixel_1 must be an integer")
        if not isinstance(number_of_pixels_for_sum,
                          int) or number_of_pixels_for_sum < 1:
            raise ValueError("number_of_pixels_for_average must be a positive integer")

        n_cols = self.pump_and_probe_spectrum_timetrace.shape[1]
        if pixel_1 < 0 or pixel_1 >= n_cols:
            raise ValueError(f"pixel_1 {pixel_1} out of range (0–{n_cols - 1})")

        # --- determine column window 1 ---------------------------------------
        half_window_1 = number_of_pixels_for_sum // 2
        start_col_1 = max(0,
                          pixel_1 - half_window_1)
        end_col_1 = min(n_cols,
                        start_col_1 + number_of_pixels_for_sum)

        # If we're at the right edge adjust start_col to keep window size
        start_col_1 = max(0,
                          end_col_1 - number_of_pixels_for_sum)

        sum_timetrace_pixel_1 = self.pump_and_probe_spectrum_timetrace.iloc[:, start_col_1:end_col_1].sum(axis=1)

        # --- determine column window 2 ---------------------------------------
        half_window_2 = number_of_pixels_for_sum // 2
        start_col_2 = max(0,
                          pixel_2 - half_window_2)
        end_col_2 = min(n_cols,
                        start_col_2 + number_of_pixels_for_sum)

        # If we're at the right edge adjust start_col to keep window size
        start_col_2 = max(0,
                          end_col_2 - number_of_pixels_for_sum)

        sum_timetrace_pixel_2 = self.pump_and_probe_spectrum_timetrace.iloc[:, start_col_2:end_col_2].sum(axis=1)

        # --- compute time‑trace ------------------------------------

        time_trace = sum_timetrace_pixel_1 - sum_timetrace_pixel_2

        timetrace_subtracting_peaks_df = pd.DataFrame({'time (ps)': self.delays,
                                                       'pp_data': time_trace})

        return timetrace_subtracting_peaks_df

    def plot_timetrace_for_given_pixel(self,
                                       pixel: int,
                                       number_of_pixels_for_average: int = 1,
                                       fig=None,
                                       ax=None,
                                       bool_save_fig=True):
        """Plot the spectrum vs delay after averaging over neighbouring pixels.

        Args:
            pixel (int): central pixel index (0‑based) around which to average.
            number_of_pixels_for_average (int, optional): how many pixels to include
                in the average (must be ≥1). Defaults to 1 (single pixel).
        """
        half_window = number_of_pixels_for_average // 2
        timetrace_for_given_pixel_df = self.get_timetrace_for_given_pixel_df(pixel,
                                                                             number_of_pixels_for_average)
        averaged_timetrace = timetrace_for_given_pixel_df['pp_data']
        # --- plot -----------------------------------------------------------

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.delays,
                averaged_timetrace)
        ax.set_xlabel('Delay (ps)')
        ax.set_ylabel(f'Spectral Difference (avg {number_of_pixels_for_average} px around {pixel})')
        ax.set_title(f'Cross‑Section at Pixel {pixel} (±{half_window} px)')
        filename = f'cross_section_pixel_{pixel}_avg{number_of_pixels_for_average}.png'
        if bool_save_fig:
            fig.savefig(filename)
        fig.show()

        return fig, ax

    def plot_timetrace_for_given_pixel_returning_data(self,
                                                      pixel: int,
                                                      number_of_pixels_for_average: int = 1,
                                                      fig=None,
                                                      ax=None,
                                                      bool_save_fig=True,
                                                      bool_get_relative_signal = False):
        """Plot the spectrum vs delay after averaging over neighbouring pixels.

        Args:
            pixel (int): central pixel index (0‑based) around which to average.
            number_of_pixels_for_average (int, optional): how many pixels to include
                in the average (must be ≥1). Defaults to 1 (single pixel).
        """
        half_window = number_of_pixels_for_average // 2
        timetrace_for_given_pixel_df = self.get_timetrace_for_given_pixel_df(pixel,
                                                                             number_of_pixels_for_average)

        averaged_timetrace = timetrace_for_given_pixel_df['pp_data']

        if bool_get_relative_signal:
            unpumed_timetrace_for_give_pixel = self._get_unpumped_timetrace_for_given_pixel_df(pixel,
                                                                                               number_of_pixels_for_average)
            averaged_timetrace=100*averaged_timetrace/unpumed_timetrace_for_give_pixel['pp_data']
        # --- plot -----------------------------------------------------------

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.delays,
                averaged_timetrace)
        ax.set_xlabel('Delay (ps)')
        if bool_get_relative_signal:
            ax.set_ylabel('Relative Signal (%)')
        else:
            ax.set_ylabel(f'Signal Magnitude (a.u.)')
        ax.set_title(f'Cross‑Section at Pixel {pixel} (±{half_window} px)')
        filename = f'cross_section_pixel_{pixel}_avg{number_of_pixels_for_average}.png'
        if bool_save_fig:
            fig.savefig(filename)
        fig.show()
        delays = self.delays
        pp_data = averaged_timetrace
        return fig, ax, delays, pp_data

    def plot_timetrace_subtracting_peaks_returning_data(self,
                                                        pixel_1: int,
                                                        pixel_2: int,
                                                        number_of_pixels_for_sum: int = 1,
                                                        fig=None,
                                                        ax=None,
                                                        bool_save_fig=True):
        """Plot the spectrum vs delay after averaging over neighbouring pixels.

        Args:
            pixel (int): central pixel index (0‑based) around which to average.
            number_of_pixels_for_sum (int, optional): how many pixels to include
                in the average (must be ≥1). Defaults to 1 (single pixel).
        """
        half_window = number_of_pixels_for_sum // 2
        timetrace_for_given_pixel_df = self.get_timetrace_subtracting_peaks_df(pixel_1=pixel_1,
                                                                               pixel_2=pixel_2,
                                                                               number_of_pixels_for_sum=number_of_pixels_for_sum)
        averaged_timetrace = timetrace_for_given_pixel_df['pp_data']
        # --- plot -----------------------------------------------------------

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.delays,
                averaged_timetrace)
        ax.set_xlabel('Delay (ps)')
        ax.set_ylabel(f'Spectral Difference (avg {number_of_pixels_for_sum} px around {pixel_1} and {pixel_2})')
        ax.set_title(f'Cross‑Section at Pixel {pixel_1} and {pixel_2} (±{half_window} px)')
        filename = f'cross_section_pixel_{pixel_1}_{pixel_2}_avg{number_of_pixels_for_sum}.png'
        if bool_save_fig:
            fig.savefig(filename)
        delays = self.delays
        pp_data = averaged_timetrace
        return fig, ax, delays, pp_data

    def plot_timetrace_abs_modulation_subtracting_peaks_returning_data(self,
                                                                       pixel_1: int,
                                                                       pixel_2: int,
                                                                       number_of_pixels_for_sum: int = 1,
                                                                       fig=None,
                                                                       ax=None,
                                                                       bool_save_fig=True,
                                                                       bool_calculate_E_field_GaP=False):

        """Plot the spectrum vs delay after averaging over neighbouring pixels.

        Args:
            pixel (int): central pixel index (0‑based) around which to average.
            number_of_pixels_for_sum (int, optional): how many pixels to include
                in the average (must be ≥1). Defaults to 1 (single pixel).
        """
        half_window = number_of_pixels_for_sum // 2
        timetrace_for_given_pixel_df = self.get_timetrace_subtracting_peaks_df(pixel_1=pixel_1,
                                                                               pixel_2=pixel_2,
                                                                               number_of_pixels_for_sum=number_of_pixels_for_sum)
        timetrace_unpumped_sum_df = self._get_timetrace_unpumped_summing_two_peaks_df(pixel_1=pixel_1,
                                                                                      pixel_2=pixel_2,
                                                                                      number_of_pixels_for_sum=number_of_pixels_for_sum)

        averaged_timetrace = timetrace_for_given_pixel_df['pp_data'] / timetrace_unpumped_sum_df['pp_data']
        if bool_calculate_E_field_GaP:
            pico = 10**-12
            nano = 10**-9
            micro=10**-6
            kilo=10**3
            cm=10**-2
            probe_wavelength = 800*nano
            fresnel_coeff_GaP = 0.46
            r41=.97*pico
            n0_cubic=3.11**3
            L=200*micro

            averaged_timetrace=averaged_timetrace/(2*np.pi*n0_cubic*r41*fresnel_coeff_GaP*L/probe_wavelength)
            averaged_timetrace=averaged_timetrace/kilo#transform V in kV
            averaged_timetrace=averaged_timetrace*cm#transform m to cm
        else:
            averaged_timetrace = averaged_timetrace*100 #make percentage

        # --- plot -----------------------------------------------------------

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.delays,
                averaged_timetrace)
        ax.set_xlabel('Delay (ps)')
        if bool_calculate_E_field_GaP:
            ax.set_ylabel('Electric Field (kV/cm)')
        else:
            ax.set_ylabel(f'Relative Modulation (%)')
        ax.set_title(f'Cross‑Section at Pixel {pixel_1} and {pixel_2} (±{half_window} px)')
        filename = f'cross_section_pixel_{pixel_1}_{pixel_2}_avg{number_of_pixels_for_sum}.png'
        if bool_save_fig:
            fig.savefig(filename)
        delays = self.delays
        pp_data = averaged_timetrace
        return fig, ax, delays, pp_data

    def plot_fft(self,
                 pixel,
                 window_start=None,
                 window_end=None,
                 fig=None,
                 ax=None,
                 plot_legend=None,
                 bool_plot_normalized=False,
                 number_of_pixels_for_average=1,
                 bool_return_data=False,):
        """Calculate and plot the FFT of the time trace for a specific pixel within a selected time window.

        Args:
            pixel (int): The pixel index (zero-based) to perform FFT on.
            window_start (float, optional): The start of the delay window (ps). Defaults to delay_start.
            window_end (float, optional): The end of the delay window (ps). Defaults to delay_end.
        """


        timetrace_df = self.get_timetrace_for_given_pixel_df(pixel=pixel,
                                                             number_of_pixels_for_average=number_of_pixels_for_average)
        fft_time_interval = (window_start, window_end)
        fft_data = FFTDataClass(pump_probe_df=timetrace_df,
                                fft_time_interval=fft_time_interval)
        fft_plot = FFTPlotClass(fft_data)
        fig, ax = fft_plot.plot_fft(fig,
                                    ax,
                                    plot_legend,
                                    bool_plot_normalized)

        ax.set_title('FFT at pixel ' + str(pixel))
        if bool_return_data:
            return fig, ax, fft_data
        else:
            return fig, ax

    def plot_fft_subtracting_peaks(self,
                                   pixel_1,
                                   pixel_2,
                                   number_of_pixels_for_sum,
                                   window_start=None,
                                   window_end=None,
                                   fig=None,
                                   ax=None,
                                   plot_legend=None,
                                   bool_plot_normalized=False):
        """Calculate and plot the FFT of the time trace for a specific pixel within a selected time window.

        Args:
            pixel (int): The pixel index (zero-based) to perform FFT on.
            window_start (float, optional): The start of the delay window (ps). Defaults to delay_start.
            window_end (float, optional): The end of the delay window (ps). Defaults to delay_end.
        """
        timetrace_df = self.get_timetrace_subtracting_peaks_df(pixel_1=pixel_1,
                                                               pixel_2=pixel_2,
                                                               number_of_pixels_for_sum=number_of_pixels_for_sum)
        fft_time_interval = (window_start, window_end)
        fft_data = FFTDataClass(pump_probe_df=timetrace_df,
                                fft_time_interval=fft_time_interval)
        fft_plot = FFTPlotClass(fft_data)
        if plot_legend is None:
            plot_legend = self.series_parameter
        fig, ax = fft_plot.plot_fft(fig,
                                    ax,
                                    plot_legend,
                                    bool_plot_normalized)

        ax.set_title(self.series_parameter)

        return fig, ax

    def get_fft_df_subtracting_peaks(self,
                                     pixel_1,
                                     pixel_2,
                                     number_of_pixels_for_sum,
                                     window_start=None,
                                     window_end=None, ):

        timetrace_df = self.get_timetrace_subtracting_peaks_df(pixel_1=pixel_1,
                                                               pixel_2=pixel_2,
                                                               number_of_pixels_for_sum=number_of_pixels_for_sum)
        fft_time_interval = (window_start, window_end)
        fft_data = FFTDataClass(pump_probe_df=timetrace_df,
                                fft_time_interval=fft_time_interval)
        fft_df = fft_data.fft_df

        return fft_df

    def plot_multi_pixel_fft(self,
                             pixel_start: int,
                             pixel_end: int,
                             number_of_pixels_for_average: int = 1,
                             time_window_start: float | None = None,
                             time_window_end: float | None = None):
        """
        Plot FFTs for a series of pixel
        s (or pixel-averaged windows) on the same axes.

        Parameters
        ----------
        pixel_start : int
            First pixel index (inclusive).
        pixel_end : int
            Last pixel index (inclusive).
        number_of_pixels_for_average : int, optional
            Width of the averaging window supplied to
            `get_timetrace_for_given_pixel_df`.  Defaults to 1.
        """
        if pixel_end < pixel_start:
            raise ValueError("pixel_end must be >= pixel_start")

        # Prepare a shared figure/axes so all FFTs appear together
        fig, ax = None, None

        for pix in range(pixel_start,
                         pixel_end + 1,
                         number_of_pixels_for_average):
            # 1. get time trace (possibly averaged)
            tt_df = self.get_timetrace_for_given_pixel_df(
                pix,
                number_of_pixels_for_average
            )

            # 2. build FFT data object
            fft_obj = FFTDataClass(
                pump_probe_df=tt_df,
                fft_time_interval=(time_window_start, time_window_end)
            )

            # 3. plot FFT on the shared axes
            fft_plotter = FFTPlotClass(fft_obj)
            fig, ax = fft_plotter.plot_fft(fig=fig,
                                           ax=ax)

        # tidy up legend and title
        if ax is not None:
            pixel_labels = [
                f"px {p}"
                for p in range(pixel_start,
                               pixel_end + 1,
                               number_of_pixels_for_average)
            ]
            ax.legend(pixel_labels,
                      title="Pixel")
            time_txt = (f"time {time_window_start}–{time_window_end} ps"
                        if time_window_start is not None and time_window_end is not None
                        else "full time range")
            ax.set_title(
                f"FFT comparison, pixels {pixel_start}–{pixel_end} "
                f"(avg {number_of_pixels_for_average} px, {time_txt})"
            )

        return fig, ax

    def plot_cross_section_at_delay(self,
                                    delay,
                                    fig=None,
                                    ax=None,
                                    bool_normalize_plot=False,
                                    plot_legend=None,
                                    x_label=None,
                                    y_label=None):
        """Plot the spectral difference vs pixel for the closest available delay.

        Args:
            delay (float): The target delay time (ps). The closest available delay will be used.
        """

        # Create new figure/axes only if not provided
        if fig is None or ax is None:
            Figure = StandardFigure()
            fig, ax = Figure.fig, Figure.ax

        # Find the index of the closest delay
        idx = np.argmin(np.abs(self.delays - delay))
        selected_delay = self.delays[idx]
        print(f"Using closest delay: {selected_delay} ps")

        # Extract the spectrum at that delay
        spectrum = self.pump_and_probe_spectrum_timetrace.iloc[idx, :]
        # --- optional normalization ---------------------------------------
        if bool_normalize_plot:
            # Divide by the largest modulus value so the spectrum is scaled to ±1
            spectrum = spectrum / np.max(np.abs(spectrum))

        # Plot on the provided or new axes
        if plot_legend is not None:
            ax.plot(range(self.pump_and_probe_spectrum_timetrace.shape[1]),
                    spectrum,
                    label=plot_legend)
            ax.legend()
        else:
            ax.plot(range(self.pump_and_probe_spectrum_timetrace.shape[1]),
                    spectrum)

        self.set_x_and_y_labels(x_label,
                                y_label,
                                ax,
                                bool_normalize_plot)

        ax.set_title(f'Cross-Section at Delay {selected_delay} ps')
        # Save the plot
        filename = f'cross_section_delay_{selected_delay:.2f}.png'
        fig.savefig(filename)
        print(f"Plot saved as {filename}")
        fig.show()

        return fig, ax

    def set_x_and_y_labels(self,
                           x_label,
                           y_label,
                           ax,
                           bool_normalize_plot):

        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if x_label is None and y_label is None:
            ax.set_xlabel('Pixel Index')
            ax.set_ylabel('Counts (normalized)' if bool_normalize_plot else 'Counts')


if __name__ == "__main__":
    # Usage
    file_path = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Phononic Cross Phase '
                 'Modulation/20250416/20250416_150138_SpectralDifference_0.txt')
    data = GlazPumpProbeData_v2(file_path)
    data.plot_spectral_timetrace()
    data.plot_timetrace_for_given_pixel(600)
    data.plot_fft(600,
                  301.5,
                  307)

    data.plot_cross_section_at_delay(301.9)
