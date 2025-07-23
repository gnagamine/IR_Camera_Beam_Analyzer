import matplotlib

import math  # Added for math.floor and math.ceil

import matplotlib.pyplot as plt  # Now import pyplot
import numpy as np
import pandas as pd


class IntensityMap:
    def __init__(self,
                 file_path: str,
                 camera_name: str = 'HIKMICRO'):
        """
        Initialize with the path to the CSV file and camera name.
        """
        self.file_path = file_path
        self.camera_name = camera_name
        if camera_name == 'HIKMICRO':
            self.pixel_size_um = 13
        if camera_name == 'NEC':
            self.pixel_size_um = 23.5
        if camera_name == 'gentec':
            self.pixel_size_um = 5.5

        self.data = None
        self.header = None  # Initialize header attribute
        self.load_data()

    def load_data(self):
        """
        Load data based on the camera type.
        This is the primary method to call for loading data.
        """
        if self.camera_name == 'HIKMICRO':
            self._load_data_hikmicro()
        elif self.camera_name == 'NEC':
            self._load_data_nec()
        elif self.camera_name == 'gentec':
            self._load_data_gentec()
        # Add more camera types here with `elif self.camera_name == 'OTHER_CAMERA': self._load_data_other_camera()`
        else:
            raise ValueError(f"Unsupported camera type: {self.camera_name}. "
                             "Please implement a loading method for this camera.")

    def _is_cell_convertible_or_empty(self,
                                      cell_str: str) -> bool:
        """
        Helper function to check if a cell string can be converted to float
        or is empty (which pandas can handle as NaN).
        It handles decimal points as '.' or ','.
        """
        stripped_cell = cell_str.strip()
        if not stripped_cell:  # Empty cells are fine
            return True
        try:
            # Replace comma with dot for decimal, then attempt float conversion
            float(stripped_cell.replace(',',
                                        '.'))
            return True
        except ValueError:
            return False

    def _load_data_gentec(self):
        """
        Load data for Gentec camera.
        It skips header lines, extracts pixel size if available,
        and reads the semicolon-separated integer data.
        NaNs are replaced with 0.0.
        """
        header_lines = []
        data_lines_start_idx = -1
        temp_pixel_size_um = None

        try:
            with open(self.file_path, "r", encoding="utf-8") as fh:
                for i, line_content in enumerate(fh):
                    stripped_line = line_content.strip()
                    if not stripped_line:  # Skip empty lines
                        header_lines.append(line_content.rstrip("\n"))
                        continue

                    if "Pixel Size:" in stripped_line:
                        try:
                            # Attempt to extract pixel size
                            temp_pixel_size_um = float(stripped_line.split(":")[1].strip())
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Could not parse Pixel Size from line: '{stripped_line}'. Error: {e}")
                        header_lines.append(line_content.rstrip("\n"))
                        continue

                    # Check if the line looks like data (all elements are numbers when split by ';')
                    cells = stripped_line.split(';')
                    if all(cell.strip().isdigit() for cell in cells if cell.strip()): # Check if all non-empty cells are digits
                        data_lines_start_idx = i
                        break  # Found the first data line
                    else:
                        header_lines.append(line_content.rstrip("\n"))

            if data_lines_start_idx == -1:
                raise ValueError("Could not find the start of data lines in Gentec file.")

            self.header = header_lines if header_lines else None
            if temp_pixel_size_um is not None:
                self.pixel_size_um = temp_pixel_size_um
                print(f"Gentec pixel size updated to: {self.pixel_size_um} um from file header.")
            elif self.pixel_size_um is None: # If not set by constructor and not found in header
                 print("Warning: Pixel size for Gentec not found in header and not set during initialization. "
                       "Cropping in 'um' might not work as expected.")


            # Read the data section
            loaded_data_pd = pd.read_csv(
                self.file_path,
                header=None,
                skiprows=data_lines_start_idx,
                sep=';',
                engine="python",
                on_bad_lines='warn',
                dtype=float # Assume data is numeric, pandas will try to convert
            )
            self.data = np.nan_to_num(loaded_data_pd.to_numpy(), nan=0.0)

        except FileNotFoundError:
            raise FileNotFoundError(f"Gentec data file not found: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Error processing Gentec file {self.file_path}: {e}")

    def _load_data_hikmicro(self):
        """
        Load the CSV file for HIKMICRO, automatically skipping any header lines.
        This version has improved header detection to correctly handle lines
        with only commas and is robust to empty cells and lines with an
        incorrect number of fields in the data section. NaNs are replaced with 0.0.
        """
        header_lines: list[str] = []
        detected_delimiter = None
        first_data_line_num = -1
        line_is_numeric_for_a_delimiter = False  # Flag to break outer loop

        try:
            with open(self.file_path,
                      "r",
                      encoding="utf-8") as fh:
                for line_number, line_content in enumerate(fh,
                                                           start=1):
                    if line_number > 32:  # Safety net for HIKMICRO header length
                        raise ValueError(
                            "Could not find a purely numeric data row within the "
                            "first 32 lines for HIKMICRO – is the file format correct or header too long?"
                        )

                    stripped_line = line_content.strip()

                    if not stripped_line:  # Blank line is considered part of header
                        header_lines.append(line_content.rstrip("\n"))
                        continue

                    # Try common delimiters, prioritizing comma for HIKMICRO
                    for delim_char_to_try in (",", ";"):
                        cells = stripped_line.split(delim_char_to_try)

                        if not cells:  # Should not occur if stripped_line is not empty
                            continue

                        # Check if all cells are either empty or convertible to float
                        all_cells_valid_format = all(self._is_cell_convertible_or_empty(cell) for cell in cells)

                        if all_cells_valid_format:
                            # If all cells are valid (empty or numeric),
                            # ensure it's not a line of pure commas (which results in all empty strings after split).
                            # A true data line must contain at least one cell that isn't just whitespace.
                            is_line_of_only_empty_cells_after_split = all(not cell.strip() for cell in cells)

                            if not is_line_of_only_empty_cells_after_split:
                                # This is identified as the first data line.
                                detected_delimiter = delim_char_to_try
                                first_data_line_num = line_number
                                line_is_numeric_for_a_delimiter = True
                                break  # Break from delimiter check loop
                            # else: it's a line like ",,," which is still header/metadata
                        # else (not all_cells_valid_format):
                        #   This means at least one cell is non-empty and non-numeric, so it's a header.
                        #   Continue to the next delimiter or next line if this was the last delimiter.

                    if line_is_numeric_for_a_delimiter:
                        # First data line found, break from line enumeration loop
                        break
                    else:
                        # This line is confirmed as a header line (either non-numeric parts or only commas)
                        header_lines.append(line_content.rstrip("\n"))

        except FileNotFoundError:
            raise FileNotFoundError(f"HIKMICRO data file not found: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Error processing HIKMICRO file {self.file_path} during header detection: {e}")

        if detected_delimiter is None or first_data_line_num == -1:
            raise ValueError(
                "Unable to auto‑detect the start of numeric data rows or delimiter for HIKMICRO. "
                "Please check the file format."
            )

        self.header = header_lines if header_lines else None
        skiprows = first_data_line_num - 1

        try:
            loaded_data_pd = pd.read_csv(
                self.file_path,
                header=None,
                skiprows=skiprows,
                sep=detected_delimiter,
                engine="python",
                on_bad_lines='warn'
            )
            # Convert to numpy array and replace any NaNs with 0.0
            self.data = np.nan_to_num(loaded_data_pd.to_numpy(),
                                      nan=0.0)

        except Exception as e:
            raise RuntimeError(f"Pandas failed to parse HIKMICRO data from {self.file_path} "
                               f"or convert to NumPy (skiprows={skiprows}, delimiter='{detected_delimiter}'): {e}")

    def _load_data_nec(self):
        """
        Load data for NEC camera.
        This is a basic implementation assuming CSV with NO header.
        It tries common delimiters (',' or ';') and decimal styles ('.' or ',').
        NaNs are replaced with 0.0.
        The first and last columns are removed from the final data.
        """
        self.header = None

        configurations = [
            {"sep": ',', "decimal": '.'},
            {"sep": ';', "decimal": '.'},
            {"sep": ';', "decimal": ','}
        ]

        last_error = None
        data_loaded_successfully = False

        for config in configurations:
            try:
                loaded_data_pd = pd.read_csv(
                    self.file_path,
                    header=None,
                    sep=config["sep"],
                    decimal=config["decimal"],
                    engine='python',
                    skipinitialspace=True,
                    on_bad_lines='warn'
                )
                # Convert to numpy array and replace any NaNs with 0.0
                temp_data = np.nan_to_num(loaded_data_pd.to_numpy(),
                                          nan=0.0)

                # Remove the first and last columns if data is not empty and has enough columns
                if temp_data.ndim == 2 and temp_data.shape[1] > 2:
                    self.data = temp_data[:, 1:-1]
                elif temp_data.ndim == 2 and temp_data.shape[1] <= 2:
                    # If 2 or fewer columns, it's unclear how to remove first AND last.
                    # Keep as is or raise error/warning. For now, keep as is.
                    print(f"Warning: NEC data has {temp_data.shape[1]} columns. Cannot remove both first and last. "
                          f"Using data as is.")
                    self.data = temp_data
                else:  # If not 2D (e.g. 1D or empty), keep as is.
                    self.data = temp_data

                data_loaded_successfully = True
                return  # Exit after successful load
            except FileNotFoundError:
                raise FileNotFoundError(f"NEC data file not found: {self.file_path}")
            except (pd.errors.ParserError, ValueError, TypeError) as e:
                last_error = e
            except Exception as e:  # Catch any other unexpected error
                last_error = e

        if not data_loaded_successfully:
            if last_error is not None:
                raise ValueError(
                    f"Failed to parse NEC data from '{self.file_path}' with all attempted configurations. "
                    f"Last error: {last_error}"
                )
            else:
                raise ValueError(f"Failed to parse NEC data from '{self.file_path}', and no specific parsing error "
                                 f"was caught.")

    def plot(self,
             ax=None,
             title: str = 'Intensity Map'):
        """
        Plot the intensity map using matplotlib.
        The color scale is automatically adjusted using percentiles to enhance visibility.
        """

        self.load_data()
        # print(self.data) # This was the line you added for debugging
        if self.data is None:
            load_method_suggestion = f"load_data() for camera '{self.camera_name}'"
            raise ValueError(f"Data not loaded. Call {load_method_suggestion} first.")

        if not isinstance(self.data,
                          np.ndarray) or not np.issubdtype(self.data.dtype,
                                                           np.number):
            try:
                self.data = np.nan_to_num(np.array(self.data,
                                                   dtype=float),
                                          nan=0.0)
            except ValueError as e:
                raise ValueError(f"Data is not purely numeric and could not be converted. Error: {e}. "
                                 "Please check the loaded data content.")

        if self.data.size == 0:  # Handle empty data array after loading/conversion
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
            plt.show()
            return fig, ax

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()  # Get the figure if ax is provided

        # Determine vmin and vmax from percentiles for robust auto-scaling
        # This helps to ignore extreme outliers and focus on the main data range.
        # If data is completely flat, vmin and vmax will be the same; imshow handles this.
        vmin = np.percentile(self.data,
                             0)  # Using 0th percentile (min)
        vmax = np.percentile(self.data,
                             100)  # Using 100th percentile (max)

        # If vmin and vmax are the same (e.g., flat data or after percentile clipping on near-flat data),
        # adjust them slightly to prevent issues with some backends or color mapping.
        if vmin == vmax:
            vmin = vmin - 0.5  # Adjust by a small amount
            vmax = vmax + 0.5
            if vmin == vmax:  # If still same (e.g. vmin was 0, now -0.5, vmax 0.5, but if original was 0, vmin=vmax=0)
                # Or if data was truly flat non-zero, e.g. all 10s.
                if vmax == 0:  # if data was all zeros
                    vmax = 1.0  # give it some range if it was all zeros.
                else:  # if data was flat but non-zero, give a small range around it
                    vmax = vmax + 0.1 * abs(vmax) if vmax != 0 else 0.1
                    vmin = vmin - 0.1 * abs(vmin) if vmin != 0 else -0.1

        im = ax.imshow(self.data,
                       cmap='viridis',
                       interpolation='nearest',
                       vmin=vmin,
                       # Set explicit min for color scale
                       vmax=vmax)  # Set explicit max for color scale

        ax.set_title(title if title != 'Intensity Map' else f'{self.camera_name} Intensity Map')

        plt.colorbar(im,
                     # Use plt.colorbar directly
                     ax=ax,
                     label='Intensity')

        plt.show()  # Use plt.show() to display the plot and run GUI event loop
        return fig, ax

    def crop_data_um(self,
                     x_range_um: tuple[float | None, float | None] | None = None,
                     y_range_um: tuple[float | None, float | None] | None = None):
        """
        Crops the self.data numpy array based on specified ranges in micrometers.

        The input coordinate system for `x_range_um` and `y_range_um` assumes an origin
        at the **bottom-left** of the image area.
        - X increases to the right (columns).
        - Y increases upwards (rows).

        For example, `y_range_um=(0, 100)` would select the bottom 100 micrometers of the image.

        Note: The underlying `self.data` NumPy array is indexed such that `self.data[0,0]`
        is the top-left pixel. The conversion from the bottom-left input coordinate
        system to array indices is handled internally by this method.

        Args:
            x_range_um (tuple[float | None, float | None] | None, optional):
                `(xmin_um, xmax_um)` for the x-axis (columns), interpreted from left edge.
                `xmin_um`: Minimum x-coordinate in micrometers (inclusive).
                           If None, effectively 0 (leftmost).
                `xmax_um`: Maximum x-coordinate in micrometers (exclusive).
                           If None, effectively the full width of the data.
                If the entire `x_range_um` tuple is None, no cropping on the x-axis.
            y_range_um (tuple[float | None, float | None] | None, optional):
                `(ymin_um, ymax_um)` for the y-axis (rows), interpreted from bottom edge.
                `ymin_um`: Minimum y-coordinate in micrometers from the bottom (inclusive).
                           If None, effectively 0 (bottommost).
                `ymax_um`: Maximum y-coordinate in micrometers from the bottom (exclusive).
                           If None, effectively the full height of the data (up to the top).
                If the entire `y_range_um` tuple is None, no cropping on the y-axis.

        Raises:
            ValueError: If data is not loaded or pixel_size_um is invalid.
        """
        if self.data is None:
            load_method_suggestion = f"load_data() for camera '{self.camera_name}'"
            raise ValueError(f"Data not loaded. Call {load_method_suggestion} first before cropping.")

        if self.data.size == 0:
            print("Warning: Data is already empty. No crop operation performed.")
            return

        if self.pixel_size_um is None or self.pixel_size_um <= 0:
            raise ValueError(
                f"Pixel size (self.pixel_size_um = {self.pixel_size_um}) is not defined or is invalid for camera '{self.camera_name}'. "
                "Cannot perform crop in micrometers."
            )

        num_rows, num_cols = self.data.shape

        # Default slice indices: no cropping (full range)
        col_start_idx, col_end_idx = 0, num_cols
        row_start_idx, row_end_idx = 0, num_rows

        # Calculate column (x-axis) slice indices (origin left)
        if x_range_um is not None:
            xmin_um, xmax_um = x_range_um
            # Effective x_min in um from left (0 if None)
            x_physical_min_um = 0.0 if xmin_um is None else xmin_um
            # Effective x_max in um from left (total width if None)
            x_physical_max_um = (num_cols * self.pixel_size_um) if xmax_um is None else xmax_um

            col_start_idx = math.floor(x_physical_min_um / self.pixel_size_um)
            col_end_idx = math.ceil(x_physical_max_um / self.pixel_size_um)

        # Calculate row (y-axis) slice indices (origin bottom)
        if y_range_um is not None:
            ymin_um_input, ymax_um_input = y_range_um

            # Effective y_min in um from bottom (0 if None)
            y_physical_min_um_from_bottom = 0.0 if ymin_um_input is None else ymin_um_input
            # Effective y_max in um from bottom (total height if None)
            y_physical_max_um_from_bottom = (num_rows * self.pixel_size_um) if ymax_um_input is None else ymax_um_input

            # Convert physical y (from bottom=0) to pixel distances from bottom
            # Smallest pixel index from bottom (inclusive start of range)
            y_pixel_dist_from_bottom_start = math.floor(y_physical_min_um_from_bottom / self.pixel_size_um)
            # Largest pixel index from bottom (exclusive end of range)
            y_pixel_dist_from_bottom_end = math.ceil(y_physical_max_um_from_bottom / self.pixel_size_um)

            # Convert distances from bottom to array row indices (where row 0 is top)
            # row_start_idx for slice corresponds to the upper physical boundary (ymax_um_input)
            row_start_idx = num_rows - y_pixel_dist_from_bottom_end
            # row_end_idx for slice corresponds to the lower physical boundary (ymin_um_input)
            row_end_idx = num_rows - y_pixel_dist_from_bottom_start

        # Clamp indices to be within the array dimensions
        col_start_idx = max(0,
                            col_start_idx)
        col_end_idx = min(num_cols,
                          col_end_idx)
        row_start_idx = max(0,
                            row_start_idx)
        row_end_idx = min(num_rows,
                          row_end_idx)

        if col_start_idx >= col_end_idx:  # Use >= to also catch zero-width selections
            print(f"Warning: Calculated x-axis crop indices [{col_start_idx}:{col_end_idx}] are invalid or result in "
                  f"an empty selection. Resulting x-dimension will be empty.")
            col_end_idx = col_start_idx  # Ensure empty slice if invalid

        if row_start_idx >= row_end_idx:  # Use >= to also catch zero-height selections
            print(f"Warning: Calculated y-axis crop indices [{row_start_idx}:{row_end_idx}] are invalid or result in "
                  f"an empty selection. Resulting y-dimension will be empty.")
            row_end_idx = row_start_idx  # Ensure empty slice if invalid

        original_shape = self.data.shape
        self.data = self.data[row_start_idx:row_end_idx, col_start_idx:col_end_idx]

    def crop_data_pixels(self,
                     x_range_pixels = None,
                     y_range_pixels = None):
        """
        Crops the self.data numpy array based on specified ranges in pixels.


        Raises:
            ValueError: If data is not loaded or pixel_size_um is invalid.
        """
        if self.data is None:
            load_method_suggestion = f"load_data() for camera '{self.camera_name}'"
            raise ValueError(f"Data not loaded. Call {load_method_suggestion} first before cropping.")

        if self.data.size == 0:
            print("Warning: Data is already empty. No crop operation performed.")
            return


        num_rows, num_cols = self.data.shape

        # Default slice indices: no cropping (full range)
        col_start_idx, col_end_idx = 0, num_cols
        row_start_idx, row_end_idx = 0, num_rows

        # Calculate column (x-axis) slice indices (origin left)
        if x_range_pixels is not None:
            xmin_um, xmax_um = x_range_pixels
            xmax_um=int(xmax_um)
            xmin_um=int(xmin_um)
            # Effective x_min in um from left (0 if None)
            x_physical_min_um = 0.0 if xmin_um is None else xmin_um
            # Effective x_max in um from left (total width if None)
            x_physical_max_um = (num_cols ) if xmax_um is None else xmax_um

            col_start_idx = math.floor(x_physical_min_um)
            col_end_idx = math.ceil(x_physical_max_um)

        # Calculate row (y-axis) slice indices (origin bottom)
        if y_range_pixels is not None:
            ymin_um_input, ymax_um_input = y_range_pixels
            ymin_um_input=int(ymin_um_input)
            ymax_um_input=int(ymax_um_input)

            # Effective y_min in um from bottom (0 if None)
            y_physical_min_um_from_bottom = 0.0 if ymin_um_input is None else ymin_um_input
            # Effective y_max in um from bottom (total height if None)
            y_physical_max_um_from_bottom = (num_rows) if ymax_um_input is None else ymax_um_input

            # Convert physical y (from bottom=0) to pixel distances from bottom
            # Smallest pixel index from bottom (inclusive start of range)
            y_pixel_dist_from_bottom_start = math.floor(y_physical_min_um_from_bottom)
            # Largest pixel index from bottom (exclusive end of range)
            y_pixel_dist_from_bottom_end = math.ceil(y_physical_max_um_from_bottom)

            # Convert distances from bottom to array row indices (where row 0 is top)
            # row_start_idx for slice corresponds to the upper physical boundary (ymax_um_input)
            row_start_idx = num_rows - y_pixel_dist_from_bottom_end
            # row_end_idx for slice corresponds to the lower physical boundary (ymin_um_input)
            row_end_idx = num_rows - y_pixel_dist_from_bottom_start

        # Clamp indices to be within the array dimensions
        col_start_idx = max(0,
                            col_start_idx)
        col_end_idx = min(num_cols,
                          col_end_idx)
        row_start_idx = max(0,
                            row_start_idx)
        row_end_idx = min(num_rows,
                          row_end_idx)

        if col_start_idx >= col_end_idx:  # Use >= to also catch zero-width selections
            print(f"Warning: Calculated x-axis crop indices [{col_start_idx}:{col_end_idx}] are invalid or result in "
                  f"an empty selection. Resulting x-dimension will be empty.")
            col_end_idx = col_start_idx  # Ensure empty slice if invalid

        if row_start_idx >= row_end_idx:  # Use >= to also catch zero-height selections
            print(f"Warning: Calculated y-axis crop indices [{row_start_idx}:{row_end_idx}] are invalid or result in "
                  f"an empty selection. Resulting y-dimension will be empty.")
            row_end_idx = row_start_idx  # Ensure empty slice if invalid

        original_shape = self.data.shape
        self.data = self.data[row_start_idx:row_end_idx, col_start_idx:col_end_idx]



if __name__ == '__main__':
    # Example for HIKMICRO camera
    # path_hik_example = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz
    # Camera/20250508/Power_series_old/IR_00045_50degrees.csv'
    # #path_hik_example = 'IR_00040_90degrees.csv'
    #
    # print(f"Attempting to load HIKMICRO camera data from: {path_hik_example}")
    # intensity_map_hik = IntensityMap(file_path=path_hik_example,
    #                                  camera_name='HIKMICRO')
    # try:
    #     intensity_map_hik.load_data()
    #     print(f"HIKMICRO data loaded successfully from {path_hik_example}.")
    #     print(f"Header lines found: {len(intensity_map_hik.header) if intensity_map_hik.header else 0}")
    #     print(f"Data shape after NaN to zero conversion: {intensity_map_hik.data.shape}")
    #
    #     fig_hik, ax_hik = intensity_map_hik.plot(title='HIKMICRO Camera Intensity Map Example')
    #
    # except Exception as e:
    #     print(f"Error loading or plotting HIKMICRO data from {path_hik_example}: {e}")
    #     import traceback
    #
    #     traceback.print_exc()
    #
    # print("-" * 30)

    # Example for NEC camera
    path_nec_example = ('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz '
                        'Camera/20250516/position series_NEC/each_delay/1,35 mm.csv')

    intensity_map_nec = IntensityMap(file_path=path_nec_example,
                                     camera_name='NEC',
                                     crop_x_range_um=(-0.5, 0.5),
                                     crop_y_range_um=(-0.5, 0.5))
    intensity_map_nec.load_data()
    data = intensity_map_nec.data
