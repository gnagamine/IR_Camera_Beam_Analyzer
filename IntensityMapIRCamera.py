import matplotlib

try:
    # Attempt to use TkAgg backend, which usually works well for GUIs.
    # This MUST be called before importing pyplot.
    matplotlib.use('TkAgg')
except ImportError:
    print("Warning: TkAgg backend not available. Plot window may not appear.")
    print("Consider installing tkinter in your Python environment (e.g., 'pip install tk').")
    print("Falling back to a non-interactive backend (Agg).")
    try:
        matplotlib.use('Agg')  # Fallback to a non-interactive backend
    except ImportError:
        print("Warning: Agg backend also not available. Plotting might fail.")
        # Matplotlib will try its default if Agg also fails.

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
        self.data = None
        self.header = None  # Initialize header attribute

    def load_data(self):
        """
        Load data based on the camera type.
        This is the primary method to call for loading data.
        """
        if self.camera_name == 'HIKMICRO':
            self._load_data_hikmicro()
        elif self.camera_name == 'NEC':
            self._load_data_nec()
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
                    print(f"Warning: NEC data has {temp_data.shape[1]} columns. Cannot remove both first and last. Using data as is.")
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
                raise ValueError(f"Failed to parse NEC data from '{self.file_path}', and no specific parsing error was caught.")
        print(self.data)

    def plot(self,
             ax=None,
             title: str = 'Intensity Map'):
        """
        Plot the intensity map using matplotlib.
        The color scale is automatically adjusted using percentiles to enhance visibility.
        """
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


if __name__ == '__main__':
    # Example for HIKMICRO camera
    # path_hik_example = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Power_series_old/IR_00045_50degrees.csv'
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
    path_nec_example ='/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250516/position series_NEC/each_delay/1,35 mm.csv'
    if path_nec_example and "Gabriel_UniBern_Local" in path_nec_example:  # Basic check if path seems valid
        print(f"Attempting to load NEC camera data from: {path_nec_example}")
        intensity_map_nec = IntensityMap(file_path=path_nec_example,
                                         camera_name='NEC')
        try:
            intensity_map_nec.load_data()
            print(f"NEC data loaded successfully from {path_nec_example}.")
            print(f"Original NEC data shape: {intensity_map_nec.data.shape if intensity_map_nec.data is not None else 'None'}")  # Check shape before potential modification

            # The column removal is now inside _load_data_nec

            print(f"Final NEC data shape for plotting: {intensity_map_nec.data.shape if intensity_map_nec.data is not None else 'None'}")
            intensity_map_nec.plot(title='NEC Camera Intensity Map')
        except Exception as e:
            print(f"Error loading or plotting NEC data from {path_nec_example}: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"Skipping NEC camera example as path seems incorrect or not set: {path_nec_example}")

