import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class IntensityMap:
    def __init__(self,
                 file_path: str,
                 camera_name = 'HIKMICRO',):
        """
        Initialize with the path to the CSV file.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load the CSV file into a pandas dataframe, automatically skipping any
        header lines (up to and including line 16).

        The routine scans the file until it encounters the first row that is
        fully numeric.  All preceding rows are treated as header / metadata and
        stored in ``self.header``.  The remaining rows are parsed with
        :pyfunc:`pandas.read_csv` and converted to a numerical
        ``numpy.ndarray`` in ``self.data``.
        """
        import itertools

        # ---------- Phase 1: detect number of header lines ------------------
        header_lines: list[str] = []
        with open(self.file_path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                # Stop if the header grows ridiculously long (safety net)
                if line_number > 32:
                    raise ValueError(
                        "Could not find a purely numeric data row within the "
                        "first 32 lines – is the file format correct?"
                    )

                stripped = line.strip()
                if not stripped:                     # blank line → header
                    header_lines.append(line.rstrip("\n"))
                    continue

                # Try to read the row with either comma or semicolon delimiter
                for delim in (",", ";"):
                    try:
                        numeric_cells = [
                            float(cell.replace(",", "."))
                            for cell in stripped.split(delim)
                        ]
                        # Success: all cells converted → data starts here
                        first_data_line = line_number
                        delimiter = delim
                        break
                    except ValueError:
                        delimiter = None
                else:
                    # At least one cell was non‑numeric → still header
                    header_lines.append(line.rstrip("\n"))
                    continue
                # Numeric row found, exit loop
                break

        # No numeric line found?
        if delimiter is None:
            raise ValueError(
                "Unable to auto‑detect numeric data rows; check the file."
            )

        self.header = header_lines if header_lines else None
        skiprows = len(header_lines)

        # ---------- Phase 2: parse the numeric matrix with pandas -----------
        self.data = (
            pd.read_csv(
                self.file_path,
                header=None,
                skiprows=skiprows,
                sep=delimiter,
                engine="python",
            ).to_numpy()
        )


    def plot(self,
             ax=None,
             title: str = 'Intensity Map'):
        """
        Plot the intensity map using matplotlib.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if ax is None:
            fig, ax = plt.subplots()
        # Display the data as an image with a colorbar
        im = ax.imshow(self.data,
                       cmap='viridis')
        ax.set_title(title)
        plt.colorbar(im,
                     ax=ax)
        return ax

if __name__ == '__main__':
    path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/NEC_camera/Spectral Dependece/LP10 Humidity 1/20250514_134932_227_001_10_01_10.csv'
    intensity_map = IntensityMap(file_path=path)
    intensity_map.load_data()
    intensity_map.plot()





