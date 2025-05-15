import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
try:
    matplotlib.use("MacOSX")
except Exception:
    matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

class RMSMapAnalyzer:
    def __init__(self,
                 csv_path,
                 camera_name = 'HIKMICRO',):
        """Initialize the RMSMapAnalyzer with the path to the CSV file."""
        self.csv_path = csv_path
        if camera_name == 'HIKMICRO':
            self.header_rows = 4
            self.matrix_rows = 192
            self.matrix_cols = 256

    def compute_rms_map(self):
        """Compute the RMS map from the temperature maps in the CSV file."""
        with open(self.csv_path, 'r') as f:
            # Skip the initial 4 header rows
            for _ in range(self.header_rows):
                f.readline()

            sum_of_squares = None
            num_frames = 0

            while True:
                line = f.readline()
                if not line:  # End of file
                    break

                if line.strip().startswith('absolute time:'):
                    # Skip the next 'time:...' row
                    f.readline()

                    # Read the 192x256 matrix
                    matrix = []
                    for _ in range(self.matrix_rows):
                        row_line = f.readline().strip()  # Read and strip the line
                        parts = row_line.split(',')  # Split by commas
                        # Remove trailing empty strings
                        while parts and parts[-1] == '':
                            parts.pop()
                        # Convert to floats, handling empty fields as 0.0
                        row = [float(x.strip()) if x.strip() else 0.0 for x in parts]
                        if len(row) != self.matrix_cols:
                            raise ValueError(f"Expected {self.matrix_cols} columns, got {len(row)}")
                        matrix.append(row)
                    matrix = np.array(matrix, dtype=np.float64)

                    # Update the sum of squares
                    if sum_of_squares is None:
                        sum_of_squares = np.square(matrix)
                    else:
                        sum_of_squares += np.square(matrix)
                    num_frames += 1

                    # Skip the empty row
                    empty_line = f.readline().strip()
                    if empty_line:
                        print(f"Warning: Expected empty line, got: '{empty_line}'")

            if num_frames == 0:
                raise ValueError("No frames found in the CSV file")

            # Compute RMS map: sqrt(mean of squares)
            rms_map = np.sqrt(sum_of_squares / num_frames)
            return rms_map

    def display_rms_map(self):
        """Compute and display the RMS map using Matplotlib."""
        rms_map = self.compute_rms_map()
        plt.imshow(rms_map, cmap='viridis')
        plt.colorbar(label='RMS Temperature')
        plt.title('RMS Map of THz Camera')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()

# Example usage
if __name__ == "__main__":
    analyzer = RMSMapAnalyzer('/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Noise_Analysis/IR_00002_Temperature Matrix.csv')
    analyzer.display_rms_map()

# class VideoRMSAnalyzer:
#     """Compute RMS noise statistics from a CSV video stack.
#
#     Args:
#         filepath (str): Path to the CSV file.
#         rows (int): Number of rows in each video frame.
#         cols (int): Number of columns in each video frame.
#         delimiter (str): Delimiter used in the CSV file.
#     """
#
#     def __init__(self, filepath, rows, cols, delimiter=","):
#         self.filepath = filepath
#         self.rows = rows
#         self.cols = cols
#         self.delimiter = delimiter
#         self.frames = self._load_frames()
#
#     def _load_frames(self):
#         frame_rows = []
#         with open(self.filepath, newline='') as csvfile:
#             reader = csv.reader(csvfile, delimiter=self.delimiter)
#             for row in reader:
#                 if len(row) == self.cols:
#                     frame_rows.append([float(val) for val in row])
#                 if len(frame_rows) == self.rows:
#                     yield np.array(frame_rows)
#                     frame_rows = []
#         if frame_rows:
#             yield np.array(frame_rows)
#
#     def compute_rms(self):
#         rms_values = []
#         for frame in self.frames:
#             rms = np.sqrt(np.mean(np.square(frame)))
#             rms_values.append(rms)
#         return rms_values
#
#     def plot_rms(self):
#         rms_values = self.compute_rms()
#         plt.plot(rms_values)
#         plt.xlabel("Frame")
#         plt.ylabel("RMS Noise")
#         plt.title("RMS Noise per Frame")
#         plt.show()
#         print('plotted')
#
# if __name__ == "__main__":
#     filepath = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Noise_Analysis/IR_00002_Temperature Matrix.csv'
#     analyzer = VideoRMSAnalyzer(
#         filepath,
#         rows=192,
#         cols=256,
#         delimiter=";",
#     )
#     analyzer.plot_rms()
