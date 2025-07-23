import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
matplotlib.use('macosx')

# Removed duplicate imports of numpy and matplotlib.pyplot

class VideoAnalyzer:  # Renamed for clarity, as it now handles std dev
    def __init__(self,
                 csv_path,
                 camera_name='HIKMICRO', ):
        """Initialize the TemperatureMapAnalyzer with the path to the CSV file."""
        self.csv_path = csv_path
        if camera_name == 'HIKMICRO':
            self.header_rows = 4
            self.matrix_rows = 192
            self.matrix_cols = 256
        # Add other camera configurations here if needed
        # else:
        #     raise ValueError(f"Unsupported camera_name: {camera_name}")

    def compute_std_dev_map(self):
        """Compute the standard deviation map from the temperature maps in the CSV file."""
        with open(self.csv_path,
                  'r') as f:
            # Skip the initial header rows
            for _ in range(self.header_rows):
                f.readline()

            sum_of_values = None
            sum_of_squares = None
            num_frames = 0

            while True:
                line = f.readline()
                if not line:  # End of file
                    break

                if line.strip().startswith('absolute time:'):
                    # Skip the next 'time:...' row
                    f.readline()

                    # Read the matrix
                    matrix_data = []
                    for _ in range(self.matrix_rows):
                        row_line = f.readline().strip()
                        parts = row_line.split(',')
                        while parts and parts[-1] == '':  # Handle trailing commas
                            parts.pop()
                        # Convert to floats, handling empty fields as  np.nan

                        row = [float(x.strip()) if x.strip() else np.nan for x in parts]

                        if len(row) != self.matrix_cols:
                            raise ValueError(
                                f"Expected {self.matrix_cols} columns, got {len(row)} in row: {row_line[:50]}..."
                            )
                        matrix_data.append(row)

                    current_matrix = np.array(matrix_data,
                                              dtype=np.float64)

                    # Update sums for standard deviation calculation
                    if sum_of_values is None:
                        sum_of_values = current_matrix
                        sum_of_squares = np.square(current_matrix)
                    else:
                        sum_of_values += current_matrix
                        sum_of_squares += np.square(current_matrix)
                    num_frames += 1

                    # Skip the empty row
                    empty_line = f.readline().strip()
                    if empty_line:
                        print(f"Warning: Expected empty line after matrix, got: '{empty_line}'")

            if num_frames == 0:
                raise ValueError("No frames found in the CSV file")
            if num_frames == 1:
                # Standard deviation is not well-defined for a single frame,
                # or will be zero depending on implementation.
                # np.std with ddof=0 will return 0.
                # Let's return a zero map of the correct shape.
                print("Warning: Only one frame found. Standard deviation map will be all zeros.")
                return np.zeros_like(sum_of_values)

            # Compute standard deviation map: sqrt(mean_of_squares - square_of_mean)
            mean_of_values = sum_of_values / num_frames
            mean_of_squares = sum_of_squares / num_frames

            # Variance = E[X^2] - (E[X])^2
            variance_map = mean_of_squares - np.square(mean_of_values)
            # Ensure variance is not negative due to floating point inaccuracies
            variance_map[variance_map < 0] = 0
            std_dev_map = np.sqrt(variance_map)

            return std_dev_map

    def plot_std_dev_map(self):
        """Compute and display the Standard Deviation map using Matplotlib."""
        std_dev_map = self.compute_std_dev_map()
        fig, ax = plt.subplots()
        im = ax.imshow(std_dev_map,
                       cmap='viridis',
                       origin='lower')  # Or 'plasma', 'magma'
        plt.colorbar(im,
                     ax=ax,
                     label='Standard Deviation of Temperature (°C or units)')
        ax.set_title('Standard Deviation Map of THz Camera')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        plt.show()

        return fig, ax

    def plot_std_dev_cropped(self,
                             crop_width_pixels,
                             center_x,
                             center_y):
        """
        Plot the standard deviation in a cropped region of the map.

        Args:
            crop_width_pixels (int): Width of the cropping window in pixels.
            center_x (int): X-coordinate of the center of the cropping window.
            center_y (int): Y-coordinate of the center of the cropping window.
        """
        # Compute the standard deviation map
        std_dev_map = self.compute_std_dev_map()

        # Calculate the cropping boundaries
        half_width = crop_width_pixels // 2
        x_min = max(0,
                    center_x - half_width)
        x_max = min(std_dev_map.shape[1],
                    center_x + half_width)
        y_min = max(0,
                    center_y - half_width)
        y_max = min(std_dev_map.shape[0],
                    center_y + half_width)

        # Crop the standard deviation map
        cropped_std_dev_map = std_dev_map[y_min:y_max, x_min:x_max]

        # Plot the cropped standard deviation map
        fig, ax = plt.subplots()
        im = ax.imshow(cropped_std_dev_map,
                       cmap='viridis',
                       origin='lower')
        plt.colorbar(im,
                     ax=ax,
                     label='Standard Deviation of Temperature (°C or units)')
        ax.set_title(f'Cropped Standard Deviation Map (Center: {center_x}, {center_y}, Width: {crop_width_pixels}) \n'
                     f'Std Average: {np.mean(cropped_std_dev_map):.3f} °C')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        plt.show()

        return fig, ax, cropped_std_dev_map


# Example usage
if __name__ == "__main__":
    # Ensure the path to your CSV file is correct
    csv_file_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Noise_Analysis/IR_00002_Temperature Matrix.csv'
    analyzer = VideoAnalyzer(csv_file_path)

    # Display the full standard deviation map
    analyzer.plot_std_dev_map()

    # Example of using the new method to plot a cropped region
    # Assuming the center of interest is at (128, 96) and we want a 50x50 pixel crop
    analyzer.plot_std_dev_cropped(crop_width_pixels=50, center_x=70+25, center_y=124+25)
