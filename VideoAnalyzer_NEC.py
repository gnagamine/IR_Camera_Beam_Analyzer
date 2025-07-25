import os

import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
from sympy.solvers.diophantine.diophantine import length

from IntensityMapIRCamera import IntensityMap
matplotlib.use('macosx')

# Removed duplicate imports of numpy and matplotlib.pyplot

class VideoAnalyzer_NEC:  # Renamed for clarity, as it now handles std dev
    def __init__(self,
                 video_directory,
                 camera_name='NEC', ):
        """Initialize the TemperatureMapAnalyzer with the path to the CSV file."""
        self.video_directory = video_directory

    def compute_std_dev_map(self):
        """Compute the standard deviation map from the temperature maps in the CSV file."""
        files_name_list = os.listdir(self.video_directory)
        maps_array_list=[]

        for filename in files_name_list:
            if filename.endswith('.csv') and not filename.startswith('.'):
                print(f'loading: {filename}')
                file_path = os.path.join(self.video_directory, filename)
                Map = IntensityMap(file_path=file_path,
                    camera_name='NEC')
                map_array=Map.data
                maps_array_list.append(map_array)

        # Stack all frames into a 3D array: (num_frames, height, width)
        print(f'Number of frames analyzed: {len(maps_array_list)}')
        stacked_maps = np.stack(maps_array_list,
                                axis=0)

        # Compute standard deviation along the time axis (axis=0)
        # Use ddof=0 for population standard deviation (common for noise analysis)
        std_dev_map = np.std(stacked_maps,
                             axis=0,
                             ddof=0)

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
                     label='counts')
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
                     label='counts')
        ax.set_title(f'Cropped Standard Deviation Map (Center: {center_x}, {center_y}, Width: {crop_width_pixels}) \n'
                     f'Std Average: {np.mean(cropped_std_dev_map):.3f} counts')
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
