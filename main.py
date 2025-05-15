import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IntensityMapIRCamera import IntensityMap
import os

class Analysis:
    def __init__(self,
                 dir_path,
                 signal_filename,
                 background_filename):
        """
        Initialize with file paths for the signal and background CSV files.
        """
        signal_path =  os.path.join(dir_path, signal_filename)
        background_path = os.path.join(dir_path, background_filename)

        self.signal = IntensityMap(signal_path)
        self.background = IntensityMap(background_path)
        self.subtracted = None

    def load_data(self):
        """
        Load data for both the signal and background.
        """
        self.signal.load_data()
        self.background.load_data()

    def subtract_background(self):
        """
        Subtract the background data from the signal data.
        """
        if self.signal.data is None or self.background.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Ensure that the arrays are of the same shape
        if self.signal.data.shape != self.background.data.shape:
            raise ValueError("Signal and background data must have the same shape.")
        self.subtracted = self.signal.data - self.background.data
        return self.subtracted

    def plot_maps(self):
        """
        Plot the signal, background, and background-subtracted maps side by side.
        """
        if self.signal.data is None or self.background.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Calculate the subtracted map if not already done
        if self.subtracted is None:
            self.subtract_background()

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1,
                                3,
                                figsize=(15, 5))

        # Plot signal
        self.signal.plot(ax=axs[0],
                         title="Signal")

        # Plot background
        self.background.plot(ax=axs[1],
                             title="Background")

        # Plot signal minus background
        im = axs[2].imshow(self.subtracted,
                           cmap='viridis')
        axs[2].set_title("Signal - Background")
        plt.colorbar(im,
                     ax=axs[2])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Replace these file paths with your CSV file locations.

    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Strong THz Focusing/IR_Camera_Beam_Analysis'
    signal_filename = 'Image_THzBeam_after_1inch_parmirror.csv'
    background_filename = 'Image_background_after_1inch_parmirror.csv'

    analysis = Analysis(dir_path = dir_path,
                        signal_filename =signal_filename,
                        background_filename = background_filename)
    analysis.load_data()
    analysis.subtract_background()
    analysis.plot_maps()