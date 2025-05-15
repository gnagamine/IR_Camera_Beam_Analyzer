import matplotlib.pyplot as plt
# Assuming AnalysisIRCamera is defined in BeamAnalysis and works as expected.
# If BeamAnalysis also uses matplotlib, ensure backend is set appropriately there or at the very start of your main script.
from BeamAnalysis import AnalysisIRCamera
import numpy as np  # Added for np.array if fitting_coefficients are not already numpy types


class CompareCameras:
    def __init__(self,
                 dir_path_NEC,
                 signal_filename_NEC,
                 dir_path_HIKMICRO,
                 signal_filename_HIKMICRO,
                 bakcground_filename_HIKMICRO, ):  # Corrected typo: bakcground_filename_HIKMICRO -> background_filename_HIKMICRO

        self.NECAnalysis = AnalysisIRCamera(dir_path=dir_path_NEC,
                                            signal_filename=signal_filename_NEC,
                                            camera_name='NEC')

        self.HIKMICROAnalysis = AnalysisIRCamera(dir_path=dir_path_HIKMICRO,
                                                 signal_filename=signal_filename_HIKMICRO,
                                                 background_filename=bakcground_filename_HIKMICRO,
                                                 # Corrected typo
                                                 camera_name='HIKMICRO')

    def plot_maps_side_by_side(self,
                               plot_width_for_cropping_in_um=500):
        """
        Plot cameras maps side by side, cropped to a specified width
        around the center of the Gaussian fit for each.
        """
        # Create a figure with 2 subplots (changed from 3 to 2 based on usage)
        fig, axs = plt.subplots(1,
                                2,
                                figsize=(15, 7))  # Adjusted figsize slightly for better aspect if cropped

        # --- Plot NEC ---
        if self.NECAnalysis.processed_signal is not None and self.NECAnalysis.processed_signal.size > 0:
            rows_NEC, cols_NEC = self.NECAnalysis.processed_signal.data.shape
            pixel_size_NEC = self.NECAnalysis.pixel_size_um

            # Define the full extent of the image in physical units (micrometers)
            # extent = [left, right, bottom, top]
            extent_NEC_left = 0
            extent_NEC_right = pixel_size_NEC * cols_NEC
            extent_NEC_bottom = 0
            extent_NEC_top = pixel_size_NEC * rows_NEC
            extent_NEC = [extent_NEC_left, extent_NEC_right, extent_NEC_bottom, extent_NEC_top]

            # Get Gaussian fit parameters (assuming xo, yo are pixel indices from top-left)
            _, NEC_fitting_coefficients, _, _ = self.NECAnalysis.fit_gaussian()
            xo_NEC_pix = NEC_fitting_coefficients['xo']  # Center column index
            yo_NEC_pix = NEC_fitting_coefficients['yo']  # Center row index (from top-left)

            # Calculate physical center of the Gaussian in micrometers for plot axes
            # (xo_NEC_pix + 0.5) for pixel center, assuming xo_NEC_pix is 0-indexed column
            x_center_phys_NEC = extent_NEC_left + (xo_NEC_pix + 0.5) * pixel_size_NEC
            # For y-axis (origin='upper' for imshow, extent y-axis is bottom-up):
            # (yo_NEC_pix + 0.5) for pixel center from top.
            # Physical y-coordinate on plot axis (increases upwards)
            y_center_phys_NEC_on_axis = extent_NEC_top - (yo_NEC_pix + 0.5) * pixel_size_NEC

            im0 = axs[0].imshow(self.NECAnalysis.processed_signal.data,
                                cmap='viridis',
                                extent=extent_NEC,
                                origin='upper',
                                # Default, but good to be explicit
                                aspect='equal')  # Ensure pixels are square in the plot
            axs[0].set_title("NEC")
            axs[0].set_xlabel("x (μm)")  # Using μ symbol
            axs[0].set_ylabel("y (μm)")  # Using μ symbol
            plt.colorbar(im0,
                         ax=axs[0],
                         label='Intensity')

            # Set plot limits for cropping
            half_crop_width_um = plot_width_for_cropping_in_um / 2.0
            axs[0].set_xlim(x_center_phys_NEC - half_crop_width_um,
                            x_center_phys_NEC + half_crop_width_um)
            axs[0].set_ylim(y_center_phys_NEC_on_axis - half_crop_width_um,
                            y_center_phys_NEC_on_axis + half_crop_width_um)
        else:
            axs[0].set_title("NEC (No Data)")
            axs[0].text(0.5,
                        0.5,
                        "No data to display",
                        ha="center",
                        va="center",
                        transform=axs[0].transAxes)

        # --- Plot HIKMICRO ---
        if self.HIKMICROAnalysis.processed_signal is not None and self.HIKMICROAnalysis.processed_signal.size > 0:
            rows_HIKMICRO, cols_HIKMICRO = self.HIKMICROAnalysis.processed_signal.data.shape
            pixel_size_HIKMICRO = self.HIKMICROAnalysis.pixel_size_um

            # Define the full extent of the image in physical units (micrometers)
            extent_HIKMICRO_left = 0
            extent_HIKMICRO_right = pixel_size_HIKMICRO * cols_HIKMICRO
            extent_HIKMICRO_bottom = 0
            extent_HIKMICRO_top = pixel_size_HIKMICRO * rows_HIKMICRO
            extent_HIKMICRO = [extent_HIKMICRO_left, extent_HIKMICRO_right, extent_HIKMICRO_bottom, extent_HIKMICRO_top]

            # Get Gaussian fit parameters
            _, HIKMICRO_fitting_coefficients, _, _ = self.HIKMICROAnalysis.fit_gaussian()
            xo_HIKMICRO_pix = HIKMICRO_fitting_coefficients['xo']  # Center column index
            yo_HIKMICRO_pix = HIKMICRO_fitting_coefficients['yo']  # Center row index (from top-left)

            # Calculate physical center of the Gaussian in micrometers for plot axes
            x_center_phys_HIKMICRO = extent_HIKMICRO_left + (xo_HIKMICRO_pix + 0.5) * pixel_size_HIKMICRO
            y_center_phys_HIKMICRO_on_axis = extent_HIKMICRO_top - (yo_HIKMICRO_pix + 0.5) * pixel_size_HIKMICRO

            im1 = axs[1].imshow(self.HIKMICROAnalysis.processed_signal.data,
                                cmap='viridis',
                                extent=extent_HIKMICRO,
                                origin='upper',
                                aspect='equal')
            axs[1].set_title("HIKMICRO")
            axs[1].set_xlabel("x (μm)")
            axs[1].set_ylabel("y (μm)")
            plt.colorbar(im1,
                         ax=axs[1],
                         label='Intensity')

            # Set plot limits for cropping
            half_crop_width_um = plot_width_for_cropping_in_um / 2.0
            axs[1].set_xlim(x_center_phys_HIKMICRO - half_crop_width_um,
                            x_center_phys_HIKMICRO + half_crop_width_um)
            axs[1].set_ylim(y_center_phys_HIKMICRO_on_axis - half_crop_width_um,
                            y_center_phys_HIKMICRO_on_axis + half_crop_width_um)
        else:
            axs[1].set_title("HIKMICRO (No Data)")
            axs[1].text(0.5,
                        0.5,
                        "No data to display",
                        ha="center",
                        va="center",
                        transform=axs[1].transAxes)

        fig.tight_layout()  # Adjust layout to prevent overlapping titles/labels
        plt.show()  # Use plt.show() to display the figure


if __name__ == "__main__":
    # Ensure AnalysisIRCamera and its dependencies (like IntensityMap) are correctly defined and accessible.
    # The IntensityMap class should ideally handle its own matplotlib backend configuration if needed,
    # or it should be done at the very top of the main script that runs this.

    dir_path_NEC = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/NEC_camera/Power Dependence/90 degrees Humidity 1'
    signal_filename_NEC = '20250514_151329_879_001_10_01_10.csv'

    dir_path_HIKMICRO = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/HIKMICRO_camera/PowerSeries'
    signal_filename_HIKMICRO = '90 degrees Humidity 0,8.csv'
    background_filename_HIKMICRO = 'background 90 degrees Humidity 0,8.csv'  # Corrected typo

    compare_cameras = CompareCameras(dir_path_NEC,
                                     signal_filename_NEC,
                                     dir_path_HIKMICRO,
                                     signal_filename_HIKMICRO,
                                     background_filename_HIKMICRO)
    compare_cameras.plot_maps_side_by_side(plot_width_for_cropping_in_um=1000)  # Example crop width
