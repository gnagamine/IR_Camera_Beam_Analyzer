import matplotlib.pyplot as plt
# Assuming AnalysisIRCamera is defined in BeamAnalysis and works as expected.
# If BeamAnalysis also uses matplotlib, ensure backend is set appropriately there or at the very start of your main script.
from BeamAnalysis import BeamAnalysis
import numpy as np  # Added for np.array if fitting_coefficients are not already numpy types


class CompareCameras:
    def __init__(self,
                 dir_path_NEC,
                 signal_filename_NEC,
                 dir_path_HIKMICRO,
                 signal_filename_HIKMICRO,
                 bakcground_filename_HIKMICRO, ):  # Corrected typo: bakcground_filename_HIKMICRO -> background_filename_HIKMICRO

        self.NECAnalysis = BeamAnalysis(dir_path=dir_path_NEC,
                                        signal_filename=signal_filename_NEC,
                                        camera_name='NEC')

        self.HIKMICROAnalysis = BeamAnalysis(dir_path=dir_path_HIKMICRO,
                                             signal_filename=signal_filename_HIKMICRO,
                                             background_filename=bakcground_filename_HIKMICRO,
                                             # Corrected typo
                                             camera_name='HIKMICRO')

    def plot_maps_side_by_side(self,
                               plot_width_for_cropping_in_um=500): #TODO center the plots in zero
        """
            Plot cameras maps side by side with cropping around specified centers.

            Parameters:
            - plot_width_for_cropping_in_um (float): Width of the cropped region in micrometers (default: 500).
            """
        # Create a figure with 2 subplots
        fig, axs = plt.subplots(1,
                                2,
                                figsize=(15, 5))

        # Plot NEC
        rows_NEC, cols_nec = self.NECAnalysis.processed_signal.shape
        extent_NEC = [0, self.NECAnalysis.pixel_size_um * cols_nec, 0, self.NECAnalysis.pixel_size_um * rows_NEC]
        _, NEC_fitting_coefficients, _, _ = self.NECAnalysis.fit_gaussian()
        xo_NEC = NEC_fitting_coefficients['xo']
        yo_NEC = NEC_fitting_coefficients['yo']

        im0 = axs[0].imshow(self.NECAnalysis.processed_signal,
                            cmap='viridis',
                            extent=extent_NEC)
        axs[0].set_title("NEC")
        axs[0].set_xlabel("x (um)")
        axs[0].set_ylabel("y (um)")
        plt.colorbar(im0,
                     ax=axs[0])

        # Crop NEC plot
        pixel_size_NEC = self.NECAnalysis.pixel_size_um
        x_center_NEC = xo_NEC * pixel_size_NEC
        y_center_NEC = pixel_size_NEC * rows_NEC - yo_NEC * pixel_size_NEC
        width = plot_width_for_cropping_in_um
        left_NEC = max(0,
                       x_center_NEC - width / 2)
        right_NEC = min(pixel_size_NEC * cols_nec,
                        x_center_NEC + width / 2)
        bottom_NEC = max(0,
                         y_center_NEC - width / 2)
        top_NEC = min(pixel_size_NEC * rows_NEC,
                      y_center_NEC + width / 2)
        axs[0].set_xlim(left_NEC,
                        right_NEC)
        axs[0].set_ylim(bottom_NEC,
                        top_NEC)

        # Plot HIKMICRO
        rows_HIKMICRO, cols_HIKMICRO = self.HIKMICROAnalysis.processed_signal.shape
        _, HIKMICRO_fitting_coefficients, _, _ = self.HIKMICROAnalysis.fit_gaussian()
        xo_HIKMICRO = HIKMICRO_fitting_coefficients['xo']
        yo_HIKMICRO = HIKMICRO_fitting_coefficients['yo']
        extent_HIKMICRO = [0, self.HIKMICROAnalysis.pixel_size_um * cols_HIKMICRO, 0,
                           self.HIKMICROAnalysis.pixel_size_um * rows_HIKMICRO]

        im1 = axs[1].imshow(self.HIKMICROAnalysis.processed_signal,
                            cmap='viridis',
                            extent=extent_HIKMICRO)
        axs[1].set_title("HIKMICRO")
        axs[1].set_xlabel("x (um)")
        axs[1].set_ylabel("y (um)")
        plt.colorbar(im1,
                     ax=axs[1])

        # Crop HIKMICRO plot
        pixel_size_HIKMICRO = self.HIKMICROAnalysis.pixel_size_um
        x_center_HIKMICRO = xo_HIKMICRO * pixel_size_HIKMICRO
        y_center_HIKMICRO = pixel_size_HIKMICRO * rows_HIKMICRO - yo_HIKMICRO * pixel_size_HIKMICRO
        left_HIKMICRO = max(0,
                            x_center_HIKMICRO - width / 2)
        right_HIKMICRO = min(pixel_size_HIKMICRO * cols_HIKMICRO,
                             x_center_HIKMICRO + width / 2)
        bottom_HIKMICRO = max(0,
                              y_center_HIKMICRO - width / 2)
        top_HIKMICRO = min(pixel_size_HIKMICRO * rows_HIKMICRO,
                           y_center_HIKMICRO + width / 2)
        axs[1].set_xlim(left_HIKMICRO,
                        right_HIKMICRO)
        axs[1].set_ylim(bottom_HIKMICRO,
                        top_HIKMICRO)

        fig.tight_layout()
        plt.show()  # Changed fig.show() to plt.show() for consistency
    def print_beam_widths(self):
        _, NEC_fitting_coefficients, _, _ = self.NECAnalysis.fit_gaussian()
        print("NEC FWHM X (um):", round(NEC_fitting_coefficients['fwhm_x'], 0))
        print("NEC FWHM Y (um):", round(NEC_fitting_coefficients['fwhm_y'], 0))

        _, HIKMICRO_fitting_coefficients, _, _ = self.HIKMICROAnalysis.fit_gaussian()
        print("HIKMICRO FWHM X (um):", round(HIKMICRO_fitting_coefficients['fwhm_x'], 0))
        print("HIKMICRO FWHM Y (um):", round(HIKMICRO_fitting_coefficients['fwhm_y'], 0))

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
    compare_cameras.print_beam_widths()
