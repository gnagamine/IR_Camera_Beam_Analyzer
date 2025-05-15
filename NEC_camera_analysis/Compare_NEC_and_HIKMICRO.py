from BeamAnalysis import AnalysisIRCamera
import matplotlib.pyplot as plt

class CompareCameras:
    def __init__(self,
                 dir_path_NEC,
                 signal_filename_NEC,
                 dir_path_HIKMICRO,
                 signal_filename_HIKMICRO,
                 bakcground_filename_HIKMICRO,):

        self.NECAnalysis = AnalysisIRCamera(dir_path = dir_path_NEC,
                        signal_filename =signal_filename_NEC,
                        camera_name = 'NEC')

        self.HIKMICROAnalysis = AnalysisIRCamera(dir_path = dir_path_HIKMICRO,
                        signal_filename =signal_filename_HIKMICRO,
                        background_filename=bakcground_filename_HIKMICRO,
                        camera_name = 'HIKMICRO')

    def plot_maps_side_by_side(self,
                               plot_width_for_cropping_in_um = 500):
        """
        Plot cameras maps side by side
        """
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1,
                                2,
                                figsize=(15, 5))

        # Plot NEC
        rows_NEC, cols_nec = self.NECAnalysis.processed_signal.data.shape
        extent_NEC = [0, self.NECAnalysis.pixel_size_um * cols_nec, 0, self.NECAnalysis.pixel_size_um * rows_NEC]
        _, NEC_fitting_coefficients, _, _  = self.NECAnalysis.fit_gaussian()
        xo_NEC = NEC_fitting_coefficients['xo']
        yo_NEC = NEC_fitting_coefficients['yo']


        im0 = axs[0].imshow(self.NECAnalysis.processed_signal.data,
                            cmap='viridis',
                            extent=extent_NEC)
        axs[0].set_title("NEC")
        axs[0].set_xlabel("x (um)")
        axs[0].set_ylabel("y (um)")
        plt.colorbar(im0,
                     ax=axs[0])

        # Plot HIKMICRO

        rows_HIKMICRO, cols_HIKMICRO = self.HIKMICROAnalysis.processed_signal.data.shape
        extent_HIKMICRO = [0, self.HIKMICROAnalysis.pixel_size_um * cols_HIKMICRO, 0, self.HIKMICROAnalysis.pixel_size_um * rows_HIKMICRO]
        _, HIKMICRO_fitting_coefficients, _, _  = self.HIKMICROAnalysis.fit_gaussian()
        xo_HIKMICRO = HIKMICRO_fitting_coefficients['xo']
        yo_HIKMICRO = HIKMICRO_fitting_coefficients['yo']

        im1 = axs[1].imshow(self.HIKMICROAnalysis.processed_signal.data,
                            cmap='viridis',
                            extent=extent_HIKMICRO)
        axs[1].set_title("HIKMICRO")
        axs[1].set_xlabel("x (um)")
        axs[1].set_ylabel("y (um)")
        plt.colorbar(im1,
                     ax=axs[1])

        fig.tight_layout()
        fig.show()



if __name__ == "__main__":

    dir_path_NEC = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/NEC_camera/Power Dependence/90 degrees Humidity 1'
    signal_filename_NEC = '20250514_151329_879_001_10_01_10.csv'


    dir_path_HIKMICRO = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250514/HIKMICRO_camera/PowerSeries'
    signal_filename_HIKMICRO = '90 degrees Humidity 0,8.csv'
    bakcground_filename_HIKMICRO = 'background 90 degrees Humidity 0,8.csv'


    compare_cameras = CompareCameras(dir_path_NEC, signal_filename_NEC, dir_path_HIKMICRO, signal_filename_HIKMICRO, bakcground_filename_HIKMICRO)
    compare_cameras.plot_maps_side_by_side()