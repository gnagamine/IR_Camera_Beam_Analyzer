from Code_utilities.BeamAnalysis import BeamAnalysis


if __name__ == "__main__":

    dir_path ='/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250516/position series_NEC/each_delay'
    signal_filename = '1,35 mm.csv'

    analysis = BeamAnalysis(dir_path = dir_path,
                            signal_filename =signal_filename,
                            camera_name='NEC')

    popt, fitting_coefficients, fig, ax = analysis.fit_gaussian(bool_save_plots=True)
    fig.show()