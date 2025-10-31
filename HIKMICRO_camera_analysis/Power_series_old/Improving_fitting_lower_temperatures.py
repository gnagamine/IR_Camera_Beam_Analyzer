from Code_utilities.BeamAnalysis import BeamAnalysis


if __name__ == "__main__":

    dir_path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Power_series'
    background_filename = 'IR_00051_background.csv'
    signal_filename = 'IR_00049_10degrees.csv'

    analysis = BeamAnalysis(dir_path = dir_path,
                            signal_filename =signal_filename,
                            background_filename = background_filename)

    popt, fitting_coefficients, fig, ax = analysis.fit_gaussian(bool_save_plots=True)
    fig.show()