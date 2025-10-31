import matplotlib.pyplot as plt
from Code_utilities.InputParametersClass import InputParametersClass
from Code_utilities.StandartFigureClass import StandardFigure
import numpy as np


class DataPlotter:
    def __init__(self, ):
        self.InputParameters = InputParametersClass()

    def _get_standart_figure_and_ax(self):
        figure = StandardFigure()
        return figure.fig, figure.ax

    def plot_relative_responsivity_vs_cutoff(self,
                                             x,
                                             y,
                                             ax=None,
                                             fig=None,
                                             label=None,):
        if fig is None:
            fig, ax = self._get_standart_figure_and_ax()

        sorted_pairs = sorted(zip(x,
                                  y))
        x, y = zip(*sorted_pairs)

        if fig is None:
            fig, ax = plt.subplots()
        ax.plot(x,
                y,
                marker='o',
                markerfacecolor='orange',
                markersize=10,
                alpha=0.5,
                label=label)

        ax.set_xlabel('Cutoff Frequency (THz)')
        ax.set_ylabel('Relative Responsivity (%)')
        ax.legend()

        fig.set_size_inches(4,
                            4)
        fig.tight_layout()

        return fig, ax


    def plot_asd_figure(self,
                        freqs,
                        mean_asd,
                        bool_print_asd_at_1hz=False,):
        fig, ax = self._get_standart_figure_and_ax()
        fig.set_size_inches(3.5,3)

        ax.loglog(freqs,
                mean_asd,
                color=self.InputParameters.color_nice_blue, )
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(f'ASD (°C / $\\sqrt{{\\mathrm{{Hz}}}}$)')
        ax.axvline(1.0,
                   color=self.InputParameters.color_nice_orange,
                   linestyle='--',
                   alpha=1)
        # Find the index of the frequency bin closest to 1.0 Hz
        idx_1hz = np.argmin(np.abs(freqs - 1.0))

        # Get the frequency and ASD value at that index
        freq_at_1hz = freqs[idx_1hz]
        asd_at_1hz = mean_asd[idx_1hz]

        if bool_print_asd_at_1hz:
            print("\n--- ASD Result ---")
            print(f"Frequency bin closest to 1 Hz: {freq_at_1hz:.4f} Hz")
            print(f"Averaged ASD at this frequency: {asd_at_1hz:.6f} °C / sqrt(Hz)")
            print("------------------\n")

        return fig, ax, asd_at_1hz
