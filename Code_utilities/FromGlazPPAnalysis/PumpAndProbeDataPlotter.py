import matplotlib.pyplot as plt
from Code_utilities.StandartFigureClass import StandardFigure


class PumpAndProbeDataPlotter:

    def plot_balanced_timetrace(self,
                                fig=None,
                                ax=None,
                                y=None,
                                x=None,
                                label=None, ):
        if fig is None:
            Figure = StandardFigure()
            fig, ax = Figure.fig, Figure.ax

        ax.plot(x,
                y,
                label=label)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Polarization Rotation (Amplitude)')
        fig.tight_layout()
        ax.legend()

        return fig, ax

    def plot_FFTs(self,
                                fig=None,
                                ax=None,
                                y=None,
                                x=None,
                                label=None, ):
        if fig is None:
            Figure = StandardFigure()
            fig, ax = Figure.fig, Figure.ax

        ax.plot(x,
                y,
                label=label)
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('FFT (a.u.)')
        fig.tight_layout()
        ax.legend()

        return fig, ax

