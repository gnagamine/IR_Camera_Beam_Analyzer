
import matplotlib.pyplot as plt
import numpy as np


class GlazMatrixDataPlotter:
    def __init__(self,
                 time_data,
                 time_trace_matrix_df):

        self.time_data = time_data
        self.time_trace_matrix_data_df = time_trace_matrix_df
        self.delay_start = float(self.time_data.min())
        self.delay_end = float(self.time_data.max())

    def plot_spectral_timetrace(self,
                                vmin_percentile=None,
                                vmax_percentile=None,
                                bool_plot_normalized_time_trace=False,
                                name_save_tag=None):
        """Create a 2D plot of spectral difference vs delay."""

        if bool_plot_normalized_time_trace:
            data = self.time_trace_matrix_data_df / np.max(self.time_trace_matrix_data_df)
        else:
            data = self.time_trace_matrix_data_df

        # Determine color scale limits based on normalized or raw data
        if bool_plot_normalized_time_trace and vmin_percentile is not None and vmax_percentile is not None:
            # For normalized data, interpret percentiles as fractional cutoffs
            vmin = vmin_percentile / 100.0
            vmax = vmax_percentile / 100.0
        elif vmin_percentile is not None and vmax_percentile is not None:
            # For raw data, use distribution percentiles
            vmin = np.percentile(data.values,
                                 vmin_percentile)
            vmax = np.percentile(data.values,
                                 vmax_percentile)
        else:
            # Default full data range
            vmin = np.min(data.values)
            vmax = np.max(data.values)

        # Build figure and axes, return them for further customization
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data.values,
                       aspect='auto',
                       origin='lower',
                       extent=[0, data.shape[1], self.delay_start, self.delay_end],
                       cmap='viridis',
                       vmin=vmin,
                       vmax=vmax)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Delay (ps)')
        fig.colorbar(im,
                     ax=ax,
                     label='counts')

        return fig, ax

