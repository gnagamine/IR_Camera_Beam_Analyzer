import os

import re
import pandas as pd

from Code_utilities.FromGlazPPAnalysis.GlazPumpProbeDataClass_v2 import GlazPumpProbeData_v2


class GlazPPSeriesAnalysis:
    def __init__(self,
                 parent_dir_path=None):
        """Initialize with parent directory path."""
        self.parent_dir_path = parent_dir_path
        self.all_files_path_list = self.get_all_files_path_list()
        self.spectral_difference_paths_list = self.get_specific_paths('SpectralDifference')
        self.pumped_spectra_paths_list = self.get_specific_paths('wPump')
        self.unpumped_spectra_paths_list = self.get_specific_paths('woPump')

    def get_all_files_path_list(self):
        parent_dir_path = self.parent_dir_path
        all_files_path_list = [
            f for f in os.listdir(parent_dir_path)
            if f.endswith(".txt") and not f.startswith(".")
        ]

        return all_files_path_list

    def get_specific_paths(self,
                           keyword=None):
        """
        Return a list of full paths for all .txt files in parent_dir_path
        containing 'SpectralDifference' in their filename.
        """
        specific_paths_list = [
            os.path.join(self.parent_dir_path,
                         f)
            for f in os.listdir(self.parent_dir_path)
            if f.endswith(".txt")
               and not f.startswith(".")
               and keyword in f
        ]
        return specific_paths_list

    def plot_all_spectrum_timetraces(self,
                                     keyword=None,
                                     bool_plot_normalized_time_trace=False):

        specific_paths_list = self.get_specific_paths(keyword)
        for path in specific_paths_list:
            data = GlazPumpProbeData_v2(path)
            fig, ax = data.plot_spectral_timetrace(
                bool_plot_normalized_time_trace=bool_plot_normalized_time_trace,
                name_save_tag=str(data.series_parameter))
            print(data.series_parameter)

        return fig, ax

    def plot_all_ffts_together(self,
                               pixel=None,
                               window_start=None,
                               window_end=None,
                               bool_organize_legend=False,
                               bool_save_fig=False,
                               plot_x_lim_min=None,
                               plot_x_lim_max=None,
                               bool_plot_normalized=False,
                               bool_return_data=False, ):

        fft_data_dict={}
        fft_fig, fft_ax = None, None
        for path in self.spectral_difference_paths_list:
            data = GlazPumpProbeData_v2(path)
            if bool_return_data:
                fft_fig, fft_ax, fft_data = data.plot_fft(pixel=pixel,
                                                window_start=window_start,
                                                window_end=window_end,
                                                plot_legend=str(data.series_parameter),
                                                fig=fft_fig,
                                                ax=fft_ax,
                                                bool_plot_normalized=bool_plot_normalized,
                                                bool_return_data=bool_return_data, )
                fft_data_dict[str(data.series_parameter)]=fft_data.fft_df
            else:
                fft_fig, fft_ax = data.plot_fft(pixel=pixel,
                                                window_start=window_start,
                                                window_end=window_end,
                                                plot_legend=str(data.series_parameter),
                                                fig=fft_fig,
                                                ax=fft_ax,
                                                bool_plot_normalized=bool_plot_normalized, )

        if plot_x_lim_min is not None:
            fft_ax.set_xlim(plot_x_lim_min,
                            plot_x_lim_max)

        if bool_organize_legend:
            resort_legend_by_angle(fft_ax,
                                   reverse=False)  # ascending

        if bool_save_fig:
            fft_fig.savefig('FFT.png')
        if bool_return_data:
            return fft_fig, fft_ax, fft_data_dict
        if not bool_return_data:
            return fft_fig, fft_ax

    def plot_single_fft(self,
                        pixel=None,
                        window_start=None,
                        window_end=None,
                        bool_organize_legend=False,
                        bool_save_fig=False,
                        plot_x_lim_min=None,
                        plot_x_lim_max=None,
                        bool_plot_normalized=False,
                        filename_keyword=None):
        """
        Plots FFTs for specific files based on a keyword in their filename.

        If filename_keyword is provided, only files whose paths contain this string
        will have their FFTs plotted on the same axes. If no keyword is provided,
        all spectral difference FFTs are plotted.

        Parameters
        ----------
        pixel : int, optional
            The pixel to extract data for.
        window_start : float, optional
            Start time for the FFT window.
        window_end : float, optional
            End time for the FFT window.
        bool_organize_legend : bool, optional
            If True, organizes the legend by angle. Defaults to False.
        bool_save_fig : bool, optional
            If True, saves the figure as 'FFT.png'. Defaults to False.
        plot_x_lim_min : float, optional
            Minimum x-axis limit for the plot.
        plot_x_lim_max : float, optional
            Maximum x-axis limit for the plot.
        bool_plot_normalized : bool, optional
            If True, plots normalized FFT data. Defaults to False.
        filename_keyword : str, optional
            If provided, only files with this string in their full path will be processed.

        Returns
        -------
        tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure and axes objects containing the FFT plots.
            Returns (None, None) if no FFTs were plotted (e.g., no matching files).
        """
        fft_fig, fft_ax = None, None
        plot_made = False  # Flag to track if any plot was successfully generated

        for path in self.spectral_difference_paths_list:
            # Filter based on filename_keyword
            if filename_keyword is not None and filename_keyword not in path:
                continue  # Skip this path if it doesn't contain the keyword

            data = GlazPumpProbeData_v2(path)

            # The first time data.plot_fft is called, it creates the fig and ax.
            # Subsequent calls plot on the same fig/ax.
            fft_fig, fft_ax = data.plot_fft(pixel=pixel,
                                            window_start=window_start,
                                            window_end=window_end,
                                            plot_legend=str(data.series_parameter),
                                            fig=fft_fig,
                                            ax=fft_ax,
                                            bool_plot_normalized=bool_plot_normalized)
            plot_made = True  # Mark that at least one plot was attempted

        # Apply post-processing only if a figure/axes object was actually created
        if plot_made and fft_ax is not None:
            if plot_x_lim_min is not None:
                fft_ax.set_xlim(plot_x_lim_min,
                                plot_x_lim_max)

            if bool_organize_legend:
                # Ensure resort_legend_by_angle is accessible (e.g., imported or in same file)
                resort_legend_by_angle(fft_ax,
                                       reverse=False)  # ascending

            if bool_save_fig:
                # Consider making the filename dynamic, e.g., based on filename_keyword or pixel
                save_filename = f'FFT_{filename_keyword or "all"}_pixel{pixel or "all"}.png'
                fft_fig.savefig(save_filename)
        elif not plot_made:
            print(f"Warning: No FFTs were plotted. No files found matching keyword '{filename_keyword}' "
                  f"or no files in spectral_difference_paths_list.")

        return fft_fig, fft_ax

    def plot_single_fft_taking_peak_difference(self,
                                               pixel_1=None,
                                               pixel_2=None,
                                               number_of_pixels_for_sum=1,
                                               window_start=None,
                                               window_end=None,
                                               bool_organize_legend=False,
                                               bool_save_fig=False,
                                               plot_x_lim_min=None,
                                               plot_x_lim_max=None,
                                               bool_plot_normalized=False,
                                               filename_keyword=None):
        """
        Plots FFTs of the difference between signals at two pixel regions.

        This method iterates through spectral difference files, calculates the FFT
        of the signal difference between two pixel regions, and plots them on the
        same axes. An optional keyword can filter which files are processed.

        Parameters
        ----------
        pixel_1 : int, optional
            The center pixel for the first region.
        pixel_2 : int, optional
            The center pixel for the second region.
        number_of_pixels_for_sum : int, optional
            The number of pixels to average around the center pixels. Defaults to 1.
        window_start : float, optional
            Start time for the FFT window.
        window_end : float, optional
            End time for the FFT window.
        bool_organize_legend : bool, optional
            If True, organizes the legend by angle. Defaults to False.
        bool_save_fig : bool, optional
            If True, saves the figure. Defaults to False.
        plot_x_lim_min : float, optional
            Minimum x-axis limit for the plot.
        plot_x_lim_max : float, optional
            Maximum x-axis limit for the plot.
        bool_plot_normalized : bool, optional
            If True, plots normalized FFT data. Defaults to False.
        filename_keyword : str, optional
            If provided, only files with this string in their path will be processed.

        Returns
        -------
        tuple
            A tuple containing the matplotlib Figure and Axes objects,
            (fig, ax). Returns (None, None) if no plots were made.
        """
        fft_fig, fft_ax = None, None
        plot_made = False  # Flag to track if any plot was successfully generated

        for path in self.spectral_difference_paths_list:
            # Filter based on filename_keyword
            if filename_keyword is not None and filename_keyword not in path:
                continue  # Skip this path if it doesn't contain the keyword

            data = GlazPumpProbeData_v2(path)

            fft_fig, fft_ax = data.plot_fft_subtracting_peaks(
                pixel_1=pixel_1,
                pixel_2=pixel_2,
                number_of_pixels_for_sum=number_of_pixels_for_sum,
                window_start=window_start,
                window_end=window_end,
                plot_legend=str(data.series_parameter),
                fig=fft_fig,
                ax=fft_ax,
                bool_plot_normalized=bool_plot_normalized)
            plot_made = True  # Mark that at least one plot was generated

        # Apply post-processing only if a figure/axes object was actually created
        if plot_made and fft_ax is not None:
            if plot_x_lim_min is not None:
                fft_ax.set_xlim(plot_x_lim_min,
                                plot_x_lim_max)

            if bool_organize_legend:
                resort_legend_by_angle(fft_ax,
                                       reverse=False)  # ascending

            if bool_save_fig:
                # Dynamic filename to avoid overwriting
                keyword_tag = f"_{filename_keyword}" if filename_keyword else ""
                save_filename = f'FFT_peak_difference{keyword_tag}.png'
                fft_fig.savefig(save_filename)
        elif not plot_made:
            print(f"Warning: No FFTs were plotted. No files found matching keyword '{filename_keyword}'.")

        return fft_fig, fft_ax

    def plot_all_ffts_together_taking_peak_difference(self,
                                                      pixel_1=None,
                                                      pixel_2=None,
                                                      number_of_pixels_for_sum=1,
                                                      window_start=None,
                                                      window_end=None,
                                                      bool_organize_legend=False,
                                                      bool_save_fig=False,
                                                      plot_x_lim_min=None,
                                                      plot_x_lim_max=None,
                                                      bool_plot_normalized=False):

        fft_fig, fft_ax = None, None
        for path in self.spectral_difference_paths_list:
            data = GlazPumpProbeData_v2(path)

            fft_fig, fft_ax = data.plot_fft_subtracting_peaks(pixel_1=pixel_1,
                                                              pixel_2=pixel_2,
                                                              number_of_pixels_for_sum=number_of_pixels_for_sum,
                                                              window_start=window_start,
                                                              window_end=window_end,
                                                              plot_legend=str(data.series_parameter),
                                                              fig=fft_fig,
                                                              ax=fft_ax,
                                                              bool_plot_normalized=bool_plot_normalized)
        if plot_x_lim_min is not None:
            fft_ax.set_xlim(plot_x_lim_min,
                            plot_x_lim_max)

        if bool_organize_legend:
            resort_legend_by_angle(fft_ax,
                                   reverse=False)  # ascending

        if bool_save_fig:
            fft_fig.savefig('FFT.png')

        return fft_fig, fft_ax

    def plot_all_ffts_together_subtracting_peaks(self,
                                                 pixel=None,
                                                 window_start=None,
                                                 window_end=None,
                                                 bool_organize_legend=False,
                                                 bool_save_fig=False,
                                                 plot_x_lim_min=None,
                                                 plot_x_lim_max=None,
                                                 bool_plot_normalized=False):

        fft_fig, fft_ax = None, None
        for path in self.spectral_difference_paths_list:
            data = GlazPumpProbeData_v2(path)

            fft_fig, fft_ax = data.plot_fft(pixel=pixel,
                                            window_start=window_start,
                                            window_end=window_end,
                                            plot_legend=str(data.series_parameter),
                                            fig=fft_fig,
                                            ax=fft_ax,
                                            bool_plot_normalized=bool_plot_normalized)
        if plot_x_lim_min is not None:
            fft_ax.set_xlim(plot_x_lim_min,
                            plot_x_lim_max)

        if bool_organize_legend:
            resort_legend_by_angle(fft_ax,
                                   reverse=False)  # ascending

        if bool_save_fig:
            fft_fig.savefig('FFT.png')

        return fft_fig, fft_ax

    def plot_all_spectrum_timetrace_separately(self,
                                               spectrum_type_keyword=None,
                                               bool_plot_normalized_time_trace=False):

        specific_paths_list = self.get_specific_paths(spectrum_type_keyword)
        for path in specific_paths_list:
            data = GlazPumpProbeData_v2(path)
            fig, ax = data.plot_spectral_timetrace(
                bool_plot_normalized_time_trace=bool_plot_normalized_time_trace,
                name_save_tag=str(data.series_parameter))
            print(data.series_parameter)
            fig.show()

    def plot_all_timetraces_at_pixel_separately(self,
                                                spectrum_type_keyword=None,
                                                pixel=None,
                                                number_of_pixels_for_average=1,
                                                save_fig_bool=False):

        specific_paths_list = self.get_specific_paths(spectrum_type_keyword)
        for path in specific_paths_list:
            data = GlazPumpProbeData_v2(path)
            fig, ax = data.plot_timetrace_for_given_pixel(pixel=pixel,
                                                          number_of_pixels_for_average=number_of_pixels_for_average
                                                          )
            print(data.series_parameter)
            ax.set_title(str(data.series_parameter))
            if save_fig_bool:
                fig.savefig(str(data.series_parameter) + '.pdf')
            fig.show()

    def plot_all_timetraces_at_pixel_separately_returning_data(self,
                                                               spectrum_type_keyword=None,
                                                               pixel=None,
                                                               number_of_pixels_for_average=1,
                                                               save_fig_bool=False):

        specific_paths_list = self.get_specific_paths(spectrum_type_keyword)

        # List to hold all Series that will form the DataFrame columns
        all_series_for_df = []

        for path in specific_paths_list:
            data = GlazPumpProbeData_v2(path)
            # The method plot_timetrace_for_given_pixel_returning_data also creates a figure and axes
            fig, ax, delays, pp_data_signal = data.plot_timetrace_for_given_pixel_returning_data(
                pixel=pixel,
                number_of_pixels_for_average=number_of_pixels_for_average
            )

            # Keep plotting functionality as implied by method name
            print(data.series_parameter)
            ax.set_title(str(data.series_parameter))
            if save_fig_bool:
                fig.savefig(str(data.series_parameter) + '.pdf')
            fig.show()

            # Ensure delays and pp_data_signal from the *current* file have compatible lengths
            if len(delays) == len(pp_data_signal):
                delay_col_name = f"delay_{data.series_parameter}"
                # Using str() for series_parameter as it's used as a column name directly
                pp_data_col_name = str(data.series_parameter)

                current_delays_series = pd.Series(delays,
                                                  name=delay_col_name)
                current_pp_data_series = pd.Series(pp_data_signal,
                                                   name=pp_data_col_name)

                all_series_for_df.append(current_delays_series)
                all_series_for_df.append(current_pp_data_series)
            else:
                print(f"Warning: Mismatch in length between delays ({len(delays)}) and pp_data ("
                      f"{len(pp_data_signal)}) for {data.series_parameter}. Skipping this series.")

        # Construct the DataFrame by concatenating all collected Series column-wise
        if all_series_for_df:
            # pd.concat will align by index; series of different lengths will result in NaNs
            pp_data_df = pd.concat(all_series_for_df,
                                   axis=1)
        else:
            pp_data_df = pd.DataFrame()  # Return an empty DataFrame if no data was processed
            print("Warning: No data was processed or all series had mismatches. Cannot create DataFrame.")

        return pp_data_df

    def plot_all_timetraces_subtracting_peaks_separately_returning_data(self,
                                                                        spectrum_type_keyword=None,
                                                                        pixel_1=None,
                                                                        pixel_2=None,
                                                                        number_of_pixels_for_sum=1,
                                                                        save_fig_bool=False):

        specific_paths_list = self.get_specific_paths(spectrum_type_keyword)

        # List to hold all Series that will form the DataFrame columns
        all_series_for_df = []

        for path in specific_paths_list:
            data = GlazPumpProbeData_v2(path)
            # The method plot_timetrace_for_given_pixel_returning_data also creates a figure and axes
            fig, ax, delays, pp_data_signal = data.plot_timetrace_subtracting_peaks_returning_data(
                pixel_1=pixel_1,
                pixel_2=pixel_2,
                number_of_pixels_for_sum=number_of_pixels_for_sum
            )

            # Keep plotting functionality as implied by method name
            print(data.series_parameter)
            ax.set_title(str(data.series_parameter))
            if save_fig_bool:
                fig.savefig(str(data.series_parameter) + '.pdf')

            # Ensure delays and pp_data_signal from the *current* file have compatible lengths
            if len(delays) == len(pp_data_signal):
                delay_col_name = f"delay_{data.series_parameter}"
                # Using str() for series_parameter as it's used as a column name directly
                pp_data_col_name = str(data.series_parameter)

                current_delays_series = pd.Series(delays,
                                                  name=delay_col_name)
                current_pp_data_series = pd.Series(pp_data_signal,
                                                   name=pp_data_col_name)

                all_series_for_df.append(current_delays_series)
                all_series_for_df.append(current_pp_data_series)
            else:
                print(f"Warning: Mismatch in length between delays ({len(delays)}) and pp_"
                      f"data ({len(pp_data_signal)}) for {data.series_parameter}. Skipping this series.")

        # Construct the DataFrame by concatenating all collected Series column-wise
        if all_series_for_df:
            # pd.concat will align by index; series of different lengths will result in NaNs
            pp_data_df = pd.concat(all_series_for_df,
                                   axis=1)
        else:
            pp_data_df = pd.DataFrame()  # Return an empty DataFrame if no data was processed
            print("Warning: No data was processed or all series had mismatches. Cannot create DataFrame.")

        return pp_data_df


def resort_legend_by_angle(ax,
                           *,
                           reverse=False):
    """
    Rebuild the legend so that entries are ordered by the numeric value
    before the word 'degrees'.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    reverse : bool, optional
        False  → ascending   (-30, -20, -10, 0, +10, +20 …)
        True   → descending
    """
    # current handles & labels
    handles, labels = ax.get_legend_handles_labels()

    # pull the signed integer preceding 'degrees'
    def angle_value(label):
        m = re.match(r'\s*([+-]?\d+)\s*degrees',
                     label,
                     flags=re.I)
        return int(m.group(1)) if m else 0  # 0 fallback for odd labels

    # sort pairs by that numeric value
    sorted_pairs = sorted(zip(handles,
                              labels),
                          key=lambda hl: angle_value(hl[1]),
                          reverse=reverse)

    # unzip and reset legend
    sorted_handles, sorted_labels = zip(*sorted_pairs)
    ax.legend(sorted_handles,
              sorted_labels, )
