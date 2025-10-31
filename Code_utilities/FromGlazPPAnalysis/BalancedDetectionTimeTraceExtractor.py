import pandas as pd


class BalancedDetectionTimeTraceExtractor:
    def __init__(self,
                 delays,
                 time_trace_matrix_df,
                 pixel_1,
                 pixel_2,
                 number_of_pixels_for_sum: int = 1):

        self.delays = delays
        self.time_trace_matrix_df = time_trace_matrix_df
        self.pixel_1 = pixel_1
        self.pixel_2 = pixel_2
        self.number_of_pixels_for_sum = number_of_pixels_for_sum

        self.timetrace_subtracting_peaks_df=self._get_timetrace_subtracting_peaks_df()


    def _get_timetrace_subtracting_peaks_df(self,):
        pixel_1 = self.pixel_1
        pixel_2 = self.pixel_2
        number_of_pixels_for_sum = self.number_of_pixels_for_sum

        if not isinstance(pixel_1,
                          int):
            raise TypeError("pixel_1 must be an integer")
        if not isinstance(number_of_pixels_for_sum,
                          int) or number_of_pixels_for_sum < 1:
            raise ValueError("number_of_pixels_for_average must be a positive integer")

        n_cols = self.time_trace_matrix_df.shape[1]
        if pixel_1 < 0 or pixel_1 >= n_cols:
            raise ValueError(f"pixel_1 {pixel_1} out of range (0–{n_cols - 1})")

        # --- determine column window 1 ---------------------------------------
        half_window_1 = number_of_pixels_for_sum // 2
        start_col_1 = max(0,
                          pixel_1 - half_window_1)
        end_col_1 = min(n_cols,
                        start_col_1 + number_of_pixels_for_sum)

        # If we're at the right edge adjust start_col to keep window size
        start_col_1 = max(0,
                          end_col_1 - number_of_pixels_for_sum)

        sum_timetrace_pixel_1 = self.time_trace_matrix_df.iloc[:, start_col_1:end_col_1].sum(axis=1)

        # --- determine column window 2 ---------------------------------------
        half_window_2 = number_of_pixels_for_sum // 2
        start_col_2 = max(0,
                          pixel_2 - half_window_2)
        end_col_2 = min(n_cols,
                        start_col_2 + number_of_pixels_for_sum)

        # If we're at the right edge adjust start_col to keep window size
        start_col_2 = max(0,
                          end_col_2 - number_of_pixels_for_sum)

        sum_timetrace_pixel_2 = self.time_trace_matrix_df.iloc[:, start_col_2:end_col_2].sum(axis=1)

        # --- compute time‑trace ------------------------------------

        time_trace = sum_timetrace_pixel_1 - sum_timetrace_pixel_2

        timetrace_subtracting_peaks_df = pd.DataFrame({'time (ps)': self.delays,
                                                       'pp_data': time_trace})

        return timetrace_subtracting_peaks_df