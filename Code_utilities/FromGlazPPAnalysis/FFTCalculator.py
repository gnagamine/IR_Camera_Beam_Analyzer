import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq
from math import ceil, log2

class FFTCalculator:
    def __init__(self,
                 time_data,
                 y_data,
                 FFT_time_lim=None,
                 zero_padding_order=1,
                 bool_take_derivative=False,
                 bool_hanning=False,):

        self.time_data = time_data
        self.y_data = y_data
        self.FFT_time_lim = FFT_time_lim

        self.x_for_fft, self.y_for_fft = self._filter_values_in_selection_interval()

        self.zero_padding_order = zero_padding_order
        self.bool_take_derivative = bool_take_derivative
        self.bool_hanning = bool_hanning

        self.fft_df=self._compute_fft()

    def _compute_fft(self):
        if self.zero_padding_order>0:
            zero_padding_enabled: bool = True
        else:
            zero_padding_enabled: bool = False

        padding_order_increase: int = self.zero_padding_order
        use_hanning_window: bool = self.bool_hanning

        time_delay_ps=self.x_for_fft
        pump_probe_data=self.y_for_fft

        # Ensure there's enough data to process
        if len(time_delay_ps) < 2:
            return pd.DataFrame()

        # --- Pre-processing ---
        sampling_interval_ps = float(np.mean(np.diff(time_delay_ps)))  # dt in ps
        mean_removed_signal = self._subtract_average(pump_probe_data)

        # Choose derivative or not
        if self.bool_take_derivative:
            # True derivative with same length as input:
            # units become (original units) / ps
            signal_for_fft = np.gradient(mean_removed_signal,
                                         sampling_interval_ps)
        else:
            signal_for_fft = mean_removed_signal

        number_of_samples = len(signal_for_fft)

        # --- Windowing ---
        if use_hanning_window:
            hanning_window = np.hanning(number_of_samples)
            signal_for_fft = signal_for_fft * hanning_window
            window_correction_factor = np.sqrt((hanning_window ** 2).sum())
        else:
            window_correction_factor = np.sqrt(number_of_samples)

        # --- Zero padding decision ---
        if zero_padding_enabled:
            padded_data, n_fft = self._perform_zero_padding_corrected(
                input_data=signal_for_fft,
                original_length_for_padding=number_of_samples,  # base on *current* length
                padding_order_increase=padding_order_increase
            )
        else:
            padded_data = signal_for_fft
            n_fft = number_of_samples

        # --- FFT ---
        fft_values = rfft(padded_data,
                          n=n_fft)
        fft_frequencies_thz = rfftfreq(n_fft,
                                       d=sampling_interval_ps)  # ps → THz (since 1/ps = THz)
        # Magnitude to ASD (one-sided); multiply by sqrt(2) for discarded negative freqs
        asd_magnitude = (np.abs(fft_values) / window_correction_factor) * np.sqrt(2.0)

        fft_df = pd.DataFrame({
            'FT Magnitude': asd_magnitude,
            'Frequency (THz)': fft_frequencies_thz,
            'Wavenumber (cm^-1)': fft_frequencies_thz * 33.35641
        })
        return fft_df

    def _subtract_average(self,
                         data):
        average_value = np.mean(data)
        return data - average_value

    def _perform_zero_padding_corrected(self,
                                       input_data: np.ndarray,
                                       original_length_for_padding: int,
                                       padding_order_increase: int = 0):
        """
        Pad to a power-of-two length with optional extra powers.

        original_length_for_padding:
            The length you want to *base* the power-of-two on (usually len(input_data)).
        padding_order_increase:
            0 -> next power of two
            1 -> 2× that
            2 -> 4× that, etc.
        """
        base_power = ceil(log2(original_length_for_padding))
        target_length = 2 ** (base_power + padding_order_increase)

        if target_length < len(input_data):
            # Safety: if the chosen base is smaller than current data, bump once more
            target_length = 2 ** (ceil(log2(len(input_data))) + padding_order_increase)

        zero_padded_data = np.pad(input_data,
                                  (0, target_length - len(input_data)),
                                  mode='constant')
        return zero_padded_data, target_length

    def _filter_values_in_selection_interval(self):
        if self.FFT_time_lim is not None:
            min_val, max_val = self.FFT_time_lim
            aux_df=pd.DataFrame()
            aux_df['x']=self.time_data
            aux_df['y']=self.y_data

            filtered_values = aux_df[
                (aux_df['x'] >= min_val) & (aux_df['x'] <= max_val)]
            filtered_x=filtered_values['x']
            filtered_y=filtered_values['y']

            return filtered_x, filtered_y
        else:
            return self.time_data, self.y_data
