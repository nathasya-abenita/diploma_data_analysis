import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, savgol_filter
from scipy.fft import fft, fftfreq
from scipy.signal.windows import tukey, hann, hamming

def analyze_time_series_fft(time_series, time_diff_days, window='none'):
    # Detrend the data (this automatically removes the mean as well)
    # scipy.signal.detrend uses a linear fit by default.
    detrended_series = detrend(time_series)

    # Apply a Hann window to mitigate spectral leakage
    if window == 'hamming':
        window = hamming(len(time_series))
    elif window == 'tukey':
        window = tukey(len(time_series), alpha=0.1)
    elif window == 'hann':
        window = hann(len(time_series))
    elif window == 'none':
        window = np.array([1 for _ in range (len(time_series))])
    processed_series = detrended_series * window
       
    # Compute the FFT
    fft_result = np.fft.fft(processed_series)

    # Calculate frequencies (in cycles per day)
    frequencies = np.fft.fftfreq(len(time_series), time_diff_days)

    # Calculate periods (in days) - handle zero frequency carefully
    periods = np.zeros_like(frequencies)

    # we only take the positive frequencies as the 
    # input is a real function, not complex.
    positive_freq_indices = frequencies > 0
    periods[positive_freq_indices] = 1 / frequencies[positive_freq_indices]
    periods[frequencies == 0] = np.inf  # Period is infinite for zero frequency

    # Calculate amplitudes
    amplitudes = np.abs(fft_result)

    return frequencies, periods, amplitudes


def plot_fft_spectrum(frequencies, periods, amplitudes):
    """
    Plots the frequency spectrum.

    Args:
        frequencies (np.ndarray): Frequencies (cycles/day).
        periods (np.ndarray): Periods (days).
        amplitudes (np.ndarray): Amplitudes.
    """

    positive_freq_indices = frequencies > 0

    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes

    ax.plot(periods[positive_freq_indices], amplitudes[positive_freq_indices])
    # ax.plot(periods, amplitudes)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Amplitude (log scale)")
    ax.set_title("Frequency Spectrum")
    ax.grid(True)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(365 * 76, 30)

    return fig,ax



def calculate_significant_peaks_smoothed(amplitudes, window_length=15, polyorder=3, threshold_factor=7.0):
    """
    Calculates significant peaks by comparing amplitudes to a smoothed spectrum.

    Args:
        amplitudes (np.ndarray): The FFT amplitudes.
        window_length (int): Length of the smoothing window (must be odd).
        polyorder (int): Order of the polynomial used in smoothing.
        threshold_factor (float): Factor to multiply the smoothed amplitudes for threshold.

    Returns:
        np.ndarray: Boolean array indicating significant peaks.
    """

    # Smooth the amplitudes using Savitzky-Golay filter
    smoothed_amplitudes = savgol_filter(amplitudes, window_length, polyorder)

    # Calculate the significance threshold
    significance_threshold = smoothed_amplitudes * threshold_factor

    # Determine significant peaks
    significant_peaks = amplitudes > significance_threshold

    return significant_peaks, significance_threshold

if __name__ == '__main__':
    # Define input file
    filename = r'./data/enso34/enso34_index.nc'

    # Open the NetCDF file
    ds = xr.open_dataset(filename)
    time_series = ds['sst'].data.flatten()

    # Parameter
    window='tukey'
    time_diff_days = 30 # monthly data

    # Perform FFT analysis
    frequencies, periods, amplitudes = analyze_time_series_fft(time_series, time_diff_days, window=window)
    fig,ax = plot_fft_spectrum(frequencies, periods, amplitudes)

    # # Find peaks
    # sig_peaks,thresh=calculate_significant_peaks_smoothed(amplitudes,window_length=23)

    # # Plot vertical lines at significant peaks
    # positive_freq_indices = frequencies > 0
    # peak_periods = periods[positive_freq_indices][sig_peaks[positive_freq_indices]]
    # peak_amplitudes = amplitudes[positive_freq_indices][sig_peaks[positive_freq_indices]]

    # for period, amplitude in zip(peak_periods, peak_amplitudes):
    #     ax.axvline(x=period, color='red', linestyle='--', linewidth=1) # add vertical lines
    #     ax.plot(period, amplitude, 'ro') # add red dot on peak

    # Show plot
    plt.show()