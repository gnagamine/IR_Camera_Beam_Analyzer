import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Force an interactive GUI backend so that plots appear in their own window
# Try macOS‑native backend first; fall back to TkAgg if that isn't available.
try:
    matplotlib.use("MacOSX")
except Exception:
    matplotlib.use("TkAgg")


def parse_relative_time(time_str):
    """Convert relative time string (e.g., '0:0:0.40') to seconds."""
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")
    h, m, s_ms = parts
    s, ms = s_ms.split('.')
    total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    return total_seconds


def analyze_temperature_points(file_path):
    """Analyze temperature data from CSV and return time trace and RMS values."""
    # Read CSV, using second row as header
    df = pd.read_csv(file_path,
                     header=1)

    # Rename columns for clarity
    df.columns = ["Timestamp", "Relative_Time", "P1_Temp", "P2_Temp"]

    # Convert relative time to seconds
    df["Time"] = df["Relative_Time"].apply(parse_relative_time)

    # Create time trace DataFrame
    time_trace = df[["Time", "P1_Temp", "P2_Temp"]]

    # Calculate RMS for P1 and P2
    rms_p1 = np.sqrt((df["P1_Temp"] ** 2).mean())
    rms_p2 = np.sqrt((df["P2_Temp"] ** 2).mean())

    # Calculate standard deviation
    std_p1 = df["P1_Temp"].std()
    std_p2 = df["P2_Temp"].std()

    # Return time trace and RMS values
    return time_trace, {"P1": rms_p1, "P2": rms_p2}, {"P1": std_p1, "P2": std_p2}


def plot_temperature_trace(time_trace):
    """
    Plots temperature as a function of time for P1 and P2.

    Parameters:
    time_trace (pd.DataFrame): DataFrame with columns 'Time', 'P1_Temp', 'P2_Temp'
    """
    fig, ax = plt.subplots()
    ax.plot(time_trace['Time'],
             time_trace['P1_Temp'],
             label='P1')
    plt.plot(time_trace['Time'],
             time_trace['P2_Temp'],
             label='P2')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature vs. Time for P1 and P2')
    ax.legend()
    ax.grid(False)
    #ax.set_ylim(0, 12)
    fig.show()
    fig.savefig('temperature_trace.pdf')


if __name__ == "__main__":
    path = '/Users/Shared/Files From c.localized/Gabriel_UniBern_Local/DataAnalysis/Low cost THz Camera/20250508/Noise_Analysis/IR_00002_Temperature Statistics_P1P2_Allframes.csv'
    time_trace, rms, std = analyze_temperature_points(path)
    print("Time Trace:")
    print(time_trace)
    print("\nRMS Values:")
    print(f"P1: {rms['P1']:.2f} °C")
    print(f"P2: {rms['P2']:.2f} °C")

    print("\nStandard Deviation Values:")
    print(f"P1: {std['P1']:.2f} °C")
    print(f"P2: {std['P2']:.2f} °C")

    plot_temperature_trace(time_trace)