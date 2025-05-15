from PowerExtractorFromPolarizer import PowerExtractorFromPolarizers
import matplotlib
try:
    # Attempt to use TkAgg backend, which usually works well for GUIs.
    # This MUST be called before importing pyplot.
    matplotlib.use('TkAgg')
except ImportError:
    print("Warning: TkAgg backend not available. Plot window may not appear.")
    print("Consider installing tkinter in your Python environment (e.g., 'pip install tk').")
    print("Falling back to a non-interactive backend (Agg).")
    try:
        matplotlib.use('Agg') # Fallback to a non-interactive backend
    except ImportError:
        print("Warning: Agg backend also not available. Plotting might fail.")
        # Matplotlib will try its default if Agg also fails.


if __name__ == "__main__":

    Calibration = PowerExtractorFromPolarizers(known_angle=90,
                                          known_voltage_at_known_angle_in_V=0.7,
                                          desired_angle=90)

    Calibration.plot_polarizer_calibration()
