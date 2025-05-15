from PowerExtractorFromPolarizer import PowerExtractorFromPolarizers
import matplotlib
if ("pycharm" in matplotlib.get_backend().lower()) or ("backend_interagg" in matplotlib.get_backend().lower()):
    # Prefer a GUI backend; fall back to the non-interactive 'Agg' backend.
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")

if __name__ == "__main__":

    Calibration = PowerExtractorFromPolarizers(known_angle=90,
                                          known_voltage_at_known_angle_in_V=0.7,
                                          desired_angle=90)

    Calibration.plot_polarizer_calibration()
