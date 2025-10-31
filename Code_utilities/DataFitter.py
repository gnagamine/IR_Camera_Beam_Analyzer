import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Code_utilities.StandartFigureClass import StandardFigure
from Code_utilities.InputParametersClass import InputParametersClass


# --- Model Functions ---
def linear_model(x,
                 a,
                 b):
    """
    A linear model function: y = a*x + b.

    Args:
        x (np.ndarray or float): Independent variable.
        a (float): Slope.
        b (float): Intercept.

    Returns:
        np.ndarray or float: Dependent variable y.
    """
    return a * x + b


def quadratic_model(x,
                    a,
                    b,
                    c):
    """
    A quadratic model function: y = a*x^2 + b*x + c.

    Args:
        x (np.ndarray or float): Independent variable.
        a (float): Quadratic coefficient.
        b (float): Linear coefficient.
        c (float): Constant term.

    Returns:
        np.ndarray or float: Dependent variable y.
    """
    return a * x ** 2 + b * x + c


class DataFitter:
    """
    A class to fit data using a specified model function (e.g., linear)
    and plot the results.
    """

    def __init__(self,
                 model_function='linear_model'):
        """
        Initializes the DataFitter with a model function.

        Args:
            model_function (callable): The function to fit the data to.
                                       It should take x-data as the first argument,
                                       followed by the model parameters.
                                       Defaults to linear_model.
        """
        self._set_fitting_model(model_function)
        self.fit_params = None
        self.fit_covariance = None

    def _set_fitting_model(self,
                           model_function):
        if model_function == 'linear_model':
            self.model_function = linear_model
        elif model_function == 'quadratic_model':
            self.model_function = quadratic_model

    def fit_data(self,
                 x_data,
                 y_data,
                 initial_guess=None):
        """
        Fits the provided x and y data using the stored model function.

        Args:
            x_data (array-like): The independent data.
            y_data (array-like): The dependent data.
            initial_guess (array-like, optional): Initial guess for the parameters.
                                                  If None, curve_fit will try to find its own.

        Returns:
            tuple:
                - popt (np.ndarray): Optimal values for the parameters.
                - pcov (np.ndarray): The estimated covariance of popt.
                                     The diagonals provide the variance of the parameter estimate.

        Raises:
            RuntimeError: If the fit is unsuccessful.
            ValueError: If x_data and y_data have mismatched shapes or are empty.
        """
        x_data_np = np.asarray(x_data)
        y_data_np = np.asarray(y_data)

        if x_data_np.shape != y_data_np.shape:
            raise ValueError("x_data and y_data must have the same shape.")
        if x_data_np.size == 0:
            raise ValueError("Input data cannot be empty.")

        try:
            # Perform the curve fit
            popt, pcov =  curve_fit(self.model_function,
                                   x_data_np,
                                   y_data_np,
                                   p0=initial_guess)
            self.fit_params = popt
            self.fit_covariance = pcov
            return popt, pcov
        except RuntimeError as e:
            print(f"Error during fitting: {e}")
            print("Consider providing an initial_guess (p0) if the fit fails to converge.")
            self.fit_params = None
            self.fit_covariance = None
            raise
        except Exception as e:
            print(f"An unexpected error occurred during fitting: {e}")
            self.fit_params = None
            self.fit_covariance = None
            raise


    def plot_data_with_fit(self,
                           x_data,
                           y_data,
                           fit_params=None,
                           title="Data with Fit",
                           x_label="X-data",
                           y_label="Y-data",
                           data_label="Original Data",
                           fit_label="Fitted Model",
                           show_plot=True):
        """
        Plots the original data points with x-error bars and the fitted line/curve.

        Args:
            x_data (array-like): The independent data.
            y_data (array-like): The dependent data.
            fit_params (array-like, optional): The parameters of the fitted model.
                                               If None, uses the parameters from the last
                                               successful call to fit_data().
            title (str): The title of the plot.
            x_label (str): The label for the x-axis.
            y_label (str): The label for the y-axis.
            data_label (str): Label for the original data points in the legend.
            fit_label (str): Label for the fitted line/curve in the legend.
            show_plot (bool): If True, calls plt.show() to display the plot.

        Returns:
            tuple: (fig, ax) The matplotlib figure and axes objects.

        Raises:
            ValueError: If fit_params are not provided and no fit has been performed yet.
        """
        if fit_params is None and self.fit_params is None:
            self.fit_data(x_data=x_data,
                          y_data=y_data)
            current_fit_params = self.fit_params
        else:
            current_fit_params = fit_params

        if current_fit_params is None:
            raise ValueError("No fit parameters available. Please run fit_data() first or provide fit_params.")

        x_data_np = np.asarray(x_data)
        y_data_np = np.asarray(y_data)

        fig, ax = plt.subplots()

        # Calculate 5% error for x
        x_error = 0.05 * x_data_np

        # Plot original data with error bars
        ax.errorbar(x_data_np,
                    y_data_np,
                    xerr=x_error,
                    fmt='o',  # 'o' for points, no line
                    label=data_label,
                    markersize=5,
                    capsize=3)  # capsize for the error bar caps

        # Generate x values for the fitted line/curve to make it smooth
        # Sort x_data for a smooth line plot, especially for non-linear fits
        x_fit = np.linspace(x_data_np.min(),
                            x_data_np.max(),
                            200)
        y_fit = self.model_function(x_fit,
                                    *current_fit_params)

        # Plot fitted line/curve
        ax.plot(x_fit,
                y_fit,
                '-',
                label=fit_label,
                linewidth=2)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True,
                linestyle=':',
                alpha=0.7)

        if show_plot:
            plt.show()

        return fig, ax

    def get_fitting_coefficients(self,
                                 x_data,
                                 y_data, ):

        self.fit_data(x_data=x_data,
                      y_data=y_data)
        current_fit_params = self.fit_params

        return current_fit_params

    def plot_data_with_fit_for_paper(self,
                                     x_data,
                                     y_data,
                                     fit_params=None,
                                     title="Data with Fit",
                                     x_label="X-data",
                                     y_label="Y-data",
                                     data_label="Original Data",
                                     fit_label="Fitted Model",
                                     show_plot=True,
                                     save_folder=None,
                                     camera_name=None):
        """
        Plots the original data points with x-error bars and the fitted line/curve.

        Args:
            x_data (array-like): The independent data.
            y_data (array-like): The dependent data.
            fit_params (array-like, optional): The parameters of the fitted model.
                                               If None, uses the parameters from the last
                                               successful call to fit_data().
            title (str): The title of the plot.
            x_label (str): The label for the x-axis.
            y_label (str): The label for the y-axis.
            data_label (str): Label for the original data points in the legend.
            fit_label (str): Label for the fitted line/curve in the legend.
            show_plot (bool): If True, calls plt.show() to display the plot.

        Returns:
            tuple: (fig, ax) The matplotlib figure and axes objects.

        Raises:
            ValueError: If fit_params are not provided and no fit has been performed yet.
        """


        if fit_params is None and self.fit_params is None:
            self.fit_data(x_data=x_data,
                          y_data=y_data)
            current_fit_params = self.fit_params
        else:
            current_fit_params = fit_params

        if current_fit_params is None:
            raise ValueError("No fit parameters available. Please run fit_data() first or provide fit_params.")

        x_data_np = np.asarray(x_data)
        y_data_np = np.asarray(y_data)

        fontsize = InputParametersClass().fontsize
        LinFig = StandardFigure(font_size=fontsize)
        fig, ax = LinFig.fig, LinFig.ax

        # Calculate 5% error for x
        x_error = 0.05 * x_data_np

        # Plot original data with error bars
        ax.errorbar(x_data_np,
                    y_data_np,
                    xerr=x_error,
                    fmt='o',  # 'o' for points, no line
                    linewidth=.5,
                    markersize=4,
                    capsize=2,
                    color=InputParametersClass().color_nice_blue,
                    alpha=.8)  # capsize for the error bar caps

        # Generate x values for the fitted line/curve to make it smooth
        # Sort x_data for a smooth line plot, especially for non-linear fits
        x_fit = np.linspace(x_data_np.min(),
                            x_data_np.max(),
                            200)
        y_fit = self.model_function(x_fit,
                                    *current_fit_params)

        # Plot fitted line/curve
        ax.plot(x_fit,
                y_fit,
                '-',
                linewidth=1,
                color=InputParametersClass().color_nice_orange,
                alpha=.8)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(False, )
        # ax.set_xlim(0, 52)
        fig_size = 2.2
        fig.set_size_inches(fig_size * 1.2,
                            fig_size)
        fig.tight_layout()

        if show_plot:
            plt.show()
        if save_folder is not None:
            fig.savefig(save_folder + '/' + camera_name + '_linearity.pdf',
                        bbox_inches='tight')

        return fig, ax
