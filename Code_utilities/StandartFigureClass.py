from Analyses.Plotting_tools.StandardAxesClass import StandardAx
import matplotlib.pyplot as plt
from Analyses.Plotting_tools.InputParametersClass import InputParametersClass
import matplotlib as mpl
ImportParameters = InputParametersClass()

class StandardFigure:
    def __init__(self,
                 title='',
                 x_size=ImportParameters.standard_x_size,
                 y_size=ImportParameters.standard_y_size,
                 x_label='',
                 y_label='',
                 linewidth_thickness=ImportParameters.linewidth,
                 font_size=ImportParameters.fontsize,
                 thick_length=ImportParameters.tick_length,
                 nbins_x_ticks=5,
                 nbins_y_ticks=5,
                 caption = None,):

        self.x_size = x_size
        self.y_size = y_size

        self.line_width = linewidth_thickness
        self.font_size = font_size
        self.tick_length = thick_length
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.nbins_x_ticks = nbins_x_ticks
        self.nbins_y_ticks = nbins_y_ticks
        self.caption = caption


        self.fig, self.ax = self.create_standart_figure_and_axes(title,
                                                                 x_label,
                                                                 y_label)
        # self.fig.subplots_adjust(left=0.1,
        #                          right=0.9,
        #                          top=0.9,
        #                          bottom=0.1)

        # plt.draw()

    def make_ax_for_SI(self,
                       ax):

        return
    def create_standart_figure_and_axes(self,
                                        title,
                                        xlabel,
                                        ylabel):

        fig, ax = plt.subplots(figsize=(self.x_size, self.y_size))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        standard_ax_instance = StandardAx(ax,
                                          title=self.title,
                                          x_label=self.x_label,
                                          y_label=self.y_label,
                                          linewidth_thickness=self.line_width,
                                          font_size=self.font_size,
                                          tick_length=self.tick_length,
                                          nbins_x_ticks=self.nbins_x_ticks,
                                          nbins_y_ticks=self.nbins_y_ticks)

        self.add_caption(standard_ax_instance.ax)

        return fig, standard_ax_instance.ax

    def add_caption (self,
                     ax):
        if self.caption == None:
            return
        else:
            ax.text(
                0.5,
                -0.2,
                # Place below the x-axis (which is at y=0 in axis coords)
                self.caption,
                ha='center',
                va='top',
                transform=ax.transAxes,
                fontsize=self.font_size*.8,
            )

            return


    def add_standard_axes(self,
                          side='right'):

        """IT STILL DOES NOT WORK PROPERLY. CANNOT FIT THE NEW AXES IN THE FIGURE."""
        # Determine the position of the new axes based on the side
        if side == 'right':
            left = self.axes_positions[-1][0] + self.axes_positions[-1][2] + 0.1  # 0.1 is the gap between axes
            bottom = self.axes_positions[-1][1]
        elif side == 'bottom':
            left = self.axes_positions[-1][0]
            bottom = self.axes_positions[-1][1] - self.axes_positions[-1][3] - 0.1  # 0.1 is the gap between axes
        else:
            raise ValueError("Invalid side. Choose either 'right' or 'bottom'.")

        # The width and height of the new axes will be the same as the last axes
        width = self.axes_positions[-1][2]
        height = self.axes_positions[-1][3]

        # Create the new axes
        ax = self.fig.add_subplot([left, bottom, width, height])

        # Configure the new axes
        standard_ax = StandardAx(ax,
                                 title=self.title,
                                 x_label=self.x_label,
                                 y_label=self.y_label,
                                 linewidth_thickness=self.line_width,
                                 font_size=self.font_size,
                                 tick_length=self.tick_length,
                                 nbins_x_ticks=self.nbins_x_ticks,
                                 nbins_y_ticks=self.nbins_y_ticks)

        # Store the position of the new axes
        self.axes_positions.append([left, bottom, width, height])

        return standard_ax.ax
