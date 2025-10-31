import matplotlib.pyplot as plt
import numpy as np
from Code_utilities.InputParametersClass import InputParametersClass


def calculate_fwhm(profile,
                   pixel_size):
    """
    Calculates the Full Width at Half Maximum (FWHM) of a 1D profile.
    Uses linear interpolation for better accuracy.

    Args:
        profile (np.ndarray): The 1D array representing the beam profile.
        pixel_size (float): The size of a pixel in micrometers.

    Returns:
        tuple: A tuple containing:
            - fwhm_um (float): The FWHM value in micrometers.
            - left_coord_um (float): The left FWHM coordinate relative to the peak (µm).
            - right_coord_um (float): The right FWHM coordinate relative to the peak (µm).
    """
    try:
        peak_value = np.max(profile)
        half_max = peak_value / 2.0
        peak_idx = np.argmax(profile)

        # Find all indices where the profile is above half-max
        above_half_max_indices = np.where(profile > half_max)[0]
        if len(above_half_max_indices) < 2:
            return 0, 0, 0

        # --- Find left coordinate by interpolation ---
        left_edge_idx = above_half_max_indices[0]
        if left_edge_idx == 0:
            left_coord_pix = 0
        else:
            y1, y2 = profile[left_edge_idx - 1], profile[left_edge_idx]
            x1, x2 = left_edge_idx - 1, left_edge_idx
            left_coord_pix = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

        # --- Find right coordinate by interpolation ---
        right_edge_idx = above_half_max_indices[-1]
        if right_edge_idx == len(profile) - 1:
            right_coord_pix = len(profile) - 1
        else:
            y1, y2 = profile[right_edge_idx], profile[right_edge_idx + 1]
            x1, x2 = right_edge_idx, right_edge_idx + 1
            right_coord_pix = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

        fwhm_pixels = right_coord_pix - left_coord_pix
        fwhm_um = fwhm_pixels * pixel_size

        # Convert coordinates to be relative to the peak for plotting
        left_coord_um = (left_coord_pix - peak_idx) * pixel_size
        right_coord_um = (right_coord_pix - peak_idx) * pixel_size

        return fwhm_um, left_coord_um, right_coord_um
    except Exception:
        return 0, 0, 0


def calculate_fwhm(profile,
                   pixel_size):
    """
    Calculates the Full Width at Half Maximum (FWHM) of a 1D profile.
    Uses linear interpolation for better accuracy.

    Args:
        profile (np.ndarray): The 1D array representing the beam profile.
        pixel_size (float): The size of a pixel in micrometers.

    Returns:
        tuple: A tuple containing:
            - fwhm_um (float): The FWHM value in micrometers.
            - left_coord_um (float): The left FWHM coordinate relative to the peak (µm).
            - right_coord_um (float): The right FWHM coordinate relative to the peak (µm).
    """
    try:
        peak_value = np.max(profile)
        half_max = peak_value / 2.0
        peak_idx = np.argmax(profile)

        # Find all indices where the profile is above half-max
        above_half_max_indices = np.where(profile > half_max)[0]
        if len(above_half_max_indices) < 2:
            return 0, 0, 0

        # --- Find left coordinate by interpolation ---
        left_edge_idx = above_half_max_indices[0]
        if left_edge_idx == 0:
            left_coord_pix = 0
        else:
            y1, y2 = profile[left_edge_idx - 1], profile[left_edge_idx]
            x1, x2 = left_edge_idx - 1, left_edge_idx
            left_coord_pix = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

        # --- Find right coordinate by interpolation ---
        right_edge_idx = above_half_max_indices[-1]
        if right_edge_idx == len(profile) - 1:
            right_coord_pix = len(profile) - 1
        else:
            y1, y2 = profile[right_edge_idx], profile[right_edge_idx + 1]
            x1, x2 = right_edge_idx, right_edge_idx + 1
            right_coord_pix = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

        fwhm_pixels = right_coord_pix - left_coord_pix
        fwhm_um = fwhm_pixels * pixel_size

        # Convert coordinates to be relative to the peak for plotting
        left_coord_um = (left_coord_pix - peak_idx) * pixel_size
        right_coord_um = (right_coord_pix - peak_idx) * pixel_size

        return fwhm_um, left_coord_um, right_coord_um
    except Exception:
        return 0, 0, 0


def plot_beam_comparison(beam1_data,
                         beam2_data,
                         pixel_size1,
                         pixel_size2,
                         label_1,
                         label_2,
                         x_shift_beam_1=0,
                         y_shift_beam_1=0,
                         show_plot=True,
                         save_plot=False,
                         filename='beam_comparison.pdf'):
    """
    Plots a comparison of two beam profiles, scaling axes to micrometers,
    aligning profiles by peak intensity, and calculating FWHM.

    Args:
        beam1_data (np.ndarray): The 2D array representing the first beam.
        beam2_data (np.ndarray): The 2D array representing the second beam.
        pixel_size1 (float): The pixel size in micrometers for the first beam.
        pixel_size2 (float): The pixel size in micrometers for the second beam.
        label_1 (str): Label for the first beam.
        label_2 (str): Label for the second beam.
        x_shift_beam_1 (float): Manual shift for beam 1's X-profile (in um).
        y_shift_beam_1 (float): Manual shift for beam 1's Y-profile (in um).
        show_plot (bool): If True, displays the plot.
        save_plot (bool): If True, saves the plot to a file.
        filename (str): The name of the file to save the plot to.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
        matplotlib.axes.Axes: The axes object containing the subplots.
    """
    fig, axs = plt.subplots(3,
                            1,
                            figsize=(4, 12))
    fig.suptitle('Beam Profile Comparison (Centered on Peak Intensity)',
                 fontsize=14)

    # --- Process Beam 1 ---
    rows1, cols1 = beam1_data.shape
    max_row1, max_col1 = np.unravel_index(np.argmax(beam1_data),
                                          (rows1, cols1))
    y_extent1, x_extent1 = rows1 * pixel_size1, cols1 * pixel_size1
    peak_x1_um, peak_y1_um = max_col1 * pixel_size1, max_row1 * pixel_size1

    im1 = axs[0].imshow(beam1_data,
                        extent=[0, x_extent1, 0, y_extent1],
                        aspect='auto',
                        cmap='viridis',
                        origin='lower')
    axs[0].set_title(label_1,
                     fontsize=12)
    axs[0].set_xlabel('Position (μm)')
    axs[0].set_ylabel('Position (μm)')
    axs[0].plot(peak_x1_um,
                peak_y1_um,
                'w+',
                markersize=8,
                markeredgewidth=2,
                label='Peak Intensity')
    axs[0].legend(loc='upper right',
                  fontsize='small')
    fig.colorbar(im1,
                 ax=axs[0],
                 label='Intensity')

    profile_x1, profile_y1 = beam1_data[max_row1, :], beam1_data[:, max_col1]
    x_axis1_shifted = (np.arange(cols1) * pixel_size1) - peak_x1_um - x_shift_beam_1
    y_axis1_shifted = (np.arange(rows1) * pixel_size1) - peak_y1_um - y_shift_beam_1

    fwhm_x1, fwhm_x1_left, fwhm_x1_right = calculate_fwhm(profile_x1,
                                                          pixel_size1)
    fwhm_y1, fwhm_y1_left, fwhm_y1_right = calculate_fwhm(profile_y1,
                                                          pixel_size1)

    # --- Process Beam 2 ---
    rows2, cols2 = beam2_data.shape
    max_row2, max_col2 = np.unravel_index(np.argmax(beam2_data),
                                          (rows2, cols2))
    y_extent2, x_extent2 = rows2 * pixel_size2, cols2 * pixel_size2
    peak_x2_um, peak_y2_um = max_col2 * pixel_size2, max_row2 * pixel_size2

    im2 = axs[1].imshow(beam2_data,
                        extent=[0, x_extent2, 0, y_extent2],
                        aspect='auto',
                        cmap='plasma',
                        origin='lower')
    axs[1].set_title(label_2,
                     fontsize=12)
    axs[1].set_xlabel('Position (μm)')
    axs[1].set_ylabel('Position (μm)')
    axs[1].plot(peak_x2_um,
                peak_y2_um,
                'w+',
                markersize=8,
                markeredgewidth=2,
                label='Peak Intensity')
    axs[1].legend(loc='upper right',
                  fontsize='small')
    fig.colorbar(im2,
                 ax=axs[1],
                 label='Intensity')

    profile_x2, profile_y2 = beam2_data[max_row2, :], beam2_data[:, max_col2]
    x_axis2_shifted = (np.arange(cols2) * pixel_size2) - peak_x2_um
    y_axis2_shifted = (np.arange(rows2) * pixel_size2) - peak_y2_um

    fwhm_x2, fwhm_x2_left, fwhm_x2_right = calculate_fwhm(profile_x2,
                                                          pixel_size2)
    fwhm_y2, fwhm_y2_left, fwhm_y2_right = calculate_fwhm(profile_y2,
                                                          pixel_size2)

    # --- Print FWHM results to console ---
    print("\n" + "=" * 30)
    print("      FWHM Analysis Results")
    print("=" * 30)
    print(f"Beam 1 ({label_1}):")
    print(f"  - X-Profile FWHM: {fwhm_x1:.2f} µm")
    print(f"  - Y-Profile FWHM: {fwhm_y1:.2f} µm")
    print(f"\nBeam 2 ({label_2}):")
    print(f"  - X-Profile FWHM: {fwhm_x2:.2f} µm")
    print(f"  - Y-Profile FWHM: {fwhm_y2:.2f} µm")
    print("=" * 30 + "\n")

    # --- Comparison Plots ---
    # X-Profile Comparison
    ax = axs[2]
    ax.plot(x_axis1_shifted,
            profile_x1,
            label=f"{label_1} (FWHM: {fwhm_x1:.2f} µm)",
            color='blue')
    ax.plot(x_axis2_shifted,
            profile_x2,
            label=f"{label_2} (FWHM: {fwhm_x2:.2f} µm)",
            color='red')
    if fwhm_x1 > 0: ax.plot([fwhm_x1_left - x_shift_beam_1, fwhm_x1_right - x_shift_beam_1],
                            [np.max(profile_x1) / 2, np.max(profile_x1) / 2],
                            '--',
                            color='blue')
    if fwhm_x2 > 0: ax.plot([fwhm_x2_left, fwhm_x2_right],
                            [np.max(profile_x2) / 2, np.max(profile_x2) / 2],
                            '--',
                            color='red')
    ax.set_title('X-Profile Comparison (at Peak Intensity)',
                 fontsize=10)
    ax.set_xlabel('Position relative to peak (μm)')
    ax.set_ylabel('Intensity')
    ax.legend(fontsize='small')
    ax.grid(True,
            linestyle='--',
            alpha=0.6)

    # Y-Profile Comparison

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_plot:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

    if show_plot:
        plt.show()

    return fig, axs


def plot_beam_comparison_for_paper(beam1_data,
                                   beam2_data,
                                   pixel_size1,
                                   pixel_size2,
                                   label_1,
                                   label_2,
                                   figures_x_size_inches,
                                   figures_y_size_inches,
                                   x_shift_beam_1=0,
                                   y_shift_beam_1=0,
                                   plot_size_um=None,
                                   show_plot=True,
                                   save_plot=False,
                                   filename='beam_comparison.pdf',
                                   fontsize=7,
                                   x_shift_thz_um=0,
                                   bool_remove_y_labels=False,
                                   bool_remove_profile_comparison_legend=False):
    """
    Plots a comparison of two beam profiles, scaling axes to micrometers,
    aligning profiles by peak intensity, and calculating FWHM.

    Args:
        beam1_data (np.ndarray): The 2D array representing the first beam.
        beam2_data (np.ndarray): The 2D array representing the second beam.
        pixel_size1 (float): The pixel size in micrometers for the first beam.
        pixel_size2 (float): The pixel size in micrometers for the second beam.
        label_1 (str): Label for the first beam.
        label_2 (str): Label for the second beam.
        x_shift_beam_1 (float): Manual shift for beam 1's X-profile (in um).
        y_shift_beam_1 (float): Manual shift for beam 1's Y-profile (in um).
        show_plot (bool): If True, displays the plot.
        save_plot (bool): If True, saves the plot to a file.
        filename (str): The name of the file to save the plot to.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
        matplotlib.axes.Axes: The axes object containing the subplots.
    """
    fig, axs = plt.subplots(3,
                            1,
                            figsize=(figures_x_size_inches, figures_y_size_inches * 3))

    # --- Process Beam 1 ---
    rows1, cols1 = beam1_data.shape
    max_row1, max_col1 = np.unravel_index(np.argmax(beam1_data),
                                          (rows1, cols1))
    y_extent1, x_extent1 = rows1 * pixel_size1, cols1 * pixel_size1
    peak_x1_um, peak_y1_um = max_col1 * pixel_size1, max_row1 * pixel_size1

    fig, axs[0] = plot_map_for_paper(beam_data=beam1_data,
                                     peak_x_um=peak_x1_um,
                                     x_extent=x_extent1,
                                     peak_y_um=peak_y1_um,
                                     y_extent=y_extent1,
                                     plot_size_um=plot_size_um,
                                     ax=axs[0],
                                     fig=fig,
                                     color_map='viridis',
                                     bool_remove_y_labels=bool_remove_y_labels)

    # --- Process Beam 2 ---
    rows2, cols2 = beam2_data.shape
    max_row2, max_col2 = np.unravel_index(np.argmax(beam2_data),
                                          (rows2, cols2))
    y_extent2, x_extent2 = rows2 * pixel_size2, cols2 * pixel_size2
    peak_x2_um, peak_y2_um = max_col2 * pixel_size2, max_row2 * pixel_size2

    fig, axs[1] = plot_map_for_paper(beam_data=beam2_data,
                                     peak_x_um=peak_x2_um,
                                     x_extent=x_extent2,
                                     peak_y_um=peak_y2_um,
                                     y_extent=y_extent2,
                                     plot_size_um=plot_size_um,
                                     ax=axs[1],
                                     fig=fig,
                                     color_map='plasma',
                                     bool_remove_x_axis_label=True,
                                     bool_remove_y_labels=bool_remove_y_labels)

    # --- Comparison Plots ---

    plot_profile_comparison_for_paper(beam1_data=beam1_data,
                                      beam2_data=beam2_data,
                                      max_row1=max_row1,
                                      max_col1=max_col1,
                                      max_row2=max_row2,
                                      max_col2=max_col2,
                                      pixel_size1=pixel_size1,
                                      pixel_size2=pixel_size2,
                                      plot_size_um=plot_size_um,
                                      label_1=label_1,
                                      label_2=label_2,
                                      fig=fig,
                                      ax=axs[2],
                                      x_shift_thz_um=x_shift_thz_um,
                                      bool_remove_y_labels=bool_remove_y_labels,
                                      bool_remove_profile_comparison_legend=bool_remove_profile_comparison_legend,
                                      bool_dashed=True)

    fig = _set_all_fontsizes(fig,
                             fontsize)
    # Use h_pad to increase the vertical space between subplots
    # fig.tight_layout(h_pad=2.0)

    if save_plot:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

    if show_plot:
        plt.show()

    return fig, axs


def plot_map_for_paper(beam_data,
                       peak_x_um,
                       x_extent,
                       peak_y_um,
                       y_extent,
                       plot_size_um,
                       ax,
                       fig,
                       bool_remove_y_labels,
                       bool_remove_x_axis_label=False,
                       color_map='viridis',
                       x_label_position='top',
                       ):
    im1 = ax.imshow(beam_data,
                    extent=[0 - peak_x_um, x_extent - peak_x_um, 0 - peak_y_um, y_extent - peak_y_um],
                    aspect='auto',
                    cmap=color_map,
                    origin='lower')

    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Position (μm)',
                  labelpad=2)  # smaller pad → closer

    if x_label_position == 'top':
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

    ax.set_xlim(-plot_size_um,
                plot_size_um)
    ax.set_ylim(-plot_size_um,
                plot_size_um)

    if bool_remove_y_labels:
        print(' Y Axis label removed')
        ax.set_ylabel("")  # remove x-axis label text
        ax.tick_params(
            axis='y',
            which='both',
            labelleft=False,  # hide the tick labels
            labelright=False)  # hide labels on right if present

    if bool_remove_x_axis_label:
        ax.set_xlabel("")  # remove x-axis label text
        ax.tick_params(axis='x',
                       which='both',
                       labelleft=False,  # hide the tick labels
                       labelright=False,
                       labeltop=False)  # hide labels on right if present

    # fig.colorbar(im1,
    #              ax=ax,
    #              label='Intensity')

    return fig, ax


def plot_profile_comparison_for_paper(beam1_data,
                                      beam2_data,
                                      max_row1,
                                      max_col1,
                                      max_row2,
                                      max_col2,
                                      pixel_size1,
                                      pixel_size2,
                                      plot_size_um,
                                      label_1,
                                      label_2,
                                      fig,
                                      ax,
                                      x_shift_thz_um,
                                      bool_remove_y_labels,
                                      bool_remove_profile_comparison_legend,
                                      bool_dashed=False
                                      ):
    fig_x_profile, ax_x_profile = fig, ax

    rows1, cols1 = beam1_data.shape
    rows2, cols2 = beam2_data.shape

    profile_x1, profile_y1 = beam1_data[max_row1, :], beam1_data[:, max_col1]
    profile_x2, profile_y2 = beam2_data[max_row2, :], beam2_data[:, max_col2]

    peak_x1_um, peak_y1_um = max_col1 * pixel_size1, max_row1 * pixel_size1
    peak_x2_um, peak_y2_um = max_col2 * pixel_size2, max_row2 * pixel_size2

    x_axis1_shifted = (np.arange(cols1) * pixel_size1) - peak_x1_um
    x_axis2_shifted = (np.arange(cols2) * pixel_size2) - peak_x2_um + x_shift_thz_um
    ax_x_profile.plot(x_axis1_shifted,
                      profile_x1,
                      color=InputParametersClass().color_nice_blue,
                      label=label_1, )
    if bool_dashed:
        ax_x_profile.plot(x_axis2_shifted,
                          profile_x2,
                          color=InputParametersClass().color_nice_orange,
                          label=label_2,
                          linestyle='--', )

    else:
        ax_x_profile.plot(x_axis2_shifted,
                          profile_x2,
                          color=InputParametersClass().color_nice_orange,
                          label=label_2, )

    ax_x_profile.set_xlim(-750,
                          +750)

    if bool_remove_profile_comparison_legend is False:

        ax_x_profile.legend(frameon=False,
                            loc='upper right',
                            bbox_to_anchor=(1.07, 1.1),
                            handlelength=1.0,
                            labelspacing=-.1, )
    else:
        ax_x_profile.set_ylim(-.03,
                              1.1)

    from matplotlib.ticker import NullLocator

    # Make major ticks point inward on both axes
    ax_x_profile.tick_params(direction='in',
                             which='major')

    # Remove minor ticks from both axes
    ax_x_profile.xaxis.set_minor_locator(NullLocator())
    ax_x_profile.yaxis.set_minor_locator(NullLocator())

    ax_x_profile.set_xlabel('Position (μm)')
    ax_x_profile.set_ylabel('Intensity (normalized)')

    ax_x_profile.grid(False)
    ax_x_profile.set_xlim(-plot_size_um,
                          plot_size_um)

    ax_x_profile.set_ylim(-.03,
                          1.5)

    if bool_remove_y_labels:
        print(' Y Axis label removed')
        ax_x_profile.set_ylabel("")  # remove x-axis label text
        ax_x_profile.tick_params(
            axis='y',
            which='both',
            labelleft=False,  # hide the tick labels
            labelright=False)  # hide labels on right if present

    return fig_x_profile, ax_x_profile


def _set_all_fontsizes(fig,
                       fontsize):
    for ax in fig.axes:  # includes plot axes AND colorbar axes
        # titles/axis labels
        ax.title.set_fontsize(fontsize)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)

        # tick labels
        for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
            ticklabel.set_fontsize(fontsize)

        # legends (if any)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(fontsize)

    return fig


def plot_beam_comparison_forQCL_paper_figure(beam1_data,
                                             beam2_data,
                                             pixel_size1,
                                             pixel_size2,
                                             label_1,
                                             label_2,
                                             figures_x_size_inches,
                                             figures_y_size_inches,
                                             x_shift_beam_1=0,
                                             y_shift_beam_1=0,
                                             plot_size_um=None,
                                             show_plot=True,
                                             save_plot=False,
                                             filename='beam_comparison.pdf',
                                             fontsize=7,
                                             x_shift_thz_um=0,
                                             bool_remove_y_labels=False,
                                             bool_remove_profile_comparison_legend=False):
    """
    Plots a comparison of two beam profiles, scaling axes to micrometers,
    aligning profiles by peak intensity, and calculating FWHM.

    Args:
        beam1_data (np.ndarray): The 2D array representing the first beam.
        beam2_data (np.ndarray): The 2D array representing the second beam.
        pixel_size1 (float): The pixel size in micrometers for the first beam.
        pixel_size2 (float): The pixel size in micrometers for the second beam.
        label_1 (str): Label for the first beam.
        label_2 (str): Label for the second beam.
        x_shift_beam_1 (float): Manual shift for beam 1's X-profile (in um).
        y_shift_beam_1 (float): Manual shift for beam 1's Y-profile (in um).
        show_plot (bool): If True, displays the plot.
        save_plot (bool): If True, saves the plot to a file.
        filename (str): The name of the file to save the plot to.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
        matplotlib.axes.Axes: The axes object containing the subplots.
    """
    fig, axs = plt.subplots(1,
                            3,
                            figsize=(figures_y_size_inches * 3.5, figures_x_size_inches))

    # --- Process Beam 1 ---
    beam1_data = np.rot90(beam1_data,
                          k=1)
    rows1, cols1 = beam1_data.shape
    max_row1, max_col1 = np.unravel_index(np.argmax(beam1_data),
                                          (rows1, cols1))
    y_extent1, x_extent1 = rows1 * pixel_size1, cols1 * pixel_size1
    peak_x1_um, peak_y1_um = max_col1 * pixel_size1, max_row1 * pixel_size1

    fig, axs[0] = plot_map_for_paper(beam_data=beam1_data,
                                     peak_x_um=peak_x1_um,
                                     x_extent=x_extent1,
                                     peak_y_um=peak_y1_um,
                                     y_extent=y_extent1,
                                     plot_size_um=plot_size_um,
                                     ax=axs[0],
                                     fig=fig,
                                     color_map='viridis',
                                     bool_remove_y_labels=bool_remove_y_labels,
                                     x_label_position='bottom')

    # --- Process Beam 2 ---
    rows2, cols2 = beam2_data.shape
    max_row2, max_col2 = np.unravel_index(np.argmax(beam2_data),
                                          (rows2, cols2))
    y_extent2, x_extent2 = rows2 * pixel_size2, cols2 * pixel_size2
    peak_x2_um, peak_y2_um = max_col2 * pixel_size2, max_row2 * pixel_size2

    fig, axs[1] = plot_map_for_paper(beam_data=beam2_data,
                                     peak_x_um=peak_x2_um,
                                     x_extent=x_extent2,
                                     peak_y_um=peak_y2_um,
                                     y_extent=y_extent2,
                                     plot_size_um=plot_size_um,
                                     ax=axs[1],
                                     fig=fig,
                                     color_map='plasma',
                                     bool_remove_x_axis_label=False,
                                     bool_remove_y_labels=True,
                                     x_label_position='bottom')

    # --- Comparison Plots ---

    plot_profile_comparison_for_paper(beam1_data=beam1_data,
                                      beam2_data=beam2_data,
                                      max_row1=max_row1,
                                      max_col1=max_col1,
                                      max_row2=max_row2,
                                      max_col2=max_col2,
                                      pixel_size1=pixel_size1,
                                      pixel_size2=pixel_size2,
                                      plot_size_um=plot_size_um,
                                      label_1=label_1,
                                      label_2=label_2,
                                      fig=fig,
                                      ax=axs[2],
                                      x_shift_thz_um=x_shift_thz_um,
                                      bool_remove_y_labels=bool_remove_y_labels,
                                      bool_remove_profile_comparison_legend=bool_remove_profile_comparison_legend)

    fig = _set_all_fontsizes(fig,
                             fontsize)
    # Use h_pad to increase the vertical space between subplots
    # fig.tight_layout(h_pad=2.0)

    if save_plot:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

    if show_plot:
        plt.show()

    return fig, axs
