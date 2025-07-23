#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:28:11 2024

@author: gabrielnagamine
"""
import os


class InputParametersClass:

    def __init__(self):



        """ploting parameters"""

        self.fontsize = 8
        self.fontsize_SI = 11
        self.fontsize_legend = 6
        self.fontsize_legend_SI = 9
        self.fontsize_word_text = 11
        self.linewidth = .5
        self.linewidth_SI = self.linewidth*2
        self.plot_linewidth = self.linewidth * 2
        self.plot_linewidth_SI = self.plot_linewidth * 2
        self.circle_size_scatter = 15
        self.circle_size_lifetime = 7
        self.circle_size_boxplot = 15
        self.circle_size_individual_plot = 7
        self.circle_size_scatter_ensemble_spectrum = 1
        self.standard_x_size = 7
        self.standard_y_size = self.standard_x_size * 3/4
        self.intensity_tracing_color = '#1f77b4'
        self.box_plot_scatter_linewidth = .2
        self.tick_length = 3
        self.tick_length_SI = self.tick_length * 2

        self.color_nice_blue= (72 / 255, 122 / 255, 220 / 255)
        self.color_nice_orange = (226 / 255, 120 / 255, 94 / 255)
        self.color_nice_green = (0 / 255, 158 / 255, 115 / 255)
        self.color_nice_pink = (188 / 255, 112 / 255, 157 / 255)
        self.AP_3_120_color_g2_fitting = (105 / 255, 50 / 255, 80 / 255)
        self.AP_3_120_color_g2_hist = (171 / 255, 178 / 255, 215 / 255)
        # self.AP_3_120_color = (224 / 255, 199 / 255, 66 / 255)
        self.HR_90_color = (38 / 255, 139 / 255,
                            133 / 255)  # (198 / 255, 150 / 255, 110 / 255)#(222 / 255, 195 / 255,
        self.HR_90_color_2 = (198 / 255, 150 / 255, 110 / 255)#(222 / 255, 195 / 255,
        # 164 / 255)#'#19439c' #(136 / 255, 28 / 255, 15 / 255)
        self.inhomo_color = (157 / 255, 206 / 255, 189 / 255)




    def rgb_fraction_to_hex(self,
                            python_color):
        # Scale the fractions to 0-255 and round to nearest integer
        r, g, b = round(python_color[0] * 255), round(python_color[1] * 255), round(python_color[2] * 255)
        # Convert to hexadecimal and return as a formatted string
        return f"#{r:02x}{g:02x}{b:02x}"

    def get_list_paths(self,
                       paths_dict):
        list_path = []
        for partial_path in paths_dict['fitting_coeff_dir_partial_list_path']:
            path = os.path.join(paths_dict['fitting_coeff_dir_path_beginning'],
                                partial_path)
            list_path.append(path)

        return list_path

    def get_list_all_paths(self):

        paths_dict_list = self.sample_paths_dict_list

        list_path_general = []
        for paths_dict in paths_dict_list:
            list_path = self.get_list_paths(paths_dict)
            list_path_general = list_path_general + list_path

        return list_path_general
    def get_sample_paths_from_measurement_folder(self,
                                                 measurement_folder_path):
        parent_path = os.path.dirname(measurement_folder_path)
        sample_dir = os.path.basename(parent_path)

        if sample_dir == 'AP_4_24':
            return self.AP_4_24_paths
        elif sample_dir == 'AP_3_108':
            return self.AP_3_108_paths
        elif sample_dir == 'AP_3_120':
            return self.AP_3_120_paths
        elif sample_dir == 'AP_4_17':
            return self.AP_4_17_paths
        elif sample_dir == 'HR_90':
            return self.HR_90_paths
        else:
            print('Sample not found in inputparameters.get_sample_paths_from_measurement_folder')
            return None




    def get_label(self,
                  sample_name):
        if sample_name == 'AP_4_24':
            label = self.AP_4_24_plotting_parameters['label']
        elif sample_name == 'AP_4_17':
            label = self.AP_4_17_plotting_parameters['label']
        elif sample_name == 'AP_3_108':
            label = self.AP_3_108_plotting_parameters['label']
        elif sample_name == 'AP_3_120':
            label = self.AP_3_120_plotting_parameters['label']
        return label


if __name__ == '__main__':
    input_parameters = InputParametersClass()

    AP_4_24_hex_color = input_parameters.rgb_fraction_to_hex(input_parameters.AP_4_24_color)
    AP_3_108_hex_color = input_parameters.rgb_fraction_to_hex(input_parameters.AP_3_108_color)
    AP_3_120_hex_color = input_parameters.rgb_fraction_to_hex(input_parameters.AP_3_120_color)
    AP_4_17_hex_color = input_parameters.rgb_fraction_to_hex(input_parameters.AP_4_17_color)
    print(AP_4_24_hex_color)
    print(AP_3_108_hex_color)
    print(AP_3_120_hex_color)
    print(AP_4_17_hex_color)
    measurement_folder_path = '/Users/Shared/Files From c.localized/Gabriel_OMEL/Data_analysis/SingleParticleSpectroscopy/AP_3_120/AndorFail_23-01-26_AP_3_120_MultipleSingleDot'
    sample_folder = input_parameters.get_sample_paths_from_measurement_folder(measurement_folder_path)