"""
Core class for compare_ocr.py
"""

from helpers import *
import json
import glob
from fuzzywuzzy import fuzz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import functools


def catch_exception(f):
    """
    Decorator for exception handling.
    """
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in', f.__name__)
            print(str(e))
    return func


class OCR_Analyzer:
    """
    Compare the JSON outputs of 2 different OCR engines.

    Based on the boundary box outputs and text classification outputs of
    2 OCR engines, outputs a JSON summary of the words on which the engines
    disagree. For further analysis, can plot boundary boxes of specific words
    over source images.

    Uses 'Intersection over Union' algorithm to calculate similarity between
    boundary boxes.

    Strictness of string matching for boundary box plotting is calculated using
    levenshtein distance between strings.

        
    Arguments:
    ocr_path_1 (String) -- path to JSON file 1
    ocr_path_2 (String) -- path to JSON file 2
    ocr_name_1 (String) -- name of ocr engine 1
    ocr_path_2 (String) -- name of ocr engine 2
    """

    @catch_exception
    def __init__(self, ocr_path_1, ocr_path_2, ocr_name_1, ocr_name_2):
        with open(ocr_path_1) as ocr_file_1:
            self.ocr_1 = pd.DataFrame(json.load(ocr_file_1))
        with open(ocr_path_2) as ocr_file_2:
            self.ocr_2 = pd.DataFrame(json.load(ocr_file_2))
        self.ocr_1_plot = self.ocr_1.copy()
        self.ocr_2_plot = self.ocr_2.copy()
        self.ocr_name_1 = ocr_name_1.lower()
        self.ocr_name_2 = ocr_name_2.lower()
        self.image_scale = 100
        self.scale_trail = []


    @catch_exception
    def load_images(self, directory):
        """
        Load source images.

        Images must be in .jpg format.
            
        Arguments:
        directory (String) -- path to image directory
        """

        self.images = [plt.imread(image) for image in glob.glob(directory + '/*.jpg')]
        if len(self.images) == 0:
            raise IOError('No images found.')


    @catch_exception
    def scale_bounds(self, scale=1.0):
        """
        Scale boundary box values.
            
        Arguments:
        scale (Float) -- value to scale boundary boxes by.
        """

        if scale == 0:
            raise ValueError('Scale value can not be zero.')
        scale_func = lambda l: [round(i*scale) for i in l]
        self.ocr_1_plot['bounds'] = self.ocr_1_plot['bounds'].apply(scale_func)
        self.ocr_2_plot['bounds'] = self.ocr_2_plot['bounds'].apply(scale_func)


    @catch_exception
    def reverse_scaling(self):
        """
        Return boundary boxes to original values.
        """

        self.ocr_1_plot['bounds'] = self.ocr_1['bounds']
        self.ocr_2_plot['bounds'] = self.ocr_2['bounds']


    @catch_exception
    def show_page(self, page):
        """
        Display source image.
            
        Arguments:
        page (Int) -- page to show

        Returns:
        ax (matplotlib.axes) -- axes with image plotted
        """

        if page > len(self.images):
            raise IndexError('Page does not exist.')
        im_data = self.images[page-1]
        ax = plot_page(im_data, self.image_scale)
        plt.show()

        return ax


    @catch_exception
    def show_boundary_boxes(self, page, word, engine='both', fuzz_threshold=100):
        """
        Show boundary boxes over source image.

        Strictness of string matching for boundary box plotting is calculated using
        levenshtein distance between strings.
            
        Arguments:
        page (Int) -- page to show
        word (String) -- word to show
        engine (String) -- engine to use, default is 'both'
        fuzz_treshold (Int) -- strictness of string matching. default is 100

        Returns:
        ax (matplotlib.axes) -- axes with image, boundary boxes, and legend plotted
        """

        if page > len(self.images):
            raise IndexError('Page does not exist.')
        im_data = self.images[page-1]
        tables = [self.ocr_1_plot, self.ocr_2_plot]
        t_names = [self.ocr_name_1, self.ocr_name_2]
        if engine == 'both':
            page_data_1 = extract_page(tables[0], page)
            page_data_2 = extract_page(tables[1], page)
            red_boxes = find_boundaries(page_data_1,word, fuzz_threshold)
            green_boxes = find_boundaries(page_data_2, word, fuzz_threshold)
        else:
            if engine not in t_names:
                raise ValueError('OCR data not detected.')
            table = tables[t_names.index(engine)]
            page_data = extract_page(table, page)
            red_boxes = find_boundaries(page_data, word, fuzz_threshold)
            t_names = [engine]
            green_boxes = []

        red_count, green_count = len(red_boxes), len(green_boxes)

        ax = plot_page(im_data, self.image_scale)
        ax = plot_boundary_boxes(ax, red_boxes, green_boxes)
        ax = build_legend(ax,
                          t_names,
                          red_count,
                          green_count)

        plt.show()

        return ax


    @catch_exception
    def compare_ocr_outputs(self, iou_threshold=0.4, verbose=0, indent=4):
        """
        Compare the outputs of both OCR engines.

        Uses 'Intersection over Union' algorithm to calculate similarity between
        boundary boxes.
            
        Arguments:
        iou_threshold (Float) -- threshold for overlapping boxes, default is 0.4
        verbose (Int) -- level of verbosity in output, default is 0
        indent (Int) -- level of indentation in JSON output

        Returns:
        (JSON) -- JSON object showing comparison between both OCR engines.
        """

        if verbose not in [0, 1]:
            raise ValueError('Verbose level must be 0 or 1.')

        output = []

        for t_1 in self.ocr_1.itertuples():

            for t_2 in self.ocr_2.itertuples():

                iou_score = iou(t_1.bounds, t_2.bounds)

                if t_1.page == t_2.page and iou_score >= iou_threshold and t_1.text != t_2.text:
                    discrepency = {self.ocr_name_1: {'page': t_1.page,
                                                     'bounds': t_1.bounds,
                                                     'text': t_1.text},
                                   self.ocr_name_2: {'page': t_2.page,
                                                     'bounds': t_2.bounds,
                                                     'text': t_2.text}}
                    if verbose == 1:
                        discrepency = {'ocr_output': discrepency,
                                       'iou_of_bounds': iou_score,
                                       'fuzz_ratio': fuzz.ratio(t_1.text,
                                                                t_2.text)}

                    output.append(discrepency)

        return json.dumps(output, indent=indent, sort_keys=True)


if __name__ == '__main__':
    pass
