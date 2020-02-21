#!/usr/bin/env python

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


class OCR_Analyzer:

    def __init__(self, ocr_path_1, ocr_path_2, ocr_name_1, ocr_name_2, ocr_image_directory):
        with open(ocr_path_1) as ocr_file_1:
            self.ocr_1 = pd.DataFrame(json.load(ocr_file_1))
        with open(ocr_path_2) as ocr_file_2:
            self.ocr_2 = pd.DataFrame(json.load(ocr_file_2))
        self.ocr_name_1 = ocr_name_1.lower()
        self.ocr_name_2 = ocr_name_2.lower()
        self.images = [plt.imread(image) for image in glob.glob(ocr_image_directory + '/*.jpg')]
        self.image_scale = 100
        self.scale_trail = []


    def scale_bounds(self, scale=1.0):
        self.ocr_1['bounds'] = self.ocr_1['bounds'].apply(lambda l: [i*scale for i in l])
        self.ocr_2['bounds'] = self.ocr_2['bounds'].apply(lambda l: [i*scale for i in l])
        if scale != 1.0:
            self.scale_trail.append(scale)


    def reverse_scaling(self):
        for scale in reversed(self.scale_trail):
            self.ocr_1['bounds'] = self.ocr_1['bounds'].apply(lambda l: [round(i/scale) for i in l])
            self.ocr_2['bounds'] = self.ocr_2['bounds'].apply(lambda l: [round(i/scale) for i in l])
            self.scale_trail.remove(scale)


    def show_page(self, page):
        try:
            im_data = self.images[page-1]
        except IndexError:
            print('Page does not exist.')
            return 0
        ax = plot_page(im_data, self.image_scale)
        plt.show()
        return ax


    def show_boundary_boxes(self, page, word, table='both', fuzz_threshold=100):
        try:
            im_data = self.images[page-1]
        except IndexError:
            print('Page does not exist.')
            return 0
        tables = [self.ocr_1, self.ocr_2]
        if table == 'both':
            page_data_1 = extract_page(tables[0], page)
            page_data_2 = extract_page(tables[1], page)
            red_boxes = find_boundaries(page_data_1, word, fuzz_threshold)
            green_boxes = find_boundaries(page_data_2, word, fuzz_threshold)
        else:
            table = tables[table-1]
            page_data = extract_page(table, page)
            red_boxes = find_boundaries(page_data, word, fuzz_threshold)
            green_boxes = []

        red_count, green_count = len(red_boxes), len(green_boxes)

        ax = plot_page(im_data, self.image_scale)
        ax = plot_boundary_boxes(ax, red_boxes, green_boxes)
        ax = build_legend(ax, self.ocr_name_1, self.ocr_name_2, red_count, green_count)

        plt.show()

        return ax


    def compare_ocr_outputs(self, iou_threshold=0.4, verbose=0, indent=4):

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
                                       'fuzz_ratio': fuzz.ratio(t_1.text, t_2.text)}

                    output.append(discrepency)

        return json.dumps(output, indent=indent, sort_keys=True)
