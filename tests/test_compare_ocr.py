from compare_ocr import OCR_Analyzer
import random
import numpy as np
import pandas as pd
import pytest
import json


ASSETS_DIR = 'tests/assets/'


@pytest.fixture(scope='function')
def ocr():
    ocr = OCR_Analyzer(ASSETS_DIR+'test_1.json',
                       ASSETS_DIR+'test_2.json',
                       'test_ocr_1',
                       'test_ocr_2')
    return ocr


def test_init(ocr):
    assert isinstance(ocr.ocr_1, pd.DataFrame)
    assert isinstance(ocr.ocr_2, pd.DataFrame)
    assert ocr.ocr_name_1 == 'test_ocr_1'
    assert ocr.ocr_name_2 == 'test_ocr_2'


def test_load_images(ocr):
    ocr.load_images(ASSETS_DIR)
    assert len(ocr.images) == 2
    ocr.images = []
    with pytest.raises(Exception) as e:
        assert ocr.load_images('bad_directory')


def test_scale_bounds(ocr):
    row1 = random.randint(0, len(ocr.ocr_1_plot['bounds'].values)-1)
    row2 = random.randint(0, len(ocr.ocr_2_plot['bounds'].values)-1)
    item1, item2 = random.randint(0, 3), random.randint(0, 3)
    bounds_1a = ocr.ocr_1_plot['bounds'].values[row1][item1]
    bounds_2a = ocr.ocr_2_plot['bounds'].values[row2][item2]
    ocr.scale_bounds(0.5)
    bounds_1b = ocr.ocr_1_plot['bounds'].values[row1][item1]
    bounds_2b = ocr.ocr_2_plot['bounds'].values[row2][item2]
    assert bounds_1b < bounds_1a
    assert bounds_2b < bounds_2a
    ocr.scale_bounds(1.5)
    bounds_1c = ocr.ocr_1_plot['bounds'].values[row1][item1]
    bounds_2c = ocr.ocr_2_plot['bounds'].values[row2][item2]
    assert bounds_1c > bounds_1b
    assert bounds_2c > bounds_2b
    ocr.scale_bounds(1.0)
    bounds_1d = ocr.ocr_1_plot['bounds'].values[row1][item1]
    bounds_2d = ocr.ocr_2_plot['bounds'].values[row2][item2]
    assert bounds_1d == bounds_1c
    assert bounds_2d == bounds_2c
    with pytest.raises(Exception) as e:
        assert ocr.scale_bounds(0)


def test_reverse_scaling(ocr):
    row1 = random.randint(0, len(ocr.ocr_1_plot['bounds'].values)-1)
    row2 = random.randint(0, len(ocr.ocr_2_plot['bounds'].values)-1)
    item1, item2 = random.randint(0, 3), random.randint(0, 3)
    ocr.reverse_scaling()
    assert ocr.ocr_1_plot['bounds'].values[row1][item1] == ocr.ocr_1['bounds'].values[row1][item1]
    assert ocr.ocr_2_plot['bounds'].values[row2][item2] == ocr.ocr_2['bounds'].values[row2][item2]


@pytest.mark.show_plot
def test_show_page(ocr):
    ocr.load_images(ASSETS_DIR)
    assert len(ocr.show_page(1).images) == 1
    assert len(ocr.show_page(2).images) == 1
    with pytest.raises(Exception) as e:
        assert ocr.show_page(3)


@pytest.mark.show_plot
def test_show_boundary_boxes(ocr):
    ocr.load_images(ASSETS_DIR)
    test_ax_1 = ocr.show_boundary_boxes(1, 'sep')
    assert len(test_ax_1.images) == 1
    assert len(test_ax_1.patches) == 2
    assert len(list(test_ax_1.get_legend().get_patches())) == 4
    test_ax_2 = ocr.show_boundary_boxes(1, 'sep', engine='test_ocr_1')
    assert len(test_ax_2.images) == 1
    assert len(test_ax_2.patches) == 1
    with pytest.raises(Exception) as e:
        assert ocr.show_boundary_boxes(3, 'sep')
    with pytest.raises(Exception) as e:
        assert ocr.show_boundary_boxes(1, 'sep', engine='bad_value')


def test_compare_ocr_outputs(ocr):
    output_1 = json.loads(ocr.compare_ocr_outputs(iou_threshold=0.5))
    assert len(output_1) == 2
    assert len(output_1[0].keys()) == 2
    output_2 = json.loads(ocr.compare_ocr_outputs(iou_threshold=0.4, verbose=1))
    assert len(output_2) == 4
    assert len(output_2[0].keys()) == 3
    with pytest.raises(Exception) as e:
        assert ocr.compare_ocr_outputs(verbose=2)
