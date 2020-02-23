from compare_ocr import OCR_Analyzer
import random
import numpy as np
import pandas as pd
import pytest
import json


ASSETS_DIR = 'tests/assets/'


@pytest.fixture(scope='function')
def ocr():
    """
    Test object initialisation.
    """
    ocr = OCR_Analyzer(ASSETS_DIR+'test_1.json',
                       ASSETS_DIR+'test_2.json',
                       'test_ocr_1',
                       'test_ocr_2')
    return ocr


def test_init(ocr):
    # Test case 1: check object 1 is pandas DataFrame
    assert isinstance(ocr.ocr_1, pd.DataFrame)
    # Test case 2: check object 2 is pandas DataFrame
    assert isinstance(ocr.ocr_2, pd.DataFrame)
    # Test case 3: ckeck name 1 is correctly set
    assert ocr.ocr_name_1 == 'test_ocr_1'
    # Test case 4: check name 2 is correctly set
    assert ocr.ocr_name_2 == 'test_ocr_2'


def test_load_images(ocr):
    ocr.load_images(ASSETS_DIR)
    # Test case 1: check images are loaded correctly
    assert len(ocr.images) == 2
    ocr.images = []
    # Test case 2: assert incorrect image directory raises error
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
    # Test case 1: check downscaling
    assert bounds_1b < bounds_1a
    assert bounds_2b < bounds_2a
    ocr.scale_bounds(1.5)
    bounds_1c = ocr.ocr_1_plot['bounds'].values[row1][item1]
    bounds_2c = ocr.ocr_2_plot['bounds'].values[row2][item2]
    # Test case 2: check upscaling
    assert bounds_1c > bounds_1b
    assert bounds_2c > bounds_2b
    ocr.scale_bounds(1.0)
    bounds_1d = ocr.ocr_1_plot['bounds'].values[row1][item1]
    bounds_2d = ocr.ocr_2_plot['bounds'].values[row2][item2]
    # Test case 3: check samescaling
    assert bounds_1d == bounds_1c
    assert bounds_2d == bounds_2c
    # Test case 4: assert scale value 0 raises error
    with pytest.raises(Exception) as e:
        assert ocr.scale_bounds(0)


def test_reverse_scaling(ocr):
    row1 = random.randint(0, len(ocr.ocr_1_plot['bounds'].values)-1)
    row2 = random.randint(0, len(ocr.ocr_2_plot['bounds'].values)-1)
    item1, item2 = random.randint(0, 3), random.randint(0, 3)
    ocr.reverse_scaling()
    # Test case 1: check reverse scaling returns bounds to original values
    assert ocr.ocr_1_plot['bounds'].values[row1][item1] == ocr.ocr_1['bounds'].values[row1][item1]
    assert ocr.ocr_2_plot['bounds'].values[row2][item2] == ocr.ocr_2['bounds'].values[row2][item2]


@pytest.mark.show_plot
def test_show_page(ocr):
    ocr.load_images(ASSETS_DIR)
    # Test case 1: check 1 image object plotted to axis per page
    assert len(ocr.show_page(1).images) == 1
    # Test case 2: same as above, different page
    assert len(ocr.show_page(2).images) == 1
    # Test case 3: assert page number not present raises error
    with pytest.raises(Exception) as e:
        assert ocr.show_page(3)


@pytest.mark.show_plot
def test_show_boundary_boxes(ocr):
    ocr.load_images(ASSETS_DIR)
    test_ax_1 = ocr.show_boundary_boxes(1, 'sep')
    # Test case 1: check 1 image object plotted to axis per page
    assert len(test_ax_1.images) == 1
    # Test case 2: check 1 patch object plotted for each engine
    assert len(test_ax_1.patches) == 2
    # Test case 3: check 2 legend patch objects plotted for each engine
    assert len(list(test_ax_1.get_legend().get_patches())) == 4
    test_ax_2 = ocr.show_boundary_boxes(1, 'sep', engine='test_ocr_1')
    # Test case 4: same as #2, only 1 engine
    assert len(test_ax_2.patches) == 1
    # Test case 5: same as #3, only 1 engine
    assert len(list(test_ax_2.get_legend().get_patches())) == 2
    # Test case 6: assert page number not present raises error
    with pytest.raises(Exception) as e:
        assert ocr.show_boundary_boxes(3, 'sep')
    # Test case 7: assert bad engine name raises error
    with pytest.raises(Exception) as e:
        assert ocr.show_boundary_boxes(1, 'sep', engine='bad_value')


def test_compare_ocr_outputs(ocr):
    output_1 = json.loads(ocr.compare_ocr_outputs(iou_threshold=0.5))
    # Test case 1: check 2 matches detected in test data when iou_threshold=0.5
    assert len(output_1) == 2
    # Test case 2: assert 2 keys present when verbose=0
    assert len(output_1[0].keys()) == 2
    output_2 = json.loads(ocr.compare_ocr_outputs(iou_threshold=0.4, verbose=1))
    # Test case 3: check 4 matches detected in test data when iou_threshold=0.4
    assert len(output_2) == 4
    # Test case 4: assert 3 keys present when verbose=1
    assert len(output_2[0].keys()) == 3
    # Test case 5: assert bad verbose value raises error
    with pytest.raises(Exception) as e:
        assert ocr.compare_ocr_outputs(verbose=2)
