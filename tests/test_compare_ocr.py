from compare_ocr import OCR_Analyzer
import random
import numpy as np
import pandas as pd
import pytest
from testfixtures import TempDirectory
import tempfile


@pytest.fixture()
def dir():
    with TempDirectory() as dir:
        yield dir


@pytest.fixture(scope='function')
def ocr(dir):
    test = b"""[{"page": 1, "bounds": [154, 65, 236, 107], "text": "Sep"}, {"page": 1, "bounds": [268, 64, 325, 105], "text": "09"}, {"page": 1, "bounds": [355, 63, 472, 103], "text": "2013"}, {"page": 1, "bounds": [502, 63, 551, 103], "text": "3:"}, {"page": 1, "bounds": [559, 63, 671, 105], "text": "34PM"}]"""
    test_json_1 = dir.write('test_1.json', test)
    test_json_2 = dir.write('test_2.json', test)
    ocr = OCR_Analyzer(test_json_1, test_json_2, 'test_ocr_1', 'test_ocr_2', 'tests/assets/')
    return ocr


def test_init(ocr):
    assert isinstance(ocr.ocr_1, pd.DataFrame)
    assert isinstance(ocr.ocr_2, pd.DataFrame)
    assert ocr.ocr_name_1 == 'test_ocr_1'
    assert ocr.ocr_name_2 == 'test_ocr_2'
    assert len(ocr.images) == 2


def test_scale_bounds(ocr):
    row1 = random.randint(0, len(ocr.ocr_1['bounds'].values)-1)
    row2 = random.randint(0, len(ocr.ocr_2['bounds'].values)-1)
    item1 = random.randint(0, 3)
    item2 = random.randint(0, 3)
    assert len(ocr.scale_trail) == 0
    bounds_1a = ocr.ocr_1['bounds'].values[row1][item1]
    bounds_2a = ocr.ocr_2['bounds'].values[row2][item2]
    ocr.scale_bounds(0.5)
    bounds_1b = ocr.ocr_1['bounds'].values[row1][item1]
    bounds_2b = ocr.ocr_2['bounds'].values[row2][item2]
    assert bounds_1b < bounds_1a
    assert bounds_2b < bounds_2a
    assert 0.5 in ocr.scale_trail
    ocr.scale_bounds(1.5)
    bounds_1c = ocr.ocr_1['bounds'].values[row1][item1]
    bounds_2c = ocr.ocr_2['bounds'].values[row2][item2]
    assert bounds_1c > bounds_1b
    assert bounds_2c > bounds_2b
    assert 1.5 in ocr.scale_trail
    ocr.scale_bounds(1)
    bounds_1d = ocr.ocr_1['bounds'].values[row1][item1]
    bounds_2d = ocr.ocr_2['bounds'].values[row2][item2]
    assert bounds_1d == bounds_1d
    assert bounds_2d == bounds_2c
    assert 1 not in ocr.scale_trail
    assert len(ocr.scale_trail) == 2


def test_reverse_scaling(ocr):
    pass


def test_show_page(ocr):
    pass


def test_show_boundary_boxes(ocr):
    pass


def test_compare_ocr_outputs(ocr):
    pass
