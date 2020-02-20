import helpers
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def test_df():
    data = {'page': [1,1,2,2],
            'bounds': [[1,2,3,4], [5,6,7,8], [9,1,2,3], [4,5,6,7]],
            'text': ['testing', 'one', 'two', 'testing']}
    df = pd.DataFrame(data)
    return df


def test_iou():
    # Test case 1: boxes intersect
    assert helpers.iou((2,1,4,3), (1,2,3,4)) == 0.14285714285714285
    # Test case 2: boxes do not intersect
    assert helpers.iou((1,2,3,4), (5,6,7,8)) == 0.0
    # Test case 3: boxes intersect at vertices only
    assert helpers.iou((1,1,2,2), (2,2,3,3)) == 0.0
    # Test case 4: boxes intersect at edge only
    assert helpers.iou((1,1,3,3), (2,3,3,4)) == 0.0


def test_find_boundaries(test_df):
    # Test case 1: one match, strict fuzz threshold
    assert len(helpers.find_boundaries(test_df, 'one', 100)) == 1
    # Test case 2: two matches, strict fuzz threshold
    assert len(helpers.find_boundaries(test_df, 'testing', 100)) == 2
    # Test case 3: no matches, strict fuzz threshold
    assert len(helpers.find_boundaries(test_df, 'testin', 100)) == 0
    # Test case 4: one match, strict fuzz threshold, show boundaries
    assert helpers.find_boundaries(test_df, 'two', 100) == [[9,1,2,3]]
    # Test case 5: two matches, strict fuzz threshold, show boundaries
    assert helpers.find_boundaries(test_df, 'testing', 100) == [[1,2,3,4], [4,5,6,7]]
    # Test case 6: no matches, strict fuzz threshold, show boundaries
    assert helpers.find_boundaries(test_df, 'testin', 100) == []
    # Test case 7: two matches, relaxed fuzz threshold, show boundaries
    assert helpers.find_boundaries(test_df, 'testin', 90) == [[1,2,3,4], [4,5,6,7]]
    # Test case 8: no matches, strict fuzz threshold, show boundaries
    assert helpers.find_boundaries(test_df, 'tow', 100) == []
    # Test case 9: one match, very relaxed fuzz threshold, show boundaries
    assert helpers.find_boundaries(test_df, 'tow', 60) == [[9,1,2,3]]


def test_extract_page(test_df):
    # Test case 1: check only one page value exists after extraction
    assert helpers.extract_page(test_df, 1)['page'].nunique() == 1
    # Test case 2: repeat of above, different page
    assert helpers.extract_page(test_df, 2)['page'].nunique() == 1
    # Test case 3: check correct output size after extraction
    assert helpers.extract_page(test_df, 1).shape == (2, 3)
    # Test case 4: repeat of above, different page
    assert helpers.extract_page(test_df, 2).shape == (2, 3)
    # Test case 5: check no rows in output if incorrect page value
    assert helpers.extract_page(test_df, 3).shape == (0, 3)
    # Test case 6: check page value correct after extraction
    assert helpers.extract_page(test_df, 1)['page'].values[0] == 1
    # Test case 7: repeat of above, different page
    assert helpers.extract_page(test_df, 2)['page'].values[0] == 2


def test_plot_page():
    pass


def test_plot_boundary_boxes():
    pass
