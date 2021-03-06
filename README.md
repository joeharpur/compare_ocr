[![Build Status](https://travis-ci.org/joeharpur/compare_ocr.svg?branch=master)](https://travis-ci.org/joeharpur/compare_ocr)

# compare_ocr
Compare the JSON outputs of 2 different [optical character recognition](https://en.wikipedia.org/wiki/Optical_character_recognition) engines.

Based on the boundary box outputs and text classification outputs of
2 OCR engines, outputs a JSON summary of the words on which the engines
disagree. For further analysis, can plot boundary boxes of specific words
over source images.

Note. Source images must be .jpg format with images of multiple pages appearing in correct order in image directory.

Uses [Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index) algorithm to calculate similarity between
boundary boxes.

Strictness of string matching for boundary box plotting is calculated using
[Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) between strings.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Initialization
```python
from compare_ocr import OCR_Analyzer

ocr = OCR_Analyzer('json_path_1', 'json_path_2', 'engine_name_1', 'engine_name_2')
```

Loading images
```python
ocr.load_images('image_directory_path')
```

Scaling boundary box values
```python
ocr.scale_bounds(0.5)
```

Return boundary boxes to original values
```python
ocr.reverse_scaling()
```

Show loaded image
```python
ocr.show_page(page_num)
```

Show boundary boxes over loaded image
```python
ocr.show_boundary_boxes(page_num, search_word, fuzz_threshold)
```

Compare OCR engine outputs
```python
ocr.compare_ocr_outputs(iou_threshold=t, verbose=v, indent=i)
```

## Testing

Run all tests
```python
coverage run -m pytest tests/
```

Show coverage report
```python
coverage report -m
```

Skip tests requiring matplotlibs plt.show() (generates plot window)
```python
python -m pytest tests/ -m "not show_plot"
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
