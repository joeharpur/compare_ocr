"""
Utility functions for compare_ocr.py
"""

import pandas as pd
import itertools
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 (List) -- first box with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 (List) -- second box with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)

    Returns:
    iou (Float) -- intersection over union value for box1, box2
    """

    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    # Calculate the coordinates and area of the intersection of box1 and box2.
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(yi2 - yi1, 0)
    inter_height = max(xi2 - xi1, 0)
    inter_area = inter_width * inter_height
    # Calculate the Union area by using Formula: Union(A,B)=A+B-Inter(A,B)
    box1_area = (box1_y2 - box1_y1) * (box1_x2 - box1_x1)
    box2_area = (box2_y2 - box2_y1) * (box2_x2 - box2_x1)
    union_area = (box1_area + box2_area) - inter_area
    # compute the IoU
    iou = inter_area / union_area

    return iou


def find_boundaries(table, word, fuzz_threshold):
    """
    Find the boundary boxes for a specified word.
    String match strictness can be tuned using fuzz_threshold.
    
    Arguments:
    table (pd.DataFrame) -- table containing text field and bounds field
    word (Str) -- search word
    fuzz_threshold (Int) -- accepted closeness between string values

    Returns:
    (List) -- list of matching boundaries
    """

    filter = lambda x: fuzz.ratio(x.lower(), word) >= fuzz_threshold
    boundaries = table[table['text'].apply(filter)]['bounds'].values

    return list(boundaries)


def extract_page(table, page):
    """
    Filter table by page number field.
        
    Arguments:
    table (pd.DataFrame) -- table containing "page" field
    page (Int) -- page value

    Returns:
    extracted (pd.DataFrame) -- filtered table
    """

    extracted = table[table['page'] == page]

    return extracted


def plot_page(im_data, scale):
    """
    Plot image data to Matplotlib axes.
        
    Arguments:
    im_data (np.array) -- array containing image data
    scale (Int) -- scale to convert image dpi to inches

    Returns:
    ax (matplotlib.axes) -- axes with image data plotted
    """

    height, width = im_data.shape[:2]
    figsize = height/scale, width/scale
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')

    return ax


def plot_boundary_boxes(ax, red_boxes, green_boxes):
    """
    Plot boundary boxes to Matplotlib axes.
        
    Arguments:
    ax (matplotlib.axes) -- pre generated axes
    red_boxes (List) -- list of boundary coordinates
    green_boxes (List) -- list of boundary coordinates

    Returns:
    ax (matplotlib.axes) -- axes with boundary boxes plotted
    """

    for r, g in itertools.zip_longest(red_boxes, green_boxes):
        if r:
            r_x, r_y, r_w, r_h = r[0], r[1], r[2]-r[0], r[3]-r[1]
            r_rect = patches.Rectangle((r_x, r_y), r_w, r_h,
                                       linewidth=2,
                                       edgecolor='r',
                                       facecolor='none')
            ax.add_patch(r_rect)

        if g:
            g_x, g_y, g_w, g_h = g[0], g[1], g[2]-g[0], g[3]-g[1]
            g_rect = patches.Rectangle((g_x, g_y), g_w, g_h,
                                       linewidth=2,
                                       edgecolor='g',
                                       facecolor='none')
            ax.add_patch(g_rect)
    return ax


def build_legend(ax, names, red_count, green_count):
    """
    Plot boundary boxes to Matplotlib axes.
        
    Arguments:
    ax (matplotlib.axes) -- pre generated axes
    names (List) -- list of ocr engine names
    red_count (Int) -- count of red boundary boxes
    green_count (Int) -- count of green boundary boxes

    Returns:
    ax (matplotlib.axes) -- axes with legend plotted
    """

    handles = []
    red_patch = patches.Patch(linewidth=2,
                              edgecolor='r',
                              facecolor='none',
                              label=names[0].capitalize())
    handles.append(red_patch)
    info_1 = patches.Patch(edgecolor='none',
                           facecolor='none',
                           label=str(red_count) + ' matches')
    handles.append(info_1)
    if green_count > 0:
        green_patch = patches.Patch(linewidth=2,
                                    edgecolor='g',
                                    facecolor='none',
                                    label=names[1].capitalize())
        handles.append(green_patch)
        info_2 = patches.Patch(edgecolor='none',
                               facecolor='none',
                               label=str(green_count) + ' matches')
        handles.append(info_2)
    ax.legend(handles=handles,
              loc='best',
              framealpha=0.5)

    return ax


if __name__ == '__main__':
    pass
