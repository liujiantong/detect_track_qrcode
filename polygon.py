# coding: utf-8


import numpy as np

from shapely.geometry import Polygon
from shapely.strtree import STRtree


def check_contain(rects):
    if not rects:
        return None

    polygons = [Polygon(np.int32(r)) for r in rects]
    tree = STRtree(polygons)

    for rect in rects:
        query_rect = Polygon(np.int32(rect)).buffer(1.0)
        result = tree.query(query_rect)
        if len(result) == len(polygons):
            return rect

    areas = [p.area for p in polygons]
    return rects[np.argmax(areas)]

