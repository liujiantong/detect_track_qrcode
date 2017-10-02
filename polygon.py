# coding: utf-8


import numpy as np

from shapely.geometry import Polygon
from shapely.strtree import STRtree


def check_cnt_contain(cnts):
    if not cnts:
        return None

    polygons = [Polygon(np.int32(r)) for r in cnts]
    tree = STRtree(polygons)

    for cnt in cnts:
        query_rect = Polygon(np.int32(cnt)).buffer(1.0)
        result = tree.query(query_rect)
        if len(result) == len(polygons):
            return cnt

    areas = [p.area for p in polygons]
    return cnts[np.argmax(areas)]

