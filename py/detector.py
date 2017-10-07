#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np

from shapely.geometry import Polygon
from shapely.strtree import STRtree
from scipy.spatial import distance

import helper


red_range1 = (0, 30)
red_range2 = (150, 180)
green_range = (30, 90)
blue_range = (90, 140)


class ToyDetector(object):

    def __init__(self, block_size=100):
        self._block_size = block_size

    @staticmethod
    def find_contours(gray):
        blurred = cv2.medianBlur(gray, 3)
        edges = cv2.Canny(blurred, 100, 120)

        found_cnts = []
        _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or hierarchy.size == 0:
            return found_cnts

        hierarchy = hierarchy[0]
        # print 'hierarchy.len:', len(hierarchy)
        # print 'hierarchy:', hierarchy

        for cnt_idx in range(len(contours)):
            area = cv2.contourArea(contours[cnt_idx], oriented=True)
            if area < 100:
                continue

            k = cnt_idx
            c = 0
            while hierarchy[k][2] != -1:
                k = hierarchy[k][2]
                c = c + 1
            if c > 0:
                found_cnts.append(contours[cnt_idx])

        return found_cnts

    @staticmethod
    def _detect_color(roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (0., 20., 0.), (180., 255., 255.))
        white_mask = cv2.inRange(hsv, (0., 0., 200.), (180., 60., 255.))

        h, w = white_mask.shape[:2]
        cnz = cv2.countNonZero(white_mask)
        wr = np.float32(cnz) / (h * w)
        # print (h, w), 'cnz:', cnz, 'white rate:', wr
        if wr > 0.80:
            return 'white'

        hst = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        cv2.normalize(hst, hst, 0, 1, cv2.NORM_MINMAX)

        rval = np.sum(hst[red_range1[0]:red_range1[1]]) + np.sum(hst[red_range2[0]:red_range1[1]])
        gval = np.sum(hst[green_range[0]:green_range[1]])
        bval = np.sum(hst[blue_range[0]:blue_range[1]])

        if rval > 0.8:
            return 'red'
        elif gval > 0.8:
            return 'green'
        elif bval > 0.8:
            return 'blue'
        return 'unknown'

    def detect_color_in(self, img, cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        cnt = cnt.reshape(-1, 2)
        square_pnts = np.float32([[0, 0], [self._block_size, 0], [self._block_size, self._block_size]])
        mtx = cv2.getAffineTransform(np.float32(cnt[:3]), square_pnts)
        dst = cv2.warpAffine(img, mtx, (w, h))
        # dst = cv2.warpAffine(img, mtx, (self._block_size, self._block_size))

        half_block_size = self._block_size / 2
        roi1, roi2, roi3, roi4 = dst[0:half_block_size, 0:half_block_size], \
                                 dst[half_block_size:self._block_size, 0:half_block_size], \
                                 dst[half_block_size:self._block_size, half_block_size:self._block_size], \
                                 dst[0:half_block_size, half_block_size:self._block_size]
        colours = self._detect_color(roi1), self._detect_color(roi2), \
                  self._detect_color(roi3), self._detect_color(roi4)
        for idx, color in enumerate(colours):
            if color == 'white':
                colours = colours[idx:] + colours[:idx]
        return colours, dst

    def detect_color_from_contours(self, img, cnts):
        square_cnts = []
        for cnt0 in cnts:
            is_square, c = self.detect_square(cnt0)
            if is_square:
                square_cnts.append(c)

        if not square_cnts:
            return [], None

        bound_cnt = self._check_cnt_contain(square_cnts)
        colors, dst = self.detect_color_in(img, bound_cnt)
        return colors, bound_cnt

    def detect_square(self, cnt):
        peri = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(cnt) == 4 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            ws = [distance.euclidean(cnt[i], cnt[(i + 1) % 4]) for i in xrange(4)]
            max_cos = np.max([helper.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
            z_val = np.std(ws) / np.mean(ws)
            # print 'z_val:%s, max_cos:%s' % (z_val, max_cos)
            if z_val < 0.18 and max_cos < 0.35:
                return True, cnt
        return False, None

    @staticmethod
    def _check_cnt_contain_rtree(cnts):
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

    @staticmethod
    def _check_cnt_contain(cnts):
        if not cnts:
            return None

        areas = [(cv2.contourArea(c), c) for c in cnts]
        sorted_cnts = sorted(areas, key=lambda tup: tup[0], reverse=True)
        for idx, itm in enumerate(sorted_cnts):
            _, cnt = itm
            for _, c in sorted_cnts[idx+1:]:
                flags = [cv2.pointPolygonTest(cnt, (pt[0], pt[1]), measureDist=False) for pt in c]
                if all(f >= 0 for f in flags):
                    return cnt

        return sorted_cnts[0][1]
