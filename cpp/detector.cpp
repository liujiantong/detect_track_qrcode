#include "detector.hpp"

#include "RTree.hpp"


typedef int ValueType;
typedef RTree<ValueType, int, 2, float> STRtree;


std::vector<cv::Point> ToyDetector::check_cnt_contain(std::vector<std::vector<cv::Point> >& cnts) {
    // if not cnts:
    //     return None
    //
    // polygons = [Polygon(np.int32(r)) for r in cnts]
    // tree = STRtree(polygons)
    //
    // for cnt in cnts:
    //     query_rect = Polygon(np.int32(cnt)).buffer(1.0)
    //     result = tree.query(query_rect)
    //     if len(result) == len(polygons):
    //         return cnt
    //
    // areas = [p.area for p in polygons]
    // return cnts[np.argmax(areas)]

    std::vector<cv::Point> result;
    STRtree tree;

    for (int i=0; i<cnts.size(); i++) {
        // tree.Insert(rects[i].min, rects[i].max, i);
    }

    return result;
}
