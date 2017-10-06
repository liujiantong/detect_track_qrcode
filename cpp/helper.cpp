#include "helper.hpp"

cv::Size get_frame_size(cv::Size size, unsigned max_width) {
    if (size.width < max_width) {
        return size;
    }
    double ratio = max_width / size.width;
    return cv::Size(max_width, (ratio * size.height));
}

cv::Rect union_rects(std::vector<cv::Rect>& rects) {
    std::vector<cv::Point> pnts;
    for (int i=0; i<rects.size(); i++) {
        pnts.push_back(cv::Point(rects[i].x, rects[i].y));
        pnts.push_back(cv::Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height));
    }
    return cv::boundingRect(pnts);
}

cv::Point center(std::vector<cv::Point>& points) {
    double cx = (points[0].x + points[1].x + points[2].x + points[3].x) / 4.0;
    double cy = (points[0].y + points[1].y + points[2].y + points[3].y) / 4.0;
    return cv::Point(cx, cy);
}

cv::Point contour_center(std::vector<cv::Point> contour, bool bin_img) {
    cv::Moments m = cv::moments(contour, bin_img);
    return cv::Point((m.m10 / m.m00), (m.m01 / m.m00));
}

double calc_distance(const cv::Point pt1, const cv::Point pt2) {
    return std::sqrt(std::pow((pt1.x - pt2.x), 2) + std::pow((pt1.y - pt2.y), 2));
}
