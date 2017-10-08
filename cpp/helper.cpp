#include "helper.hpp"
#include <opencv2/imgproc.hpp>

#include <numeric>


cv::Size get_frame_size(cv::Size size, unsigned max_width) {
    if (size.width < max_width) {
        return size;
    }
    double ratio = double(max_width) / size.width;
    return cv::Size(max_width, (ratio * size.height));
}

cv::Rect union_rects(std::vector<cv::Rect>& rects) {
    if (rects.empty()) {
        cv::Rect r = cv::Rect();
        return r;
    }

    std::vector<cv::Point> pnts;
    for (int i=0; i<rects.size(); i++) {
        pnts.push_back(cv::Point(rects[i].x, rects[i].y));
        pnts.push_back(cv::Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height));
    }
    return cv::boundingRect(pnts);
}

cv::Point pnts_center(std::vector<cv::Point>& points) {
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

double angle_cos(cv::Point pt0, cv::Point pt1, cv::Point pt2) {
    double dx1 = pt0.x - pt1.x;
    double dy1 = pt0.y - pt1.y;
    double dx2 = pt2.x - pt1.x;
    double dy2 = pt2.y - pt1.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
};

double sum_histogram(cv::Mat& hist, const cv::Range& range) {
    double sum = 0;
    for (int i=range.start; i<range.end; i++) {
        sum += cvRound(hist.at<float>(i));
    }
    return sum;
}

std::tuple<double, double> calc_mean_stdev(const std::vector<double> v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size());

    return std::make_tuple(mean, stdev);
}
