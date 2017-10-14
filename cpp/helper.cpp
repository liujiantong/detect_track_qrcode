#include "helper.hpp"
#include <opencv2/imgproc.hpp>

#include <numeric>
#include <cmath>


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

cv::Point contour_center(std::vector<cv::Point> contour) {
    cv::Moments m = cv::moments(contour, false);
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
        sum += hist.at<float>(i);
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

direct_pos_t calc_direct(cv::Point head, cv::Point tail) {
    cv::Point pnt0(tail.x, head.y);
    double cos_val = angle_cos(pnt0, head, tail);
    double angle = std::acos(cos_val);
    // std::cout << "cos_val:" << cos_val << ", angle:" << angle << std::endl;
    int degree = (int)((angle / 3.1416) * 180 + 22.5) / 45;
    if (tail.y < pnt0.y) {
        degree *= -1;
    }
    direct_pos_t d = {angle, (direct_t)degree, tail};
    return d;
}

// FIXME:
shape_t detect_shape(cv::Mat& edged) {
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours, found_cnts;
    cv::findContours(edged, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (hierarchy.empty()) {
        return NONE_SHAPE;
    }

    // auto cnt = contours[0];
    std::vector<std::tuple<double, int> > areas;
    for (int i=0; i<contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        areas.push_back(std::make_tuple(area, i));
    }

    std::sort(areas.begin(), areas.end(),
    [](const std::tuple<double, int>& a, const std::tuple<double, int>& b) -> bool {
        return std::get<0>(a) > std::get<0>(b);
    });
    auto cnt = contours[std::get<1>(areas[0])];

    double peri = cv::arcLength(cnt, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(cnt, approx, peri*0.03, true);

    int n_edge = approx.size();
    switch (n_edge) {
        case 3:
            return TRIANGLE;
        case 4:
            return SQUARE;
        case 5:
            return PENTAGON;
        case 6:
            return HEXAGON;
        default:
            return NONE_SHAPE;
    }
}

std::string join(std::vector<std::string>& v) {
    std::ostringstream imploded;
    std::copy(v.begin(), v.end(), std::ostream_iterator<std::string>(imploded, " "));
    return imploded.str();
}

std::string toy_direct_name(direct_t d) {
    switch (d) {
        case EAST_DIR:
            return "EAST";
        case EAST_NORTH_DIR:
            return "EAST_NORTH";
        case NORTH_DIR:
            return "NORTH";
        case WEST_NORTH_DIR:
            return "WEST_NORTH";
        case WEST_DIR:
            return "WEST";
        case WEST_SOUTH_DIR:
            return "WEST_SOUTH";
        case SOUTH_DIR:
            return "SOUTH";
        case EAST_SOUTH_DIR:
            return "EAST_SOUTH";
        default:
            return "UNKNOWN";
    }
}

std::string color_name(color_t c) {
    switch (c) {
        case WHITE:
            return "white";
        case RED:
            return "red";
        case YELLOW:
            return "yellow";
        case GREEN:
            return "green";
        case CYAN:
            return "cyan";
        case BLUE:
            return "blue";
        case MAGENTA:
            return "magenta";
        default:
            return "white";
    }
}

std::vector<std::string> get_color_names(std::vector<color_t>& colors) {
    std::vector<std::string> v;
    v.reserve(colors.size());
    std::transform(colors.begin(), colors.end(),  std::back_inserter(v), [](color_t c) {
        return color_name(c);
    });
    return v;
}
