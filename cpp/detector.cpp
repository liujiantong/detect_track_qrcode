#include "detector.hpp"
#include "helper.hpp"
#include "spdlog/spdlog.h"

#include <vector>
#include <algorithm>

namespace spd = spdlog;


const cv::Range RED_RANGE1(0, 30);
const cv::Range RED_RANGE2(150, 180);
const cv::Range GREEN_RANGE(30, 90);
const cv::Range BLUE_RANGE(90, 140);


std::vector<std::vector<cv::Point> > ToyDetector::find_code_contours(cv::Mat& gray) {
    cv::Mat blurred, edges;
    cv::medianBlur(gray, blurred, 3);
    cv::Canny(blurred, edges, 100, 120);

    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours, found_cnts;
    cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (hierarchy.empty()) {
        return found_cnts;
    }

    // cv::Vec4i hierarchy0 = hierarchy[0];
    for (int cnt_idx=0; cnt_idx<contours.size(); cnt_idx++) {
        double area = cv::contourArea(contours[cnt_idx], true);
        if (area < 100) {
            continue;
        }

        int k = cnt_idx, c = 0;
        while (hierarchy[k][2] != -1) {
            k = hierarchy[k][2];
            c++;
        }
        if (c > 0) {
            found_cnts.push_back(contours[cnt_idx]);
        }
    }

    return found_cnts;
}


std::vector<std::string> ToyDetector::detect_color_from_contours(cv::Mat& img,
    std::vector<std::vector<cv::Point> >& cnts,
    std::vector<cv::Point>& out_cnt) {

    auto logger = spd::get("console");
    std::vector<std::vector<cv::Point>> square_cnts;
    for (auto cnt0 : cnts) {
        std::tuple<bool, std::vector<cv::Point> > result = detect_square(cnt0);
        if (std::get<0>(result)) {
            square_cnts.push_back(std::get<1>(result));
        }
    }
    logger->debug("square_cnts.size:{}", square_cnts.size());

    if (square_cnts.empty()) {
        std::vector<std::string> colors;
        return colors;
    }

    cv::Mat out_dst;
    std::vector<cv::Point> bound_cnt = check_cnt_contain(square_cnts);
    std::vector<std::string> colors = detect_color_in(img, bound_cnt, out_dst);
    // fill out_cnt
    out_cnt.assign(bound_cnt.begin(), bound_cnt.end());
    logger->debug("colors.size:{}", colors.size());

    return colors;
}


std::vector<std::string> ToyDetector::detect_color_in(cv::Mat& img, std::vector<cv::Point>& cnt, cv::Mat& out_dst) {
    auto logger = spd::get("console");

    std::vector<std::string> colors;
    cv::Point2f src_pnts[3], square_pnts[3];

    src_pnts[0] = cv::Point2f(cnt[0]);
    src_pnts[1] = cv::Point2f(cnt[1]);
    src_pnts[2] = cv::Point2f(cnt[2]);
    square_pnts[0] = cv::Point2f(0.0f, 0.0f );
    square_pnts[1] = cv::Point2f(_block_size, 0.0f );
    square_pnts[2] = cv::Point2f(_block_size, _block_size);

    // cv::Rect r = cv::boundingRect(cnt);
    // if (r.width < _block_size || r.height < _block_size) {
    //     return colors;
    // }

    // out_dst.create(r.size(), img.type());
    out_dst.create(_block_size, _block_size, img.type());

    cv::Mat mtx = cv::getAffineTransform(src_pnts, square_pnts);
    // cv::warpAffine(img, out_dst, mtx, cv::Size(r.width, r.height));
    cv::warpAffine(img, out_dst, mtx, cv::Size(_block_size, _block_size));
    int half_block_size = _block_size / 2;

    cv::Mat roi1 = out_dst(cv::Rect(0, 0, half_block_size, half_block_size));
    cv::Mat roi2 = out_dst(cv::Rect(half_block_size, 0, half_block_size, half_block_size));
    cv::Mat roi3 = out_dst(cv::Rect(half_block_size, half_block_size, half_block_size, half_block_size));
    cv::Mat roi4 = out_dst(cv::Rect(0, half_block_size, half_block_size, half_block_size));

    /* FIXME: for debug
    cv::imwrite("roi1.png", roi1);
    cv::imwrite("roi2.png", roi2);
    cv::imwrite("roi3.png", roi3);
    cv::imwrite("roi4.png", roi4);
    */

    colors.push_back(detect_color(roi1));
    colors.push_back(detect_color(roi2));
    colors.push_back(detect_color(roi3));
    colors.push_back(detect_color(roi4));

    logger->debug("colors.size:{}", colors.size());

    int color_idx = -1;
    for (int i=0; i<colors.size(); i++) {
        if (colors[i] == "white") {
            color_idx = i;
            break;
        }
    }

    if (color_idx != -1) {
        std::rotate(colors.begin(), colors.begin() + color_idx, colors.end());
    }
    //return std::make_tuple(colors, dst);
    return colors;
}


std::tuple<bool, std::vector<cv::Point> > ToyDetector::detect_square(std::vector<cv::Point>& cnt) {
    double peri = cv::arcLength(cnt, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(cnt, approx, peri*0.02, true);

    if (approx.size() == 4 && cv::isContourConvex(approx)) {
        std::vector<double> coss;
        std::vector<double> distances;

        for (int i=0; i<4; i++) {
            coss.push_back( angle_cos(approx[i], approx[(i+1) % 4], approx[(i+2) % 4]) );
            distances.push_back(calc_distance(approx[i], approx[(i+1) % 4]));
        }

        double max_cos = *std::max_element(coss.begin(), coss.end());
        std::tuple<double, double> mean_std = calc_mean_stdev(distances);

        double z_val = std::get<1>(mean_std);
        if (std::abs(std::get<0>(mean_std) - 0.0) > 0.01) {
            z_val = std::get<1>(mean_std) / std::get<0>(mean_std);
        }

        if (z_val < 0.18 && max_cos < 0.35) {
            return std::make_tuple(true, approx);
        }
    }

    return std::make_tuple(false, approx);
}


std::string ToyDetector::detect_color(cv::Mat& roi) {
    auto logger = spd::get("console");

    cv::Mat hsv, mask, white_mask, hist;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(hsv, cv::Scalar(0, 20, 0), cv::Scalar(180, 255, 255), mask);
    cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 60, 255), white_mask);

    int w = white_mask.cols;
    int h = white_mask.rows;
    int cnz = cv::countNonZero(white_mask);

    float wr = cnz / ((float)(h * w));
    if (wr > 0.8) {
        return "white";
    }

    // int hsize = 30;
    int hsize = 180;
    int channels[] = {0};
    float hue_range[] = { 0.0f, 180.0f };
    const float* ranges[] = { hue_range };

    cv::calcHist(&hsv, 1, channels, mask, hist, 1, &hsize, ranges, true);
    cv::normalize(hist, hist, 0.0, 1.0, cv::NORM_MINMAX);

    double rval = sum_histogram(hist, RED_RANGE1) + sum_histogram(hist, RED_RANGE2);
    double gval = sum_histogram(hist, GREEN_RANGE);
    double bval = sum_histogram(hist, BLUE_RANGE);

    // find max value
    std::string colors[] = {"red", "green", "blue"};
    double max_val = std::max({rval, gval, bval});

    int pos = -1;
    double vals[] = {rval, gval, bval};
    for (int i=0; i<3; i++) {
        // logger->debug("{}:{}", colors[i], vals[i]);
        if (vals[i] == max_val) {
            pos = i;
        }
    }

    if (max_val > 0.8) {
        return colors[pos];
    }
    return "unknown";
}

std::vector<cv::Point> ToyDetector::check_cnt_contain(std::vector<std::vector<cv::Point> >& cnts) {
    if (cnts.empty()) {
        std::vector<cv::Point> result;
        return result;
    }

    std::vector<std::tuple<double, int> > areas;
    for (int i=0; i<cnts.size(); i++) {
        double area = cv::contourArea(cnts[i]);
        areas.push_back(std::make_tuple(area, i));
    }

    std::sort(areas.begin(), areas.end(),
    [](const std::tuple<double, int>& a, const std::tuple<double, int>& b) -> bool {
        return std::get<0>(a) > std::get<0>(b);
    });

    int counter = 0;
    std::vector<std::tuple<double, int> >::iterator pos;
    for (pos=areas.begin(); pos<areas.end()-1; ++pos) {
        counter++;
        int idx = std::get<1>(*pos);
        std::vector<cv::Point> cnt = cnts[idx];

        std::vector<double> flags;
        std::vector<std::tuple<double, int> >::iterator pos1;
        for (pos1=pos+counter; pos1<areas.end(); ++pos1) {
            int idx1 = std::get<1>(*pos1);
            std::vector<cv::Point> cnt1 = cnts[idx1];
            for (cv::Point& pt : cnt1) {
                double flag = cv::pointPolygonTest(cnt, pt, false);
                flags.push_back(flag);
            }
        }

        if (std::all_of(flags.begin(), flags.end(), [](double f) -> bool {return f >= 0;})) {
            return cnt;
        }
    }

    int idx = std::get<1>(areas[0]);
    return cnts[idx];
}
