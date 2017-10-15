#include "toy.hpp"
#include "detector.hpp"
#include "helper.hpp"
#include "spdlog/spdlog.h"

#include <vector>
#include <algorithm>
#include <opencv2/calib3d.hpp>


namespace spd = spdlog;


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


toy_code_t ToyDetector::detect_code_from_contours(cv::Mat& img,
    std::vector<std::vector<cv::Point> >& cnts,
    std::vector<cv::Point>& out_cnt) {

    auto logger = spd::get("toy");
    std::vector<std::vector<cv::Point>> square_cnts;
    for (auto cnt0 : cnts) {
        std::tuple<bool, std::vector<cv::Point> > result = detect_square(cnt0);
        if (std::get<0>(result)) {
            square_cnts.push_back(std::get<1>(result));
        }
    }
    logger->debug("square_cnts.size:{}", square_cnts.size());

    if (square_cnts.empty()) {
        toy_code_t toycode = {};
        return toycode;
    }

    cv::Mat out_dst;
    std::vector<cv::Point> bound_cnt = check_cnt_contain(square_cnts);
    auto toycode = detect_code_in(img, bound_cnt, out_dst);
    // fill out_cnt
    out_cnt.assign(bound_cnt.begin(), bound_cnt.end());

    return toycode;
}


toy_code_t ToyDetector::detect_code_in(cv::Mat& img, std::vector<cv::Point>& cnt, cv::Mat& out_dst) {
    auto logger = spd::get("toy");

    int half_block_size = _block_size / 2;
    cv::Rect r = cv::boundingRect(cnt);
    if (r.width < half_block_size || r.height < half_block_size) {
        toy_code_t toycode = {};
        return toycode;
    }

    cv::Mat warped_dst;
    std::vector<cv::Point2f> roi_pnts;
    std::vector<cv::Point2f> dst_pnts(4);

    roi_pnts.push_back(cv::Point2f(cnt[0]));
    roi_pnts.push_back(cv::Point2f(cnt[1]));
    roi_pnts.push_back(cv::Point2f(cnt[2]));
    roi_pnts.push_back(cv::Point2f(cnt[3]));
    dst_pnts[0].x = 0;
    dst_pnts[0].y = 0;
    dst_pnts[1].x = (float)std::max(norm(roi_pnts[0] - roi_pnts[1]), norm(roi_pnts[2] - roi_pnts[3]));
    dst_pnts[1].y = 0;
    dst_pnts[2].x = (float)std::max(norm(roi_pnts[0] - roi_pnts[1]), norm(roi_pnts[2] - roi_pnts[3]));
    dst_pnts[2].y = (float)std::max(norm(roi_pnts[1] - roi_pnts[2]), norm(roi_pnts[3] - roi_pnts[0]));
    dst_pnts[3].x = 0;
    dst_pnts[3].y = (float)std::max(norm(roi_pnts[1] - roi_pnts[2]), norm(roi_pnts[3] - roi_pnts[0]));

    cv::Size warped_size(cvRound(dst_pnts[2].x), cvRound(dst_pnts[2].y));
    cv::Mat H = findHomography(roi_pnts, dst_pnts);
    cv::warpPerspective(img, warped_dst, H, warped_size);
    // for debug
    // cv::imwrite("warped_dst.png", warped_dst);

    auto shapes = detect_shape_in(warped_dst);

    cv::Size half_size(warped_size.width/2, warped_size.height/2);
    logger->debug("half_size:[{}, {}]", half_size.width, half_size.height);

    cv::Mat roi1 = warped_dst(cv::Rect(0, 0, half_size.width, half_size.height));
    cv::Mat roi2 = warped_dst(cv::Rect(half_size.width, 0, half_size.width, half_size.height));
    cv::Mat roi3 = warped_dst(cv::Rect(half_size.width, half_size.height, half_size.width, half_size.height));
    cv::Mat roi4 = warped_dst(cv::Rect(0, half_size.height, half_size.width, half_size.height));

    /* NOTE: for debug
    cv::imwrite("roi1.png", roi1);
    cv::imwrite("roi2.png", roi2);
    cv::imwrite("roi3.png", roi3);
    cv::imwrite("roi4.png", roi4);
    */

    std::array<color_t, 4> colors = {{
        detect_color(roi1), detect_color(roi2),
        detect_color(roi3), detect_color(roi4)
    }};

    int shp_idx = -1;
    for (int i=0; i<shapes.size(); i++) {
        if (shapes[i] == TRIANGLE) {
            shp_idx = i;
            break;
        }
    }

    // find triangle here
    if (shp_idx != -1) {
        std::rotate(shapes.begin(), shapes.begin() + shp_idx, shapes.end());
        std::rotate(colors.begin(), colors.begin() + shp_idx, colors.end());
    }

    toy_code_t toycode = {colors, shapes};
    return toycode;
}


std::array<shape_t, 4> ToyDetector::detect_shape_in(cv::Mat& roi) {
    cv::Mat gray, blurred, edged;
    cv::cvtColor(roi, gray, CV_BGR2GRAY);
    cv::medianBlur(gray, blurred, 3);
    cv::Canny(blurred, edged, 100, 120);

    cv::Size half_size(roi.cols/2, roi.rows/2);
    cv::Mat roi1 = edged(cv::Rect(0, 0, half_size.width, half_size.height));
    cv::Mat roi2 = edged(cv::Rect(half_size.width, 0, half_size.width, half_size.height));
    cv::Mat roi3 = edged(cv::Rect(half_size.width, half_size.height, half_size.width, half_size.height));
    cv::Mat roi4 = edged(cv::Rect(0, half_size.height, half_size.width, half_size.height));

    std::array<shape_t, 4> shapes = {{
        detect_shape(roi1), detect_shape(roi2),
        detect_shape(roi3), detect_shape(roi4)
    }};

    return shapes;
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


// FIXME: color detection
color_t ToyDetector::detect_color(cv::Mat& roi) {
    auto logger = spd::get("toy");

    cv::Mat hsv, mask, white_mask, hist;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(hsv, cv::Scalar(0, 20, 0), cv::Scalar(180, 255, 255), mask);
    cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 60, 255), white_mask);

    int w = white_mask.cols;
    int h = white_mask.rows;
    int cnz = cv::countNonZero(white_mask);

    float wr = cnz / ((float)(h * w));
    if (wr > 0.8) {
        return WHITE;
    }

    // int hsize = 30;
    int hsize = 180;
    int channels[] = {0};
    float hue_range[] = { 0.0f, 180.0f };
    const float* ranges[] = { hue_range };

    cv::calcHist(&hsv, 1, channels, mask, hist, 1, &hsize, ranges, true);
    cv::normalize(hist, hist, 0.0, 1.0, cv::NORM_MINMAX);

    double rval = sum_histogram(hist, RED_RANGE1) + sum_histogram(hist, RED_RANGE2);
    double yval = sum_histogram(hist, YELLOW_RANGE);
    double gval = sum_histogram(hist, GREEN_RANGE);
    double cval = sum_histogram(hist, CYAN_RANGE);
    double bval = sum_histogram(hist, BLUE_RANGE);
    double mval = sum_histogram(hist, MAGENTA_RANGE);

    logger->debug("r:{}, y:{}, g:{}, c:{}, b::{}, m:{}", rval, yval, gval, cval, bval, mval);

    // find max value
    std::vector<std::tuple<color_t, double> > tups = {
        {RED, rval}, {YELLOW, yval}, {GREEN, gval}, {CYAN, cval}, {BLUE, bval}, {MAGENTA, mval}
    };
    auto max_tup = *std::max_element(tups.begin(), tups.end(),
                                     [](const std::tuple<color_t, double>& t1,
                                        const std::tuple<color_t, double>& t2) -> bool {
                                        return std::get<1>(t1) < std::get<1>(t2);
                                    });
    double max_val = std::get<1>(max_tup);
    if (max_val > 1.0f) {
        return std::get<0>(max_tup);
    }
    return UNKNOWN;
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
