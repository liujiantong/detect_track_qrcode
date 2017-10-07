#include "detector.hpp"
#include "helper.hpp"

#include <algorithm>


const cv::Range RED_RANGE1(10, 30);
const cv::Range RED_RANGE2(150, 180);
const cv::Range GREEN_RANGE(30, 90);
const cv::Range BLUE_RANGE(90, 140);


std::vector<std::vector<cv::Point> > ToyDetector::find_contours(cv::Mat& gray) {
    cv::Mat blurred, edges;
    cv::medianBlur(gray, blurred, 3);
    cv::Canny(blurred, edges, 100, 120);

    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours, found_cnts;
    cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (hierarchy.empty()) {
        return found_cnts;
    }

    cv::Vec4i hierarchy0 = hierarchy[0];
    for (int cnt_idx=0; cnt_idx<contours.size(); cnt_idx++) {
        double area = cv::contourArea(contours[cnt_idx], true);
        if (area < 100) {
            continue;
        }

        int k = cnt_idx, c = 0;
        while (hierarchy0[k][2] != -1) {
            k = hierarchy0[k][2];
            c++;
        }
        if (c > 0) {
            found_cnts.push_back(contours[cnt_idx]);
        }
    }

    return found_cnts;
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
        if (std::get<0>(mean_std) != 0.0) {
            z_val = std::get<1>(mean_std) / std::get<0>(mean_std);
        }

        if (z_val < 0.18 && max_cos < 0.35) {
            return std::make_tuple(true, approx);
        }
    }

    return std::make_tuple(false, approx);
}


std::string ToyDetector::detect_color(cv::Mat& roi) {
    cv::Mat hsv, mask, white_mask, hue, hist;
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

    hue.create(hsv.size(), hsv.depth());
    int ch[] = { 0, 0 };
    cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

    // int hsize = 30;
    int hsize = 180;
    float hue_range[] = { 0, 180 };
    const float* ranges = { hue_range };

    cv::calcHist(&hue, 1, 0, cv::Mat(), hist, 1, &hsize, &ranges);
    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

    double rval = sum_histogram(hist, RED_RANGE1) + sum_histogram(hist, RED_RANGE2);
    double gval = sum_histogram(hist, GREEN_RANGE);
    double bval = sum_histogram(hist, BLUE_RANGE);

    if (rval > 0.8) {
        return "red";
    } else if (gval > 0.8) {
        return "green";
    } else if (bval > 0.8) {
        return "blue";
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
