#include "detector.hpp"
#include "helper.hpp"

#include <tuple>


const int RED_RANGE1[]  = {0, 30};
const int RED_RANGE2[]  = {150, 180};
const int GREEN_RANGE[] = {30, 90};
const int BLUE_RANGE[]  = {90, 140};


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

    double rval = sum_histogram(hist, RED_RANGE1[0], RED_RANGE1[1]) + \
                  sum_histogram(hist, RED_RANGE2[0], RED_RANGE2[1]);
    double gval = sum_histogram(hist, GREEN_RANGE[0], GREEN_RANGE[1]);
    double bval = sum_histogram(hist, BLUE_RANGE[0], BLUE_RANGE[1]);

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
