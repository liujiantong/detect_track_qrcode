#include "detector.hpp"

#include <tuple>


std::string ToyDetector::detect_color(cv::Mat& roi) {

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
