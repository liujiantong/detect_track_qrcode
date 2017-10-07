#ifndef __HIVE_DETECTOR_HPP__
#define __HIVE_DETECTOR_HPP__

#include <tuple>
#include <string>
#include <opencv2/opencv.hpp>


class ToyDetector {
public:
    ToyDetector(int block_size=100) : block_size(block_size) {
    };

    std::vector<std::vector<cv::Point> > find_contours(cv::Mat& gray);
    std::tuple<bool, std::vector<cv::Point> > detect_square(std::vector<cv::Point>& cnt);

private:
    static std::vector<cv::Point> check_cnt_contain(std::vector<std::vector<cv::Point> >& cnts);
    static std::string detect_color(cv::Mat& roi);

    int block_size;
};

#endif // __HIVE_DETECTOR_HPP__
