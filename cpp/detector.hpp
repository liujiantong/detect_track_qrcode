#ifndef __HIVE_DETECTOR_HPP__
#define __HIVE_DETECTOR_HPP__

#include <tuple>
#include <string>
#include <opencv2/opencv.hpp>

#include "toy.hpp"


class ToyDetector {
public:
    ToyDetector(int block_size=100) : _block_size(block_size) {
    };

    std::vector<std::vector<cv::Point> > find_code_contours(cv::Mat& gray);
    std::tuple<bool, std::vector<cv::Point> > detect_square(std::vector<cv::Point>& cnt);
    static color_t detect_color(cv::Mat& roi);
    toy_code_t detect_code_in(cv::Mat& img, std::vector<cv::Point>& cnt, cv::Mat& out_dst);
    std::array<shape_t, 4> detect_shape_in(cv::Mat& roi_edged);
    toy_code_t detect_code_from_contours(cv::Mat& img,
        std::vector<std::vector<cv::Point> >& cnts,
        std::vector<cv::Point>& out_cnt);

private:
    static std::vector<cv::Point> check_cnt_contain(std::vector<std::vector<cv::Point> >& cnts);

    int _block_size;
};

#endif // __HIVE_DETECTOR_HPP__
