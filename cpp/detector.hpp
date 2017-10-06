#ifndef __HIVE_DETECTOR_HPP__
#define __HIVE_DETECTOR_HPP__

#include <string>
#include <opencv2/opencv.hpp>

class ToyDetector {
public:
    ToyDetector(int block_size=100) : block_size(block_size) {
    };

private:
    static std::vector<cv::Point> check_cnt_contain(std::vector<std::vector<cv::Point> >& cnts);

    static double angle_cos(cv::Point pt0, cv::Point pt1, cv::Point pt2) {
        double dx1 = pt0.x - pt1.x;
        double dy1 = pt0.y - pt1.y;
        double dx2 = pt2.x - pt1.x;
        double dy2 = pt2.y - pt1.y;
        return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
    };

    int block_size;
};

#endif // __HIVE_DETECTOR_HPP__
