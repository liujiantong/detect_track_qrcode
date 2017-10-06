#ifndef __HIVE_HELPER_HPP__
#define __HIVE_HELPER_HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>


cv::Size get_frame_size(cv::Size size, unsigned max_width=1024);
cv::Rect union_rects(std::vector<cv::Rect>& rects);
cv::Point center(std::vector<cv::Point>& points);
cv::Point contour_center(std::vector<cv::Point> contour, bool bin_img=true);
double calc_distance(const cv::Point pt1, const cv::Point pt2);
double sum_histogram(cv::Mat& hist, int low, int high);


#endif // __HIVE_HELPER_HPP__
