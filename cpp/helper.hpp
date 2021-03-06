#ifndef __HIVE_HELPER_HPP__
#define __HIVE_HELPER_HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <tuple>

#include "toy.hpp"


cv::Size get_frame_size(cv::Size size, unsigned max_width=1024);
cv::Rect union_rects(std::vector<cv::Rect>& rects);
cv::Point pnts_center(std::vector<cv::Point>& points);
cv::Point contour_center(std::vector<cv::Point> contour);
double calc_distance(const cv::Point pt1, const cv::Point pt2);
double angle_cos(cv::Point pt0, cv::Point pt1, cv::Point pt2);
double sum_histogram(cv::Mat& hist, const cv::Range& range);
std::tuple<double, double> calc_mean_stdev(const std::vector<double> v);
direct_pos_t calc_direct(cv::Point head, cv::Point tail);

std::string join(std::vector<std::string>& v);

std::string toy_direct_name(direct_t d);
std::string color_name(color_t c);
std::string shape_name(shape_t s);

std::vector<std::string> get_color_names(std::array<color_t, 4>& colors);
std::string color_names_str(std::array<color_t, 4>& colors);
std::vector<std::string> get_shape_names(std::array<shape_t, 4>& shapes);
std::string shape_names_str(std::array<shape_t, 4>& shapes);

shape_t detect_shape(cv::Mat& img);
std::string toy_code_str(toy_code_t& code);


#endif // __HIVE_HELPER_HPP__
