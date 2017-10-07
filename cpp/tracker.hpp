#ifndef __HIVE_TRACKER_HPP__
#define __HIVE_TRACKER_HPP__

#include <opencv2/opencv.hpp>
#include <vector>

#include "camera.hpp"


typedef void (*tracking_callback)();


class ToyTracker {

public:
    ToyTracker(SimpleCamera* cam) : _camera(cam) {
    };

private:
    SimpleCamera* _camera;
    std::vector<cv::Point> _tracker_centers;
    bool _debug;
    int _max_nb_of_centers;
    tracking_callback _tracking_cb;
    bool _is_running;
    cv::Size _frame_size;
    cv::Mat _frame;
    cv::Mat _debug_frame;
    cv::KalmanFilter _kalman;
    cv::Ptr<cv::BackgroundSubtractorMOG2> _fgbg;

    void init_tracker();

};

#endif // __HIVE_TRACKER_HPP__
