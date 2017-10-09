#ifndef __HIVE_MOCK_TRACKER_HPP__
#define __HIVE_MOCK_TRACKER_HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto/white_balance.hpp>
#include <vector>
#include <deque>

#include "camera.hpp"


class MockTracker;
typedef void (*tracking_callback)(MockTracker*);


class MockTracker {

public:
    MockTracker(SimpleCamera* cam, int nb_of_cntr=30, bool debug=false) : _camera(cam),
    _max_nb_of_centers(nb_of_cntr), _debug(debug) {
        _tracking_cb = nullptr;
        _united_fg = cv::Rect(0, 0, -1, -1);
        init_tracker();
    };

    void track();


    cv::Mat* get_frame() {
        return &_frame;
    };

    cv::Mat* get_debug_frame() {
        return &_debug_frame;
    };

    cv::Point get_last_toy_center() {
        if (_tracker_centers.empty()) {
            return cv::Point(-1, -1);
        }
        return _tracker_centers.back();
    };

    std::vector<std::string> get_toy_colors() {
        return _toy_colors;
    };

    void set_tracking_callback(tracking_callback cb) {
        _tracking_cb = cb;
    };

    void stop_tracking() {
        _is_running = false;
    };

private:
    SimpleCamera* _camera;
    std::deque<cv::Point> _tracker_centers;
    int _max_nb_of_centers;
    tracking_callback _tracking_cb;
    bool _is_running;
    cv::Size _frame_size;
    cv::Mat _frame;
    cv::Mat _debug_frame;
    bool _debug;

    cv::Rect _united_fg;
    std::vector<cv::Point> _toy_contour;
    std::vector<std::string> _toy_colors;
    float _toy_radius;
    cv::Mat _measurement;
    cv::Mat _toy_prediction;

    cv::KalmanFilter _kalman;
    cv::Ptr<cv::BackgroundSubtractorMOG2> _fgbg;
    cv::Ptr<cv::xphoto::SimpleWB> _wb;

    void init_tracker();
    void read_from_camera();

    void add_new_tracker_point(cv::Point pnt, int min_distance=20, int max_distance=1000);
    // cv::Rect compute_fg_bound_rect(const cv::Mat frm, cv::Size max_size, cv::Mat& kernel);
    cv::Rect compute_fg_bound_rect(const cv::Mat& frm, cv::Size max_size, cv::Mat& kernel);
    void draw_debug_things(bool draw_fg=false, bool draw_contour=true, bool draw_prediction=true);

    void clear_debug_things() {
        _toy_radius = 0;
        _toy_colors.clear();
        _toy_contour.clear();
        _toy_prediction = cv::Mat();
    };

};

#endif // __HIVE_MOCK_TRACKER_HPP__
