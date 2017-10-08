#ifndef __HIVE_TRACKER_HPP__
#define __HIVE_TRACKER_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

#include "camera.hpp"


typedef void (*tracking_callback)();


class ToyTracker {

public:
    ToyTracker(SimpleCamera* cam, int nb_of_cntr=30, bool debug=false) : _camera(cam),
    _max_nb_of_centers(nb_of_cntr), _debug(debug) {
        _tracking_cb = NULL;
        _united_fg = cv::Rect(0, 0, -1, -1);
    };

    void track();

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

    void init_tracker();
    void read_from_camera();

    void add_new_tracker_point(cv::Point pnt, int min_distance=20, int max_distance=1000);
    cv::Rect compute_bound_rect(cv::Mat& frm, cv::Size max_size, cv::Mat& kernel);
    void draw_debug_things(bool draw_fg=false, bool draw_contour=true, bool draw_prediction=true);

    void clear_debug_things() {
        _toy_radius = 0;
        _toy_colors.clear();
        _toy_contour.clear();
        _toy_prediction = cv::Mat();
    };

    void stop_tracking() {
        _is_running = false;
    };

    cv::Mat* get_debug_image() {
        return &_debug_frame;
    };

    cv::Mat* get_frame() {
        return &_frame;
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

};

#endif // __HIVE_TRACKER_HPP__
