#include "tracker.hpp"
#include "helper.hpp"


void ToyTracker::init_tracker() {
    cv::Size size = _camera->get_frame_width_and_height();
    _frame_size = get_frame_size(size);

    int n_states = 4, n_measurements = 2;
    _kalman.init(n_states, n_measurements);

    /* DYNAMIC MODEL
    [1, 0, 1, 0]
    [0, 1, 0, 1]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    */
    int dt = 1;
    _kalman.transitionMatrix.at<double>(0, 0) = dt;
    _kalman.transitionMatrix.at<double>(0, 2) = dt;
    _kalman.transitionMatrix.at<double>(1, 1) = dt;
    _kalman.transitionMatrix.at<double>(1, 3) = dt;
    _kalman.transitionMatrix.at<double>(2, 2) = dt;
    _kalman.transitionMatrix.at<double>(3, 3) = dt;

    /* MEASUREMENT MODEL
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    */
    _kalman.measurementMatrix.at<double>(0, 0) = 1;  // x
    _kalman.measurementMatrix.at<double>(1, 1) = 1;  // y

    /*
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    */
    float cov = 0.03f;
    _kalman.processNoiseCov.at<double>(0, 0) = cov;
    _kalman.processNoiseCov.at<double>(1, 1) = cov;
    _kalman.processNoiseCov.at<double>(2, 2) = cov;
    _kalman.processNoiseCov.at<double>(3, 3) = cov;

    _fgbg = cv::createBackgroundSubtractorMOG2(300, 16, false);
}
