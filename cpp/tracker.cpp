#include "tracker.hpp"
#include "detector.hpp"
#include "helper.hpp"

#include <opencv2/xphoto/white_balance.hpp>


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


void ToyTracker::track() {
    _is_running = true;

    ToyDetector detector;
    cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

    while (true) {
        // TODO:

        if (!_is_running) {
            break;
        }
    }


    // TODO:
    // self._is_running = True
    //
    // detector = ToyDetector()
    // wb = cv2.xphoto.createSimpleWB()
    // kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    //
    // while True:
}


void ToyTracker::read_from_camera() {
    cv::Mat frame;
    cv::Mat* p_frame = _camera->read();
    cv::resize(*p_frame, frame, _frame_size, cv::INTER_AREA);
    cv::flip(frame, _frame, 1);
}


cv::Rect ToyTracker::compute_bound_rect(cv::Mat& frm, int max_x, int max_y, cv::Mat& kernel) {
    cv::Mat fg_mask;
    _fgbg->apply(frm, fg_mask);

    cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, kernel);
    cv::threshold(fg_mask, fg_mask, 60, 255, cv::THRESH_BINARY);

    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(fg_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> boxes;
    for (std::vector<cv::Point> c : contours) {
        if (cv::contourArea(c) > 100) {
            boxes.push_back(cv::boundingRect(c));
        }
    }

    if (boxes.empty()) {
        return cv::Rect(0, 0, -1, -1);
    }

    cv::Rect r = union_rects(boxes);
    int w = r.width;
    if (r.x + w > max_x) {
        w = max_x - r.x;
    }
    int h = r.height;
    if (r.y + h > max_y) {
        h = max_y - r.y;
    }

    return cv::Rect(r.x, r.y, w, h);
}

void ToyTracker::add_new_tracker_point(cv::Point pnt, int min_distance, int max_distance) {
    if (_tracker_centers.empty()) {
        _tracker_centers.push_back(pnt);
    } else {
        double dst = calc_distance(pnt, _tracker_centers.back());
        if (dst > min_distance && dst < max_distance) {
            if (_tracker_centers.size() > _max_nb_of_centers) {
                _tracker_centers.pop_front();
            }
            _tracker_centers.push_back(pnt);
        }
    }
}
