#include "mocktracker.hpp"
#include "detector.hpp"
#include "helper.hpp"
#include "spdlog/spdlog.h"

#include <opencv2/core.hpp>
#include <iostream>
#include <chrono>


namespace spd = spdlog;

int hsize = 16;
float hranges[] = {0, 180};
const float* phranges = hranges;

void MockTracker::init_tracker() {
    auto logger = spd::get("toy");

    cv::Size size = _camera->get_frame_size();
    _frame_size = get_frame_size(size);

    // allocate _frame and _debug_image here
    _frame.create(_frame_size.height, _frame_size.width, CV_32F);
    _debug_frame.create(_frame_size.height, _frame_size.width, CV_32F);

    init_kalman();
    _fgbg = cv::createBackgroundSubtractorMOG2(300, 16, false);
    _wb = cv::xphoto::createSimpleWB();
    logger->info("init_tracker done.");
}


void MockTracker::init_kalman() {
    auto logger = spd::get("toy");
    logger->debug("init_kalman start");

    int n_states = 4, n_measurements = 2;
    _kalman.init(n_states, n_measurements);

    // FIXME:BUG HERE
    /* DYNAMIC MODEL
    [1, 0, 1, 0]
    [0, 1, 0, 1]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    */
    _kalman.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0,  0,1,0,1,  0,0,1,0,  0,0,0,1);

    /* MEASUREMENT MODEL
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    */
    cv::setIdentity(_kalman.measurementMatrix);

    /*
    [dt, 0, 0, 0]
    [0, dt, 0, 0]
    [0, 0, dt, 0]
    [0, 0, 0, dt]
    */
    float dt = 0.045; // time between measurements (1/FPS)
    cv::setIdentity(_kalman.processNoiseCov, cv::Scalar::all(dt));
    cv::setIdentity(_kalman.measurementNoiseCov, cv::Scalar::all(dt));
    // cv::setIdentity(_kalman.errorCovPost, cv::Scalar::all(dt));

    _measurement = cv::Mat::zeros(n_measurements, 1, CV_32F);
    _toy_prediction = cv::Mat::zeros(4, 1, CV_32F);
}


void MockTracker::track() {
    auto logger = spd::get("toy");
    logger->info("tracker starting...");

    _is_running = true;

    ToyDetector detector;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

    while (true) {
        read_from_camera();

        _frame.copyTo(_debug_frame);
        cv::Rect united_rect = compute_fg_bound_rect(_frame, _frame_size, kernel);

        if (united_rect.width > 0) {
            _united_fg = united_rect;
            int roi_x = _united_fg.x, roi_y = _united_fg.y;

            cv::Mat roi_gray;
            cv::Mat roi_image = _frame(_united_fg);
            cv::cvtColor(roi_image, roi_gray, cv::COLOR_BGR2GRAY);

            std::vector<std::vector<cv::Point> > founds = detector.find_code_contours(roi_gray);

            if (!founds.empty()) {
                _wb->balanceWhite(roi_image, roi_image);

                std::vector<cv::Point> cnt;
                std::vector<std::string> colors = detector.detect_color_from_contours(roi_image, founds, cnt);

                if (!cnt.empty()) {
                    _toy_contour.clear();
                    _toy_contour = cnt;
                    _toy_colors = colors;

                    for (auto& p : _toy_contour) {
                        p += cv::Point(roi_x, roi_y);
                    }

                    cv::Point cntr = pnts_center(_toy_contour);
                    _measurement.at<float>(0) = cntr.x;
                    _measurement.at<float>(1) = cntr.y;
                    _kalman.correct(_measurement);
                    logger->info("measurement: ({}, {})", cntr.x, cntr.y);

                    _toy_prediction = _kalman.predict();
                    // std::cout << "_toy_prediction:" << cv::format(_toy_prediction, cv::Formatter::FMT_NUMPY) << std::endl;
                    logger->info("toy_prediction: ({}, {})", _toy_prediction.at<float>(0), _toy_prediction.at<float>(1));

                    cv::Point2f c;
                    cv::minEnclosingCircle(cnt, c, _toy_radius);
                    add_new_tracker_point(cntr);
                }
            }
        } else {
            logger->debug("begin clear_debug_things");
            clear_debug_things();
        }

        if (_debug) {
            draw_debug_things(true);
        }

        if (_tracking_cb) {
            _tracking_cb(this);
        }

        if (!_is_running) {
            break;
        }

        std::this_thread::yield();
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}


void MockTracker::read_from_camera() {
    // concurrent frame write here.
    auto logger = spd::get("toy");

    cv::Mat frm = _camera->read();
    if (frm.empty()) {
        logger->error("read_from_camera: frm empty");
        return;
    }

    cv::resize(frm, _frame, _frame_size, cv::INTER_AREA);
    // cv::flip(_frame, _frame, 1);
}


void MockTracker::calc_hist(cv::Mat& roi, cv::Mat& roi_mask) {
    calcHist(&roi, 1, 0, roi_mask, _toy_hist, 1, &hsize, &phranges);
    normalize(_toy_hist, _toy_hist, 0, 180, cv::NORM_MINMAX);
}


void MockTracker::camshift(cv::Mat& hue, cv::Mat& mask, cv::Rect track_window) {
    cv::Mat backproj;
    cv::calcBackProject(&hue, 1, 0, _toy_hist, backproj, &phranges);
    backproj &= mask;
    cv::RotatedRect track_box = cv::CamShift(backproj, track_window, cv::TermCriteria(
                                             cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1 ));
    if( track_window.area() <= 1 ) {
        // TODO:
    }
}


void MockTracker::draw_debug_things(bool draw_fg, bool draw_contour, bool draw_prediction) {
    auto logger = spd::get("toy");
    logger->debug("draw_debug_things");

    if (draw_fg && _united_fg.width > 0) {
        cv::rectangle(_debug_frame, _united_fg, cv::Scalar(0, 255, 0), 2);
    }

    if (draw_contour && !_toy_contour.empty()) {
        std::vector<std::vector<cv::Point> > cnts0 = {_toy_contour};
        cv::drawContours(_debug_frame, cnts0, 0, cv::Scalar(0, 0, 255), 2);
    }

    if (draw_prediction && _toy_radius > 0) {
        // FIXME: Floating point exception: 8, (mocktracker.cpp:195)
        cv::Point c(_toy_prediction.at<float>(0), _toy_prediction.at<float>(1));
        cv::circle(_debug_frame, c, _toy_radius, cv::Scalar(255, 0, 0), 2);
        logger->debug("draw prediction");
    }
}


cv::Rect MockTracker::compute_fg_bound_rect(const cv::Mat& frm, cv::Size max_size, cv::Mat& kernel) {
    auto logger = spd::get("toy");
    logger->debug("compute_fg_bound_rect start");

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
    if (r.x + w > max_size.width) {
        w = max_size.width - r.x;
    }
    int h = r.height;
    if (r.y + h > max_size.height) {
        h = max_size.height - r.y;
    }

    return cv::Rect(r.x, r.y, w, h);
}

void MockTracker::add_new_tracker_point(cv::Point pnt, int min_distance, int max_distance) {
    auto logger = spd::get("toy");
    logger->debug("add_new_tracker_point start...");

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
