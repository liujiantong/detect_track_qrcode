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

    // init_kalman();
    _fgbg = cv::createBackgroundSubtractorMOG2(300, 16, false);
    _wb = cv::xphoto::createSimpleWB();
    logger->info("init_tracker done.");
}


void MockTracker::init_kalman() {
    auto logger = spd::get("toy");
    logger->debug("init_kalman start");

    int n_states = 4, n_measurements = 2;
    _kalman.init(n_states, n_measurements);

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


void MockTracker::kalman_track(cv::Point cntr) {
    auto logger = spd::get("toy");

    kalman_tracked = true;
    _measurement.at<float>(0) = cntr.x;
    _measurement.at<float>(1) = cntr.y;
    _kalman.correct(_measurement);
    logger->debug("measurement: ({}, {})", cntr.x, cntr.y);
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
            int r = std::min(united_rect.width, united_rect.height) / 7;
            _united_fg = (united_rect + cv::Size(r, r)) &
                         cv::Rect(0, 0, _frame_size.width, _frame_size.height);
            // _united_fg = united_rect;
            int roi_x = _united_fg.x, roi_y = _united_fg.y;

            cv::Mat roi_gray;
            cv::Mat roi_image = _frame(_united_fg);
            cv::cvtColor(roi_image, roi_gray, cv::COLOR_BGR2GRAY);

            auto founds = detector.find_code_contours(roi_gray);

            if (!founds.empty()) {
                _wb->balanceWhite(roi_image, roi_image);

                std::vector<cv::Point> cnt;
                auto colors = detector.detect_color_from_contours(roi_image, founds, cnt);

                if (!cnt.empty()) {
                    _track_window = cv::boundingRect(_toy_contour);

                    _toy_contour.clear();
                    _toy_contour = cnt;
                    _toy_colors = colors;

                    for (auto& p : _toy_contour) {
                        p += cv::Point(roi_x, roi_y);
                    }

                    if (_toy_hist.empty()) {
                        cv::Mat hsv;
                        cv::cvtColor(_frame, hsv, CV_BGR2HSV);
                        calc_hist(_frame, _track_window);
                    }

                    cv::Point cntr = pnts_center(_toy_contour);
                    // kalman_track(cntr);

                    cv::Point2f c;
                    cv::minEnclosingCircle(cnt, c, _toy_radius);
                    add_new_tracker_point(cntr);
                }
            } else {
                // check status and camshift tracking here
                logger->info("I lost square, search it by camshift");
                if (!_toy_hist.empty()) {
                    if (_track_window.width < 1) {
                        int w = _frame_size.width, h = _frame_size.height;
                        _track_window = cv::Rect(0, 0, w / 4, h / 4);
                    }
                    cv::Mat hsv, mask;
                    cv::cvtColor(_frame, hsv, CV_BGR2HSV);
                    cv::inRange(hsv, cv::Scalar(0, 20, 10), cv::Scalar(180, 255, 255), mask);

                    logger->debug("_track_window0:[{}, {}, {}, {}]", _track_window.x, _track_window.y, _track_window.width, _track_window.height);
                    camshift_track(hsv, mask, _track_window);
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
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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


void MockTracker::calc_hist(cv::Mat& hsv, cv::Rect roi_rect) {
    auto logger = spd::get("toy");

    cv::Mat mask;
    int channels[] = {0};

    cv::Mat roi = hsv(roi_rect);
    cv::inRange(roi, cv::Scalar(0, 20, 0), cv::Scalar(180, 255, 255), mask);

    cv::calcHist(&roi, 1, channels, mask, _toy_hist, 1, &hsize, &phranges);
    normalize(_toy_hist, _toy_hist, 0, 180, cv::NORM_MINMAX);
    logger->info("calc_hist calcHist done");
}


cv::Rect MockTracker::camshift_track(cv::Mat& hsv, cv::Mat& mask, cv::Rect track_win) {
    auto logger = spd::get("toy");
    logger->debug("camshift_track start");

    cv::Mat hue, backproj;
    hue.create(hsv.size(), hsv.depth());
    int ch[] = {0, 0};
    cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

    cv::calcBackProject(&hue, 1, 0, _toy_hist, backproj, &phranges);
    backproj &= mask;
    // cv::RotatedRect track_box ignored
    cv::CamShift(backproj, track_win, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1 ));
    if (track_win.area() <= 9) {
        // TODO: lost target.
        int r = (std::min(backproj.cols, backproj.rows) + 5)/6;
        _track_window = cv::Rect(track_win.x-r, track_win.y-r, track_win.x+2*r, track_win.y+2*r) &
                                 cv::Rect(0, 0, _frame_size.width, _frame_size.height);
        logger->warn("I lost toy after camshift. [x:{}, y:{}, w:{}, h:{}].", track_win.x-r, track_win.y-r, track_win.x+2*r, track_win.y+2*r);
    } else {
        logger->info("I found toy again");
        _track_window = track_win;
    }

    logger->debug("_track_window:[{}, {}, {}, {}]", _track_window.x, _track_window.y, _track_window.width, _track_window.height);
    return _track_window;
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

    // if (kalman_tracked && draw_prediction && _toy_radius > 0) {
    //     cv::Point c(_toy_prediction.at<float>(0), _toy_prediction.at<float>(1));
    //     cv::circle(_debug_frame, c, _toy_radius, cv::Scalar(255, 0, 0), 2);
    //     logger->debug("draw prediction");
    // }

    if (draw_prediction && !_toy_hist.empty() && _track_window.width > 0) {
        cv::rectangle(_debug_frame, _track_window, cv::Scalar(255, 0, 0), 2);
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
    return (r & cv::Rect(0, 0, max_size.width, max_size.height));
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
