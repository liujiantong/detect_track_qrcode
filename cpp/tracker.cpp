#include "tracker.hpp"
#include "detector.hpp"
#include "helper.hpp"
#include "spdlog/spdlog.h"

#include <chrono>
#include <opencv2/xphoto/white_balance.hpp>


namespace spd = spdlog;

void ToyTracker::init_tracker() {
    auto logger = spd::get("console");

    cv::Size size = _camera->get_frame_width_and_height();
    // logger->debug("size from camera - w:{}, h{}", size.width, size.height);

    _frame_size = get_frame_size(size);
    logger->debug("_frame_size - w:{}, h{}", _frame_size.width, _frame_size.height);

    // allocate _frame and _debug_image here
    _frame.create(_frame_size.height, _frame_size.width, CV_32F);
    _debug_frame.create(_frame_size.height, _frame_size.width, CV_32F);

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

    _measurement = cv::Mat(n_measurements, 1, CV_32F);
    // _toy_prediction = cv::Mat(n_measurements, 1, CV_32F);

    _fgbg = cv::createBackgroundSubtractorMOG2(300, 16, false);

    logger->debug("init_tracker done.");
}


void ToyTracker::track() {
    auto logger = spd::get("console");
    logger->debug("tracker starting...");

    _is_running = true;

    ToyDetector detector;
    cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

    logger->debug("ready for while loop");

    while (true) {
        read_from_camera();
        logger->debug("read_from_camera done");

        _frame.copyTo(_debug_frame);
        logger->debug("copyTo _debug_frame done");

        cv::Rect united_rect = compute_fg_bound_rect(_frame, _frame_size, kernel);
        logger->debug("united_rect: w:{}, h:{}", united_rect.width, united_rect.height);

        if (united_rect.width > 0) {
            _united_fg = united_rect;
            int roi_x = _united_fg.x, roi_y = _united_fg.y;
            // int roi_w = _united_fg.width, roi_h = _united_fg.height;
            logger->debug("get roi_x:{} done", roi_x);

            cv::Mat roi_gray;
            cv::Mat roi_image = _frame(_united_fg);
            cv::cvtColor(roi_image, roi_gray, cv::COLOR_BGR2GRAY);
            logger->debug("cvtColor done");

            std::vector<std::vector<cv::Point> > founds = detector.find_contours(roi_gray);
            logger->debug("founds.size:{}", founds.size());

            if (!founds.empty()) {
                wb->balanceWhite(roi_image, roi_image);
                logger->debug("balanceWhite done");

                std::vector<cv::Point> cnt;
                std::vector<std::string> colors = detector.detect_color_from_contours(roi_image, founds, cnt);
                logger->debug("detect_color_from_contours cnt.size:{}", cnt.size());

                if (!cnt.empty()) {
                    _toy_contour.clear();
                    _toy_contour = cnt;
                    _toy_colors = colors;

                    for (auto& p : _toy_contour) {
                        p += cv::Point(roi_x, roi_y);
                    }
                    logger->debug("roi rect offset done.");

                    cv::Point cntr = pnts_center(_toy_contour);
                    _measurement.at<double>(0) = cntr.x;
                    _measurement.at<double>(1) = cntr.y;
                    _toy_prediction = _kalman.predict();
                    logger->debug("kalman predicted");

                    cv::Point2f c(cntr.x, cntr.y);
                    cv::minEnclosingCircle(cnt, c, _toy_radius);
                    add_new_tracker_point(cntr);
                    logger->debug("add_new_tracker_point, cntr:[{},{}]", cntr.x, cntr.y);
                }
            }
        } else {
            logger->debug("begin clear_debug_things");
            clear_debug_things();
        }

        if (_debug) {
            draw_debug_things();
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


void ToyTracker::read_from_camera() {
    // FIXME: concurrent frame write here.
    auto logger = spd::get("console");
    // logger->debug("_frame_size w:{}, h:{}", _frame_size.width, _frame_size.height);

    cv::Mat frame(_frame_size.height, _frame_size.width, CV_32F);
    cv::Mat* p_frame = _camera->read();
    cv::Mat cloned = (*p_frame).clone();
    logger->debug("read to p_frame done");

    cv::resize(cloned, frame, _frame_size, cv::INTER_AREA);
    cv::flip(frame, _frame, 1);
    logger->debug("-----> ToyTracker::read_from_camera done");
}


void ToyTracker::draw_debug_things(bool draw_fg, bool draw_contour, bool draw_prediction) {
    auto logger = spd::get("console");
    logger->debug("draw_debug_things");

    if (draw_fg && _united_fg.width > 0) {
        cv::rectangle(_debug_frame, _united_fg, cv::Scalar(0, 255, 0), 2);
    }

    if (draw_contour && !_toy_contour.empty()) {
        std::vector<std::vector<cv::Point> > cnts0 = {_toy_contour};
        cv::drawContours(_debug_frame, cnts0, 0, cv::Scalar(0, 0, 255), 2);
    }

    if (draw_prediction && _toy_radius > 0) {
        cv::Point c(_toy_prediction.at<float>(0), _toy_prediction.at<float>(0));
        cv::circle(_debug_frame, c, _toy_radius, cv::Scalar(255, 0, 0), 2);
    }
}


cv::Rect ToyTracker::compute_fg_bound_rect(const cv::Mat& frm_img, cv::Size max_size, cv::Mat& kernel) {
    auto logger = spd::get("console");
    logger->debug("compute_fg_bound_rect start");

    cv::Mat frm = frm_img.clone();
    logger->debug("frm cloned");

    cv::Mat fg_mask;
    _fgbg->apply(frm, fg_mask);
    logger->debug("_fgbg apply done");

    cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, kernel);
    cv::threshold(fg_mask, fg_mask, 60, 255, cv::THRESH_BINARY);
    logger->debug("fg_mask threshold done");

    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(fg_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    logger->debug("fg_mask findContours done");

    std::vector<cv::Rect> boxes;
    for (std::vector<cv::Point> c : contours) {
        if (cv::contourArea(c) > 100) {
            boxes.push_back(cv::boundingRect(c));
        }
    }
    logger->debug("filter boxes done. size:{}", boxes.size());

    if (boxes.empty()) {
        return cv::Rect(0, 0, -1, -1);
    }

    cv::Rect r = union_rects(boxes);
    logger->debug("boxes union_rects done");

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
