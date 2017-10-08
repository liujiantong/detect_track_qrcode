#include "tracker.hpp"
#include "detector.hpp"
#include "helper.hpp"

#include <chrono>
#include <opencv2/xphoto/white_balance.hpp>


void ToyTracker::init_tracker() {
    cv::Size size = _camera->get_frame_width_and_height();
    _frame_size = get_frame_size(size);

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

    _fgbg = cv::createBackgroundSubtractorMOG2(300, 16, false);
}


void ToyTracker::track() {
    _is_running = true;

    ToyDetector detector;
    cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

    while (true) {
        read_from_camera();
        _frame.copyTo(_debug_frame);

        cv::Rect united_rect = compute_bound_rect(_frame, _frame_size, kernel);
        if (united_rect.width > 0) {
            _united_fg = united_rect;
            int roi_x = _united_fg.x, roi_y = _united_fg.y;
            int roi_w = _united_fg.width, roi_h = _united_fg.height;

            cv::Mat roi_gray;
            cv::Mat roi_image = _frame(_united_fg);
            cv::cvtColor(roi_image, roi_gray, cv::COLOR_BGR2GRAY);

            std::vector<std::vector<cv::Point> > founds = detector.find_contours(roi_gray);
            if (!founds.empty()) {
                wb->balanceWhite(roi_image, roi_image);
                std::vector<cv::Point> cnt;
                std::vector<std::string> colors = detector.detect_color_from_contours(roi_image, founds, cnt);
            }
        }

        if (!_is_running) {
            break;
        }

        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }


    /* TODO:
    while True:
        self._read_from_camera()
        self._debug_frame = self._frame.copy()

        united_rect = self._compute_bound_rect(self._frame, self._frame_width, self._frame_height, kernel)
        if united_rect is not None:
            self._united_fg = united_rect
            roi_x, roi_y, roi_w, roi_h = self._united_fg

            roi_image = self._frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            // roi_image = wb.balanceWhite(roi_image)
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

            founds = detector.find_contours(roi_gray)
            if founds:
                roi_image = wb.balanceWhite(roi_image)
                colors, cnt = detector.detect_color_from_contours(roi_image, founds)
                if cnt is not None:
                    self._toy_colors = colors
                    self._toy_contour = cnt + np.array([roi_x, roi_y])
                    self._measurement = helper.center(self._toy_contour)
                    self._kalman.correct(self._measurement)
                    self._toy_prediction = self._kalman.predict()
                    _, self._toy_radius = cv2.minEnclosingCircle(cnt)
                    self._add_new_tracker_point()
        else:
            self._clear_debug_things()

        if self._debug:
            self._draw_debug_things(draw_fg=False)

        if self._tracking_callback is not None:
            try:
                self._tracking_callback()
            except TypeError:
                import warnings
                warnings.warn(
                    "Tracker callback function is not working because of wrong arguments! "
                    "It takes zero arguments")

        if not self._is_running:
            break
    */
}


void ToyTracker::read_from_camera() {
    cv::Mat frame;
    cv::Mat* p_frame = _camera->read();
    cv::resize(*p_frame, frame, _frame_size, cv::INTER_AREA);
    cv::flip(frame, _frame, 1);
}


cv::Rect ToyTracker::compute_bound_rect(cv::Mat& frm, cv::Size max_size, cv::Mat& kernel) {
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
