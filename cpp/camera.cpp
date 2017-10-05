#include <iostream>
#include <thread>

#include "camera.hpp"


void SimpleCamera::start_camera() {
    init_camera();
    _worker = std::thread(&SimpleCamera::update_camera, this);
} // SimpleCamera::start_camera


void SimpleCamera::release_camera() {
    _is_running = false;
    _cam->release();
    delete _cam;
} // SimpleCamera::release_camera


bool SimpleCamera::init_camera() {
    if (!_cam->isOpened()) {
        std::cerr << "Failed to open the video device or video file!\n" << std::endl;
        return false;
    }

    _fps = _cam->get(cv::CAP_PROP_FPS);
    _frame_height = _cam->get(cv::CAP_PROP_FRAME_HEIGHT);
    _frame_width = _cam->get(cv::CAP_PROP_FRAME_WIDTH);

    *_cam >> _frame;
    if (_frame.empty())
        _ret = false;
    else
        _ret = true;

    return _ret;
} // SimpleCamera::init_camera


void SimpleCamera::update_camera() {
    while (true) {
        if (_is_running) {
            *_cam >> _frame;
            if (_frame.empty()) {
                _ret = false;
            }
        } else {
            break;
        }
    }
} // SimpleCamera::update_camera
