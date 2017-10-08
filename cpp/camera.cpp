#include <iostream>
#include <thread>
#include <stdexcept>

#include "spdlog/spdlog.h"

#include "camera.hpp"


namespace spd = spdlog;

void SimpleCamera::start_camera() {
    auto logger = spd::get("console");
    logger->info("start camera");

    bool ret = init_camera();
    if (!ret) {
        spd::get("console")->error("Failed to init camera");
        throw std::runtime_error("Failed to init camera");
    }
    logger->debug("ready to start worker");

    _worker = std::thread(&SimpleCamera::update_camera, this);
    _worker.detach();
} // SimpleCamera::start_camera


bool SimpleCamera::init_camera() {
    auto logger = spd::get("console");

    if (!_cam.isOpened()) {
        // std::cerr << "Failed to open the video device or video file!\n" << std::endl;
        spd::get("console")->error("Failed to open the video device or video file!");
        return false;
    }

    _fps = _cam.get(cv::CAP_PROP_FPS);
    _frame_size.width = _cam.get(cv::CAP_PROP_FRAME_WIDTH);
    _frame_size.height = _cam.get(cv::CAP_PROP_FRAME_HEIGHT);
    logger->debug("width:{}, height:{}", _frame_size.width, _frame_size.height);

    _cam >> _frame;
    // logger->debug("read to _frame");

    if (_frame.empty()) {
        _ret = false;
    } else {
        _is_running = true;
        _ret = true;
    }

    return _ret;
} // SimpleCamera::init_camera


void SimpleCamera::update_camera() {
    while (_is_running) {
        _cam >> _frame;
        if (_frame.empty()) {
            _ret = false;
        }
        std::this_thread::yield();
    }
} // SimpleCamera::update_camera
