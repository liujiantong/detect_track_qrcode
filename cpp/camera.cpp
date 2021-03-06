#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>

#include "spdlog/spdlog.h"

#include "camera.hpp"


namespace spd = spdlog;

void SimpleCamera::start_camera() {
    auto logger = spd::get("toy");
    logger->info("start camera");

    bool ret = init_camera();
    if (!ret) {
        logger->error("Failed to init camera");
        throw std::runtime_error("Failed to init camera");
    }
    logger->debug("ready to start worker");

    std::hash<std::thread::id> hasher;
    logger->debug("_worker:{} before start", hasher(_worker.get_id()));
    _worker = std::thread(&SimpleCamera::update_camera, this);
    logger->info("_worker:{} started", hasher(_worker.get_id()));
    _worker.detach();
} // SimpleCamera::start_camera


bool SimpleCamera::init_camera() {
    auto logger = spd::get("toy");

    if (!_cam.isOpened()) {
        logger->error("Failed to open the video device or video file!");
        return false;
    }
    logger->debug("_cam opened");

    _fps = _cam.get(cv::CAP_PROP_FPS);
    _frame_size.width = _cam.get(cv::CAP_PROP_FRAME_WIDTH);
    _frame_size.height = _cam.get(cv::CAP_PROP_FRAME_HEIGHT);
    // limit the exposure time
    // _cam.set(cv::CAP_PROP_EXPOSURE, 40);
    logger->debug("width:{}, height:{}", _frame_size.width, _frame_size.height);

    {
        std::lock_guard<std::mutex> lg(v_mutex);
        _cam >> _frame;
    }
    logger->debug("====> SimpleCamera read to _frame");

    if (_frame.empty()) {
        logger->error("_frame is empty!!!");
        _frm_ret = false;
    } else {
        _is_running = true;
        _frm_ret = true;
    }

    return _frm_ret;
} // SimpleCamera::init_camera


void SimpleCamera::update_camera() {
    auto logger = spd::get("toy");

    while (_is_running) {
        cv::Mat img_buf;
        _cam >> img_buf;
        if (!img_buf.empty()) {
            std::lock_guard<std::mutex> lg(v_mutex);
            img_buf.copyTo(_frame);
        } else {
            // NOTE:
            img_buf.copyTo(_frame);
            _frm_ret = false;
            logger->warn("img_buf empty!!!");
        }

        std::this_thread::yield();
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
} // SimpleCamera::update_camera
