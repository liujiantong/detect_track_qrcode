#include "camera.hpp"
#include "mocktracker.hpp"
#include "spdlog/spdlog.h"

#include <iterator>
#include <iostream>
#include <sstream>
#include <chrono>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "helper.hpp"


namespace spd = spdlog;


void tracking_cb(MockTracker* tracker) {
    auto logger = spd::get("toy");
    cv::Mat* debug_frame = tracker->get_debug_frame();
    cv::Point toy_center = tracker->get_last_toy_center();
    auto code = tracker->get_toy_code();
    logger->info("toy code:{}, code:{}", toy_code_str(code), code.encode());

    cv::imshow("debug frame", *debug_frame);

    direct_pos_t dp = tracker->get_direct_pos();
    logger->info("center:[{},{}], position:[{},{}], direction:{}", toy_center.x, toy_center.y, dp.position.x, dp.position.y, toy_direct_name(dp.direct));

    char key = (char)cv::waitKey(3);
    if (key == 27 || key == 'q' || key == 'Q') {
        tracker->stop_tracking();
    }
}


int main(int argc, char const *argv[]) {
    // auto logger = spd::stdout_color_mt("toy");
    auto logger = spd::rotating_logger_mt("toy", "toy.log", 1048576 * 5, 3);
    logger->set_level(spd::level::info);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");

    // std::string video_src = "0";
    std::string video_src = "../../image/test.avi";
    SimpleCamera camera(video_src);
    camera.start_camera();

    MockTracker tracker(&camera, 30, true);
    tracker.set_tracking_callback(&tracking_cb);
    tracker.track();

    camera.release_camera();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 0;
}
