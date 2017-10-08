#include "camera.hpp"
#include "tracker.hpp"
#include "spdlog/spdlog.h"

#include <iterator>
#include <iostream>
#include <sstream>
#include <chrono>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


namespace spd = spdlog;


std::string join(std::vector<std::string>& v) {
    std::ostringstream imploded;
    std::copy(v.begin(), v.end(), std::ostream_iterator<std::string>(imploded, ","));
    return imploded.str();
}


void tracking_cb(ToyTracker* tracker) {
    auto logger = spd::get("console");
    logger->debug("############ tracking_cb start");

    cv::Mat* debug_frame = tracker->get_debug_frame();
    // cv::Point toy_center = tracker->get_last_toy_center();
    auto colors = tracker->get_toy_colors();

    cv::imshow("debug frame", *debug_frame);

    char key = (char) cv::waitKey(3);
    if (key == 27 || key == 'q' || key == 'Q') {
        tracker->stop_tracking();
    }

    logger->debug("toy colors: {}", join(colors));
}


int main(int argc, char const *argv[]) {
    auto logger = spd::stdout_color_mt("console");
    logger->set_level(spd::level::debug);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");

    std::string video_src = "0";
    SimpleCamera camera(video_src);
    camera.start_camera();

    logger->debug("camera started");

    ToyTracker tracker(&camera, 30, true);
    tracker.set_tracking_callback(&tracking_cb);
    tracker.track();

    camera.release_camera();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 0;
}
