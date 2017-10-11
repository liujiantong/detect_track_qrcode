#include "camera.hpp"
#include "spdlog/spdlog.h"

#include <iostream>
#include <chrono>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


namespace spd = spdlog;

void help(const char** av) {
    auto logger = spd::get("toy");
    logger->info("The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer.\n"
                 "Usage:\n{0} <video file, image sequence or device number>\n"
                 "q,Q,esc -- quit\n"
                 "space   -- save frame\n\n"
                 "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*\n"
                 "\texample: {0} 0"
                 "\tYou may also pass a video file instead of a device number\n"
                 "\texample: {{0}} video.avi\n"
                 "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video.\n"
                 "\texample: {0} right%%02d.jpg\n", av[0]);
}


int main(int argc, char const *argv[]) {
    auto logger = spd::stdout_color_mt("toy");

    cv::CommandLineParser parser(argc, argv, "{help h||}{@input||}");
    if (parser.has("help")) {
        help(argv);
        return 0;
    }

    std::string arg = parser.get<std::string>("@input");
    if (arg.empty()) {
        help(argv);
        return 1;
    }

    logger->info("Toy app starting...");

    SimpleCamera camera(arg);
    camera.start_camera();

    string window_name = "video | q or esc to quit";
    logger->info("press space to save a picture. q or esc to quit");
    cv::namedWindow(window_name, cv::WINDOW_KEEPRATIO); //resizable window;

    std::string filename = "frame_cap.png";

    while (true) {
        cv::Mat frame = camera.read();
        cv::imshow(window_name, frame);

        char key = (char) cv::waitKey(10);
        switch (key) {
        case 'q':
        case 'Q':
        case 27: //escape key
            return 0;
        case ' ': //Save an image
            imwrite(filename, frame);
            logger->info("{} saved", filename);
            break;
        default:
            break;
        }
    }

    camera.release_camera();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
