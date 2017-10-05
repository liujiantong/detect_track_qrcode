#include "camera.hpp"

#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


void help(const char** av) {
    std::cout << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << std::endl
         << "Usage:\n" << av[0] << " <video file, image sequence or device number>" << std::endl
         << "q,Q,esc -- quit" << std::endl
         << "space   -- save frame" << std::endl << std::endl
         << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << std::endl
         << "\texample: " << av[0] << " 0" << std::endl
         << "\tYou may also pass a video file instead of a device number" << std::endl
         << "\texample: " << av[0] << " video.avi" << std::endl
         << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << std::endl
         << "\texample: " << av[0] << " right%%02d.jpg" << std::endl;
}

int main(int argc, char const *argv[]) {
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

    SimpleCamera camera(arg);
    camera.start_camera();

    string window_name = "video | q or esc to quit";
    std::cout << "press space to save a picture. q or esc to quit" << std::endl;
    cv::namedWindow(window_name, cv::WINDOW_KEEPRATIO); //resizable window;

    while (true) {
        cv::Mat* p_frame = camera.read();
        cv::imshow(window_name, *p_frame);

        char key = (char) cv::waitKey(10);

        switch (key) {
        case 'q':
        case 'Q':
        case 27: //escape key
            return 0;
        default:
            break;
        }
    }

    return 0;
}
