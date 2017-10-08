#ifndef __HIVE_CAMERA_HPP__
#define __HIVE_CAMERA_HPP__

#include <opencv2/videoio.hpp>

#include <string>
#include <thread>


class SimpleCamera {
public:
    SimpleCamera(std::string& vsrc) : _ret(false), _video_src(vsrc), _is_running(false) {
        // try to open a video camera, through the use of an integer param
        _cam.open(atoi(_video_src.c_str()));
        // if this fails, try to open string as a video file or image sequence
        if (!_cam.isOpened()) {
            _cam.open(_video_src);
        }
    };

    void start_camera();
    void release_camera() {
        _is_running = false;
    };

    cv::Mat* read() {
        return &_frame;
    };

    cv::Size get_frame_width_and_height() {
        return _frame_size;
    };

    inline double get_fps() {
        return _fps;
    };

    inline bool is_running() {
        return _is_running;
    };

private:
    bool init_camera();
    void update_camera();

private:
    cv::VideoCapture _cam;
    cv::Mat _frame;
    double _fps;
    cv::Size _frame_size;
    // unsigned _frame_width;
    // unsigned _frame_height;
    bool _ret;
    std::string _video_src;
    bool _is_running;
    std::thread _worker;
};

#endif // __HIVE_CAMERA_HPP__
