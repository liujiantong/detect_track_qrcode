#ifndef __HIVE_CAMERA_HPP__
#define __HIVE_CAMERA_HPP__

#include <opencv2/videoio.hpp>

#include <string>
#include <thread>
#include <mutex>


class SimpleCamera {
public:
    SimpleCamera(std::string& vsrc) : _ret(false), _video_src(vsrc), _is_running(false), _worker() {
        // try to open string as a video file or image sequence
        _cam.open(_video_src);
        // if this fails, try to open a video camera, through the use of an integer param
        if (!_cam.isOpened()) {
            _cam.open(atoi(_video_src.c_str()));
        }
    };

    ~SimpleCamera() {
        if (_worker.joinable()) {
            _worker.join();
        }
    };

    void start_camera();
    void release_camera() {
        _is_running = false;
    };

    cv::Mat read() {
        std::lock_guard<std::mutex> lg(v_mutex);
        return _frame;
    };

    cv::Size get_frame_size() {
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
    bool _ret;
    std::string _video_src;
    bool _is_running;
    std::thread _worker;

    std::mutex v_mutex;
};

#endif // __HIVE_CAMERA_HPP__
