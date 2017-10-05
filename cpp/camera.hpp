#ifndef __HIVE_CAMERA_HPP__
#define __HIVE_CAMERA_HPP__

#include <opencv2/videoio.hpp>
#include <thread>


class SimpleCamera {
public:
    // SimpleCamera(unsigned vsrc=cv::CAP_ANY) : _cam(NULL), _ret(false), _video_src(vsrc),
    // _video_file_src(NULL), _video_type(0), _is_running(false), _worker(NULL) {
    //
    // };

    SimpleCamera(std::string& vsrc) : _cam(NULL), _ret(false),
    _video_src(vsrc), _is_running(false) {
        // try to open string, this will attempt to open it as a video file or image sequence
        _cam = new cv::VideoCapture(_video_src);

        // if this fails, try to open as a video camera, through the use of an integer param
        if (!_cam->isOpened()) {
            _cam->open(atoi(_video_src.c_str()));
        }
    };

    void start_camera();
    void release_camera();

    cv::Mat* read() {
        return &_frame;
    };

    void get_frame_width_and_height(unsigned* w, unsigned* h) {
        *w = _frame_width;
        *h = _frame_height;
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
    cv::VideoCapture* _cam;
    cv::Mat _frame;
    double _fps;
    unsigned _frame_width;
    unsigned _frame_height;
    bool _ret;
    std::string _video_src;
    bool _is_running;
    std::thread _worker;
};

#endif // __HIVE_CAMERA_HPP__
