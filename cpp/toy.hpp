#ifndef __TOY_HPP__
#define __TOY_HPP__

#include <string>
#include <opencv2/opencv.hpp>


typedef enum direct_e {
    EAST_DIR = 0,
    EAST_NORTH_DIR,
    NORTH_DIR,
    WEST_NORTH_DIR,
    WEST_DIR,
    EAST_SOUTH_DIR = -1,
    SOUTH_DIR = -2,
    WEST_SOUTH_DIR = -3,
    UNKNOWN_DIR
} direct_t;

typedef struct direct_pos_s {
    double  angle;
    direct_t direct;
    cv::Point position;
} direct_pos_t;

#define NONE_DERICT_POS {0, UNKNOWN_DIR, cv::Point(0, 0)};

#endif // __TOY_HPP__
