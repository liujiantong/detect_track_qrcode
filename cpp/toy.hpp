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
    UNKNOWN_DIR = -100
} direct_t;

typedef struct direct_pos_s {
    double  angle;
    direct_t direct;
    cv::Point position;
} direct_pos_t;

#define NONE_DERICT_POS {0, UNKNOWN_DIR, cv::Point(-1, -1)};

/*
H =   0°...30° => RED
H =  30°...90° => Yellow
H =  90°..150° => Green
H = 150°..210° => Cyan
H = 210°..270° => Blue
H = 270°..330° => Magenta
H = 330°..360° => RED

H =   0°...60° => RED
H =  60°..180° => Green
H = 180°..300° => Blue
H = 300°..360° => RED
*/

#define RED_RANGE1     cv::Range(0, 15)
#define YELLOW_RANGE   cv::Range(15, 30)
#define GREEN_RANGE    cv::Range(45, 75)
#define CYAN_RANGE     cv::Range(75, 105)
#define BLUE_RANGE     cv::Range(105, 135)
#define MAGENTA_RANGE  cv::Range(135, 165)
#define RED_RANGE2     cv::Range(165, 180)


typedef enum color_e {
    RED = 0,
    YELLOW,
    GREEN,
    CYAN,
    BLUE,
    MAGENTA
} color_t;

typedef enum shape_e {
    TRIANGLE = 3,
    SQUARE,
    PENTAGON
} shape_t;

typedef struct color_code_s {
    std::array<color_t, 4> colors;
    std::array<shape_t, 4> shapes;
    int encode() {
        // TODO:
        return 0;
    }
} color_code_t;


#endif // __TOY_HPP__
