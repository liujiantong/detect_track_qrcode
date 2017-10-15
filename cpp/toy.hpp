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
#define YELLOW_RANGE   cv::Range(15, 45)
#define GREEN_RANGE    cv::Range(45, 75)
#define CYAN_RANGE     cv::Range(75, 105)
#define BLUE_RANGE     cv::Range(105, 135)
#define MAGENTA_RANGE  cv::Range(135, 165)
#define RED_RANGE2     cv::Range(165, 180)


typedef enum color_e {
    UNKNOWN = -1,
    RED = 0,
    YELLOW,
    GREEN,
    CYAN,
    BLUE,
    MAGENTA,
    WHITE
} color_t;

typedef enum shape_e {
    NONE_SHAPE = -1,
    TRIANGLE = 3,
    SQUARE,
    PENTAGON,
    HEXAGON
} shape_t;


/**
 * block shapes: 4, color types: 6
 * block number: 4
 * start block:  shape:[TRIANGLE], color:[ALL].
 *               TRIANGLE only used in start block
 * block1:3:     shape:[SQUARE, PENTAGON, HEXAGON], color:[ALL]
 */
typedef struct toy_code_s {
    std::array<color_t, 4> colors;
    std::array<shape_t, 4> shapes;

    int encode() {
        if (colors.empty() || shapes.empty()) {
            return -1;
        }

        int n_colors = 6, n_shapes = 3;
        int base = n_colors * n_shapes;

        int code_sum = 0;
        int c0 = colors[0];
        if (c0 < RED || c0 > WHITE) return -1;

        // blocks == colors == 4
        for (int b=1; b<4; b++) { // block loop
            if ((colors[b] < RED || colors[b] > WHITE) ||
                (shapes[b] < TRIANGLE || shapes[b] > HEXAGON)) {
                return -1;
            }

            int c = colors[b], s = shapes[b]-3;
            // the value of a given block (shape as row and color as cols)
            int block_val = c + std::pow(n_colors, s);
            code_sum += block_val * std::pow(base, (b-1));
        }

        int start_block_val = c0 * std::pow(base, 3);
        return start_block_val + code_sum;
    }
} toy_code_t;


#endif // __TOY_HPP__
