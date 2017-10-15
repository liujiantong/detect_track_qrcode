#include "detector.hpp"
#include "helper.hpp"
#include "spdlog/spdlog.h"

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto/white_balance.hpp>
#include <algorithm>


namespace spd = spdlog;


#if 0
void calc_color(cv::Mat& img) {
    auto logger = spd::get("toy");

    cv::Mat hsv, mask, hist;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 20, 0), cv::Scalar(180, 255, 255), mask);

    int hsize = 180;
    int channels[] = {0};
    float hue_range[] = { 0.0f, 180.0f };
    const float* ranges[] = { hue_range };

    cv::calcHist(&hsv, 1, channels, mask, hist, 1, &hsize, ranges, true);
    cv::normalize(hist, hist, 0.0, 1.0, cv::NORM_MINMAX);

    double rval = sum_histogram(hist, RED_RANGE1) + sum_histogram(hist, RED_RANGE2);
    double gval = sum_histogram(hist, GREEN_RANGE);
    double bval = sum_histogram(hist, BLUE_RANGE);

    logger->info("r:{}, g:{}, b:{}", rval, gval, bval);
}
#endif


int main(int argc, char const *argv[]) {
    auto logger = spd::stdout_color_mt("toy");
    // auto logger = spd::daily_logger_mt("toy", "toy.log", 0, 0);
    logger->set_level(spd::level::debug);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");

    cv::Mat gray;
    cv::Mat image = cv::imread("/Users/liutao/mywork/detect_track_qrcode/image/pic01.jpg");
    // cv::Mat roi3 = cv::imread("roi3.png");

    cv::Size size = get_frame_size(cv::Size(image.cols, image.rows), 800);
    cv::resize(image, image, size, cv::INTER_AREA);
    cv::imwrite("resized.png", image);

    logger->debug("cols:{}, rows:{}", image.cols, image.rows);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    ToyDetector detector;
    std::vector<std::vector<cv::Point> > founds = detector.find_code_contours(gray);
    logger->debug("founds.size:{}", founds.size());

    if (!founds.empty()) {
        auto wb = cv::xphoto::createSimpleWB();
        wb->balanceWhite(image, image);
        logger->debug("wb created");

        std::vector<cv::Point> cnt;
        toy_code_t code = detector.detect_color_from_contours(image, founds, cnt);
        logger->debug("colors.size:{}, cnt.size:{}", code.colors.size(), cnt.size());
        logger->debug("toy_code:{}", toy_code_str(code));

        if (!cnt.empty()) {
            std::vector<std::vector<cv::Point> > cnts = {cnt};
            cv::drawContours(image, cnts, 0, cv::Scalar(0, 0, 255), 2);
            cv::imshow("color detector", image);
            cv::imwrite("detect_output.png", image);

            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    return 0;
}
