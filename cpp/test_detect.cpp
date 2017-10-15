#include "detector.hpp"
#include "helper.hpp"
#include "spdlog/spdlog.h"

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto/white_balance.hpp>
#include <algorithm>


namespace spd = spdlog;

int main(int argc, char const *argv[]) {
    auto logger = spd::stdout_color_mt("toy");
    // auto logger = spd::daily_logger_mt("toy", "toy.log", 0, 0);
    logger->set_level(spd::level::debug);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");

    cv::Mat gray;
    cv::Mat image = cv::imread("/Users/liutao/mywork/detect_track_qrcode/image/pic04.jpg");
    cv::Mat m_roi = cv::imread("/Users/liutao/mywork/detect_track_qrcode/image/magenta.png");
    cv::Mat c_roi = cv::imread("/Users/liutao/mywork/detect_track_qrcode/image/cyan.png");
    cv::Mat y_roi = cv::imread("/Users/liutao/mywork/detect_track_qrcode/image/yellow.png");

    cv::Size size = get_frame_size(cv::Size(image.cols, image.rows), 800);
    cv::resize(image, image, size, cv::INTER_AREA);
    // cv::imwrite("resized.png", image);

    logger->debug("cols:{}, rows:{}", image.cols, image.rows);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    ToyDetector detector;

    color_t mroi_color = detector.detect_color(m_roi);
    logger->debug("mroi_color:{}, name:{}", mroi_color, color_name(mroi_color));
    color_t croi_color = detector.detect_color(c_roi);
    logger->debug("croi_color:{}, name:{}", croi_color, color_name(croi_color));
    color_t yroi_color = detector.detect_color(y_roi);
    logger->debug("yroi_color:{}, name:{}", yroi_color, color_name(yroi_color));

    std::vector<std::vector<cv::Point> > founds = detector.find_code_contours(gray);
    logger->debug("founds.size:{}", founds.size());

    if (!founds.empty()) {
        auto wb = cv::xphoto::createSimpleWB();
        wb->balanceWhite(image, image);
        logger->debug("wb created");

        std::vector<cv::Point> cnt;
        toy_code_t code = detector.detect_code_from_contours(image, founds, cnt);
        logger->debug("colors.size:{}, cnt.size:{}", code.colors.size(), cnt.size());
        logger->info("toy_code:{}, code:{}", toy_code_str(code), code.encode());

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
