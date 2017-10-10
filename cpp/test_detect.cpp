#include "detector.hpp"
#include "helper.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto/white_balance.hpp>


int main(int argc, char const *argv[]) {
    cv::Mat gray;
    cv::Mat image = cv::imread("/Users/liutao/mywork/detect_track_qrcode/image/pic01.jpg");

    cv::Size size = get_frame_size(cv::Size(image.cols, image.rows), 800);
    cv::resize(image, image, size, cv::INTER_AREA);
    cv::imwrite("resized.png", image);

    std::cout << "cols:" << image.cols << ", rows:" << image.rows << std::endl;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    ToyDetector detector;
    std::vector<std::vector<cv::Point> > founds = detector.find_code_contours(gray);
    std::cout << "founds.size:" << founds.size() << std::endl;

    for (int i=0; i<founds.size(); i++) {
        cv::drawContours(image, founds, i, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("draw_cnts.png", image);

    if (!founds.empty()) {
        auto wb = cv::xphoto::createSimpleWB();
        wb->balanceWhite(image, image);
        std::cout << "wb created" << std::endl;

        std::vector<cv::Point> cnt;
        std::vector<std::string> colors = detector.detect_color_from_contours(image, founds, cnt);
        std::cout << "colors.size:" << colors.size() << ", cnt.size:" << cnt.size() << std::endl;

        for (auto c : colors) {
            std::cout << c << std::endl;
        }
        std::cout << "print colors" << std::endl;

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
