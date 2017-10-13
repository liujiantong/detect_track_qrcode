#include "helper.hpp"

#include <iostream>
#include <CppUTest/TestHarness.h>
#include <CppUTest/CommandLineTestRunner.h>


using namespace std;

TEST_GROUP(HelperTest) {
    void setup() {
      // Init stuff
    }

    void teardown() {
      // Uninit stuff
    }
};

TEST(HelperTest, AngleCos) {
    cv::Point pt1(11, 10), pt2(10, 10), pt3(11, 11), pt4(11, 9);
    double cosv = angle_cos(pt1, pt2, pt3);
    double cosv1 = angle_cos(pt1, pt2, pt4);
    CHECK_EQUAL(cosv, cosv1);
}

TEST(HelperTest, CalcDirect) {
    cv::Point tail1(0, 0), head1(1, 1);
    direct_pos_t dp1 = calc_direct(tail1, head1);
    cout << "dp1 direct:" << dp1.direct << endl;
    CHECK_EQUAL(dp1.direct, EAST_NORTH_DIR);

    cv::Point tail2(0, 0), head2(1, 0);
    direct_pos_t dp2 = calc_direct(tail2, head2);
    cout << "dp2 direct:" << dp2.direct << endl;
    CHECK_EQUAL(dp2.direct, EAST_DIR);

    cv::Point tail3(10, 10), head3(11, 8);
    direct_pos_t dp3 = calc_direct(tail3, head3);
    cout << "dp3 direct:" << dp3.direct << endl;
    CHECK_EQUAL(dp3.direct, EAST_SOUTH_DIR);
}


int main(int argc, char const *argv[]) {
    return CommandLineTestRunner::RunAllTests(argc, argv);
}
