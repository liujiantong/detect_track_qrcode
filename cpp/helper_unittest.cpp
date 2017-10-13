#include "helper.hpp"

#include <iostream>
// #include <gtest/gtest.h>

using namespace std;


int main(int argc, char const *argv[]) {
    cv::Point pt1(11, 10), pt2(10, 10), pt3(11, 11), pt4(11, 9);
    double cosv = angle_cos(pt1, pt2, pt3);
    cout << "cosv:" << cosv << endl;
    double cosv1 = angle_cos(pt1, pt2, pt4);
    cout << "cosv1:" << cosv1 << endl;

    cv::Point tail1(0, 0), head1(1, 1);
    direct_pos_t dp1 = calc_direct(tail1, head1);
    cout << "dp1 direct:" << dp1.direct << endl;
    assert(dp1.direct == EAST_NORTH_DIR);

    cv::Point tail2(0, 0), head2(1, 0);
    direct_pos_t dp2 = calc_direct(tail2, head2);
    cout << "dp2 direct:" << dp2.direct << endl;
    assert(dp2.direct == EAST_DIR);

    cv::Point tail3(10, 10), head3(11, 8);
    direct_pos_t dp3 = calc_direct(tail3, head3);
    cout << "dp3 direct:" << dp3.direct << endl;
    assert(dp3.direct == EAST_SOUTH_DIR);

    return 0;
}
