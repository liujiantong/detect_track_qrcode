cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

enable_testing()
find_package(PkgConfig REQUIRED)
pkg_search_module(CPPUTEST REQUIRED cpputest>=3.8)

project( detect_track_qrcode )
find_package( OpenCV REQUIRED )

# add_executable( video_cap video_cap.cpp )
# add_executable( testapp testapp.cpp camera.cpp )
# target_link_libraries( video_cap ${OpenCV_LIBS} )
# target_link_libraries( testapp ${OpenCV_LIBS} )

add_executable( toyapp toyapp.cpp mocktracker.cpp camera.cpp detector.cpp helper.cpp )
add_executable( test_detect test_detect.cpp detector.cpp helper.cpp )
add_executable( helper_unittest helper_unittest.cpp helper.cpp )

target_link_libraries( toyapp ${OpenCV_LIBS} )
target_link_libraries( test_detect ${OpenCV_LIBS} )
target_link_libraries( helper_unittest ${OpenCV_LIBS} ${CPPUTEST_LDFLAGS})
