cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

option( COMPILE_TESTS "Compile the tests" OFF )

find_package( PkgConfig REQUIRED )
pkg_search_module( CPPUTEST REQUIRED cpputest>=3.8 )

project( detect_track_qrcode )
find_package( OpenCV REQUIRED )

set( TOY_LIB_NAME toy )
set(TOY_LIB_SOURCE
    mocktracker.cpp
    camera.cpp
    detector.cpp
    helper.cpp
)

add_library( ${TOY_LIB_NAME} ${TOY_LIB_SOURCE} )
target_link_libraries( ${TOY_LIB_NAME} ${OpenCV_LIBS} )

# add_executable( video_cap video_cap.cpp )
# add_executable( testapp testapp.cpp )
# target_link_libraries( video_cap ${OpenCV_LIBS} )
# target_link_libraries( testapp ${TOY_LIB_NAME} )

add_executable( toyapp toyapp.cpp )
add_executable( test_detect test_detect.cpp )

target_link_libraries( toyapp ${TOY_LIB_NAME} )
target_link_libraries( test_detect ${TOY_LIB_NAME} )

if( COMPILE_TESTS )
    add_executable( helper_unittest helper_unittest.cpp )
    target_link_libraries( helper_unittest ${TOY_LIB_NAME} ${CPPUTEST_LDFLAGS} )
endif()
