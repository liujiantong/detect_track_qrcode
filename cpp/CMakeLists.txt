cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project( detect_track_qrcode )
find_package( OpenCV REQUIRED )

# add_executable( video_cap video_cap.cpp )
add_executable( toyapp toyapp.cpp camera.cpp )

# target_link_libraries( video_cap ${OpenCV_LIBS} )
target_link_libraries( toyapp ${OpenCV_LIBS} )
