CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(fsiv_tutorial_opencv)
ENABLE_LANGUAGE(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")

FIND_PACKAGE(OpenCV REQUIRED )
LINK_LIBRARIES(${OpenCV_LIBS})
include_directories ("${OpenCV_INCLUDE_DIRS}")

add_executable(show_extremes show_extremes.cpp common_code.cpp common_code.hpp)
add_executable(show_img show_img.cpp)
add_executable(show_video show_video.cpp)
add_executable(comp_stats comp_stats.cpp)
add_executable(fsiv_tutorial_opencv_test_common_code test_common_code.cpp common_code.cpp common_code.hpp)
set_target_properties(fsiv_tutorial_opencv_test_common_code PROPERTIES OUTPUT_NAME "test_common_code")

