CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(edge_detector)
ENABLE_LANGUAGE(CXX)

# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")

# Find OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Define executables
add_executable(stereo_disparity stereo_disparity.cpp)

# Link OpenCV libraries
target_link_libraries(stereo_disparity ${OpenCV_LIBS})

# Optional: Set output names if needed
set_target_properties(stereo_disparity PROPERTIES OUTPUT_NAME "stereo_disparity")
