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
add_executable(augReal augReal.cpp)
add_executable(camera_calibration camera_calibration.cpp)

# Link OpenCV libraries
target_link_libraries(augReal ${OpenCV_LIBS})
target_link_libraries(camera_calibration ${OpenCV_LIBS})

# Optional: Set output names if needed
set_target_properties(augReal PROPERTIES OUTPUT_NAME "augReal")
set_target_properties(camera_calibration PROPERTIES OUTPUT_NAME "camera_calibration")
