# Prerequisites

Ensure that you have the following installed:
- CMake (minimum version 3.0)
- OpenCV (minimum version 4.x)
- A C++ compiler (e.g., GCC, Clang)

## Build Instructions

1. Clone the repository and navigate to the project directory.

2. Build Project:
   ```bash
   mkdir build; cd build; cmake ..; make
   ```

## Usage

Run the compiled binary with the following command:

   ```bash
   ./stereo_sparse stereo_image.jpg calibration.yml out.obj
   ```
