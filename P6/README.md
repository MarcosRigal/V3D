# Camera Calbration 3D

This program prints a 3D axis on this video showing the calibration pattern and show a cube 1x1x1 on each black square of the calibration pattern

---

## Prerequisites

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

Run the compiled binares with the following commands:

   ```bash
   ./camera_calibration ../calibration intrinsics.yml
   ```

   ```bash
   ./augReal 4 ../intrinsics.yml ../video/augreal.mp4
   ```