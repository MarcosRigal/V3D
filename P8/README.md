# Stereo Rectification Viewer

This tool visually checks the quality of stereo rectification using a stereo image and a pre-generated stereo camera calibration file.

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

Run the compiled binary with the following command:

   ```bash
   ./stereo_checkundistorted ../calibration/m001.jpg ../stereoparms.yml
   ```

- **`stereo_image.jpg`**: Path to the stereo image containing the left and right views concatenated side by side.  
  - The left view occupies the left half, and the right view occupies the right half of the image.
  - Ensure the total width of the stereo image is twice the width of a single view (e.g., 1280×480 for two 640×480 views).

- **`stereocalibration.yml`**: Path to the calibration file in YAML format containing the required stereo calibration parameters.