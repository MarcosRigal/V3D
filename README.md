# Vision 3D Practices Repository

This repository contains a collection of tools, implementations, and tutorials developed for 3D vision and image processing tasks. The project focuses on practical exercises, including camera calibration, stereo vision, edge detection, and image manipulation.

## Features
- Camera calibration and visualization of 3D objects.
- Stereo camera calibration and rectification.
- Disparity map generation and 3D reconstruction.
- Image processing utilities such as filtering, color conversion, and histogram computation.
- Comprehensive video tutorials for guided learning.

---

## Prerequisites
- CMake (minimum version 3.0)
- OpenCV (minimum version 4.x)
- A C++ compiler (e.g., GCC, Clang)

---

## Build Instructions
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

---

## Key Tools and Applications

### Camera Calibration and 3D Visualization
- Visualizes a 3D axis and places objects on a calibration pattern.
- **Run Commands:**
  ```bash
  ./camera_calibration ../calibration intrinsics.yml
  ./augReal 3 ../intrinsics.yml ../video/augreal.mp4
  ```

### Stereo Calibration and Rectification
- Performs stereo camera calibration and checks rectification quality.
- **Run Commands:**
  ```bash
  ./stereo_calibrate ../calibration/ ../calibration/out.yml
  ./stereo_checkundistorted ../calibration/m001.jpg ../stereoparms.yml
  ```

### Disparity and Sparse 3D Reconstruction
- Generates disparity maps and performs sparse 3D reconstruction.
- **Run Commands:**
  ```bash
  ./stereo_disparity stereo_image.jpg calibration.yml out.obj
  ./stereo_sparse stereo_image.jpg calibration.yml out.obj
  ```

### Image Processing Utilities
- Includes tools for filtering, color balancing, histogram computation, and edge detection.
- Covers advanced image enhancement and transformation techniques.

---

## Video Tutorials
Detailed walkthroughs are available for all tasks and implementations.

### Video Links and Chapters

#### Video 1
[https://youtu.be/99BAqCQkwGk](https://youtu.be/99BAqCQkwGk)
- fsiv_find_min_max_loc_1
- fsiv_find_min_max_loc_2
- show_extremes

#### Video 2
[https://youtu.be/u_YK6rNgAtk](https://youtu.be/u_YK6rNgAtk)
- fsiv_convert_image_byte_to_float
- fsiv_convert_image_float_to_byte
- fsiv_cbg_process

#### Video 3
[https://youtu.be/jsuDr2BYWz4](https://youtu.be/jsuDr2BYWz4)
- fsiv_color_rescaling
- fsiv_gray_world_color_balance
- fsiv_compute_image_histogram

#### Video 4
[https://youtu.be/czKYyx4K6NE](https://youtu.be/czKYyx4K6NE)
- fsiv_create_box_filter
- fsiv_create_gaussian_filter
- fsiv_filter2D

#### Video 5
[https://youtu.be/BMcMNRkbDj4](https://youtu.be/BMcMNRkbDj4)
- fsiv_compute_gradient_magnitude
- fsiv_percentile_edge_detector
- fsiv_canny_edge_detector
