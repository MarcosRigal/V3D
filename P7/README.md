# Stereo Calibration Tool

This tool performs stereo camera calibration using chessboard images and outputs the calibration results in a `.yml` file.

## Prerequisites

Ensure that you have the following installed:
- CMake (minimum version 3.0)
- OpenCV (minimum version 4.x)
- A C++ compiler (e.g., GCC, Clang)

## Build Instructions

1. Clone the repository and navigate to the project directory.

2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Generate the Makefile using CMake:
   ```bash
   cmake ..
   ```

4. Compile the project:
   ```bash
   make
   ```

## Usage

Once compiled, you can run the stereo calibration tool as follows:

```bash
./stereo_calibrate <path_to_images> <output_file>
```

### Example:

1. Place your stereo images (e.g., `img1.jpg`, `img2.jpg`, ...) in a folder, such as `calibration/`.

2. Run the program with:
   ```bash
   ./stereo_calibrate ../calibration/ ../calibration/out.yml
   ```

This will:
- Read stereo images from `../calibration/`
- Perform stereo calibration
- Save the calibration results to `../calibration/out.yml`

## Output

The output file (e.g., `out.yml`) will contain the following calibration data:
- `LEFT_K`: Intrinsic matrix of the left camera
- `LEFT_D`: Distortion coefficients of the left camera
- `RIGHT_K`: Intrinsic matrix of the right camera
- `RIGHT_D`: Distortion coefficients of the right camera
- `R`: Rotation matrix between the two cameras
- `T`: Translation vector between the two cameras
- `E`: Essential matrix
- `F`: Fundamental matrix

## Notes

- Ensure that the chessboard pattern in your stereo images matches the settings in the code (e.g., `checkerboardSize` and `squareSize`).
- Stereo images must be in grayscale or convertible to grayscale.
- Images should be named and ordered consistently (e.g., `img1.jpg`, `img2.jpg`, ...).