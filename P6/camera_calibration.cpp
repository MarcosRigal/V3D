#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

void readImages(const std::string& dirPath, std::vector<std::string>& imagePaths) {
    std::string adjustedDir = dirPath;
    if (!adjustedDir.empty() && adjustedDir.back() != '/') {
        adjustedDir += "/";
    }
    
    cv::glob(adjustedDir + "*.jpg", imagePaths);
    
    for (const auto& path : imagePaths) {
        std::cout << "Found image: " << path << std::endl;
    }
}

// Simple function to preprocess an image (convert to grayscale, equalize histogram, optional Gaussian blur)
cv::Mat preprocessImage(const cv::Mat &img, bool applyBlur, int blurKernel) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    if (applyBlur) {
        cv::GaussianBlur(gray, gray, cv::Size(blurKernel, blurKernel), 0);
    }
    return gray;
}

int main(int argc, char** argv) {
    // We expect exactly 2 arguments: <image_directory> <output_file.yml>
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <output_file.yml>\n";
        return -1;
    }

    // Read all .jpg images from the specified directory
    std::vector<std::string> images;
    readImages(argv[1], images);

    // If no images found, exit
    if (images.empty()) {
        std::cerr << "No .jpg images found in directory: " << argv[1] << std::endl;
        return -1;
    }

    // The calibration board dimensions (inner corners)
    const int boardWidth = 8;
    const int boardHeight = 5;
    cv::Size boardSize(boardWidth, boardHeight);

    // We'll store the first image's size to ensure all images match
    cv::Size referenceImageSize;

    // Vectors to store 3D points (in "object" or real-world coordinates)
    // and corresponding 2D points (in image coordinates)
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    // Ranges of parameters to test
    std::vector<float> squareSizes = {1.0, 2.0, 3.0, 4.0, 5.0};     // The real size of each square on the chessboard
    std::vector<int> blurKernels = {3, 5, 7};                       // Kernel sizes for Gaussian blur
    std::vector<int> cornerSubPixWinSizes = {3, 5, 7};              // Window sizes for cornerSubPix
    std::vector<bool> applyBlurs = {true, false};                   // Whether or not to apply Gaussian blur

    double bestRMS = DBL_MAX;
    float bestSquareSize = 0.0f;
    bool bestApplyBlur = false;
    int bestBlurKernel = 0;
    int bestSubPixWinSize = 0;
    cv::Mat bestCameraMatrix;
    cv::Mat bestDistCoeffs;

    // Try all combinations of the given parameters
    for (float squareSize : squareSizes) {
        for (bool applyBlur : applyBlurs) {
            for (int blurKernel : blurKernels) {
                for (int subPixWinSize : cornerSubPixWinSizes) {
                    std::cout << "\nTesting Params: SquareSize=" << squareSize
                              << ", Blur=" << (applyBlur ? "Yes" : "No")
                              << ", Blur Kernel=" << blurKernel
                              << ", SubPix Window=" << subPixWinSize << "\n";

                    // Clear vectors from previous iteration
                    objectPoints.clear();
                    imagePoints.clear();
                    int successfulDetections = 0;
                    referenceImageSize = cv::Size(); // reset reference size for each combo

                    // Process each image
                    for (const auto &imagePath : images) {
                        cv::Mat img = cv::imread(imagePath);
                        if (img.empty()) {
                            std::cerr << "❌ Failed to open image: " << imagePath << std::endl;
                            continue;
                        }

                        cv::Size imageSize = img.size();
                        if (referenceImageSize.empty()) {
                            referenceImageSize = imageSize;
                        } else if (imageSize != referenceImageSize) {
                            std::cerr << "❌ Image size mismatch for: " << imagePath << std::endl;
                            continue;
                        }

                        // Convert to grayscale, optionally blur
                        cv::Mat gray = preprocessImage(img, applyBlur, blurKernel);

                        // Find chessboard corners
                        std::vector<cv::Point2f> corners;
                        bool found = cv::findChessboardCorners(
                            gray, boardSize, corners,
                            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
                        );

                        if (found) {
                            // Refine corner locations
                            cv::cornerSubPix(
                                gray, corners, cv::Size(subPixWinSize, subPixWinSize), cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 0.001)
                            );

                            // Generate the 3D points for the current board configuration
                            std::vector<cv::Point3f> obj;
                            for (int i = 0; i < boardHeight; i++) {
                                for (int j = 0; j < boardWidth; j++) {
                                    obj.emplace_back(j * squareSize, i * squareSize, 0);
                                }
                            }

                            objectPoints.push_back(obj);
                            imagePoints.push_back(corners);
                            successfulDetections++;

                            cv::drawChessboardCorners(img, boardSize, corners, found);                           
                            cv::imshow("Corners Found", img);
                            cv::waitKey(300);
                        }
                    }

                    std::cout << "✅ Corners found in " << successfulDetections
                              << " out of " << images.size() << " images for this combination.\n";

                    if (successfulDetections == 0) {
                        std::cout << "❌ Skipping calibration due to zero successful detections.\n";
                        continue;
                    }

                    // Camera calibration
                    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
                    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
                    std::vector<cv::Mat> rvecs, tvecs;

                    double rms = cv::calibrateCamera(
                        objectPoints, imagePoints, referenceImageSize,
                        cameraMatrix, distCoeffs, rvecs, tvecs,
                        cv::CALIB_RATIONAL_MODEL,
                        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-5)
                    );

                    std::cout << "📊 RMS Error for this combination: " << rms << "\n";

                    // Keep track of the best combination
                    if (rms < bestRMS) {
                        bestRMS = rms;
                        bestSquareSize = squareSize;
                        bestApplyBlur = applyBlur;
                        bestBlurKernel = blurKernel;
                        bestSubPixWinSize = subPixWinSize;
                        bestCameraMatrix = cameraMatrix.clone();
                        bestDistCoeffs = distCoeffs.clone();
                    }
                }
            }
        }
    }

    // Print out the best parameters found
    std::cout << "\n🏆 Best Parameters Found:\n";
    std::cout << "  Square Size: " << bestSquareSize << "\n";
    std::cout << "  Apply Blur: " << (bestApplyBlur ? "Yes" : "No") << "\n";
    std::cout << "  Blur Kernel: " << bestBlurKernel << "\n";
    std::cout << "  SubPix Window: " << bestSubPixWinSize << "\n";
    std::cout << "  Best RMS Error: " << bestRMS << "\n";

    // Save calibration data to the specified file
    cv::FileStorage fs(argv[2], cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "CameraMatrix" << bestCameraMatrix;
        fs << "DistCoeffs" << bestDistCoeffs;
        fs << "RMS" << bestRMS;
        fs.release();
        std::cout << "✅ Calibration data saved to " << argv[2] << "\n";
    } else {
        std::cerr << "❌ Failed to save calibration data to " << argv[2] << "\n";
    }

    cv::destroyAllWindows();
    return 0;
}
