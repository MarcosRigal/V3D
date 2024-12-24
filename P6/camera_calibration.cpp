#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

// Function for preprocessing
cv::Mat preprocessImage(const cv::Mat &img, bool applyBlur, int blurKernel) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray); // Enhance contrast

    if (applyBlur) {
        cv::GaussianBlur(gray, gray, cv::Size(blurKernel, blurKernel), 0);
    }

    return gray;
}

// Function to show image for debugging
void showImage(const std::string &windowName, const cv::Mat &img) {
    cv::imshow(windowName, img);
    cv::waitKey(10); // Brief pause
}

int main() {
    const int boardWidth = 8;  // Inner corners per row
    const int boardHeight = 5; // Inner corners per column
    cv::Size boardSize(boardWidth, boardHeight);

    std::vector<std::string> images = {
        "./P6/calibration/cal01.jpg",
        "./P6/calibration/cal02.jpg",
        "./P6/calibration/cal03.jpg",
        "./P6/calibration/cal04.jpg",
        "./P6/calibration/cal05.jpg",
        "./P6/calibration/cal06.jpg",
        "./P6/calibration/cal07.jpg"
    };

    cv::Size referenceImageSize;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    // Parameter Ranges
    std::vector<float> squareSizes = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<int> blurKernels = {3, 5, 7};
    std::vector<int> cornerSubPixWinSizes = {3, 5, 7};
    std::vector<bool> applyBlurs = {true, false};

    double bestRMS = DBL_MAX;
    float bestSquareSize;
    bool bestApplyBlur;
    int bestBlurKernel;
    int bestSubPixWinSize;
    cv::Mat bestCameraMatrix;
    cv::Mat bestDistCoeffs;

    for (float squareSize : squareSizes) {
        for (bool applyBlur : applyBlurs) {
            for (int blurKernel : blurKernels) {
                for (int subPixWinSize : cornerSubPixWinSizes) {
                    std::cout << "\nTesting Params: SquareSize=" << squareSize
                              << ", Blur=" << (applyBlur ? "Yes" : "No")
                              << ", Blur Kernel=" << blurKernel
                              << ", SubPix Window=" << subPixWinSize << "\n";

                    objectPoints.clear();
                    imagePoints.clear();
                    int successfulDetections = 0;

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

                        std::vector<cv::Point2f> corners;
                        cv::Mat gray = preprocessImage(img, applyBlur, blurKernel);

                        bool found = cv::findChessboardCorners(
                            gray, boardSize, corners,
                            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
                        );

                        if (found) {
                            cv::cornerSubPix(
                                gray, corners, cv::Size(subPixWinSize, subPixWinSize), cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 0.001)
                            );

                            std::vector<cv::Point3f> obj;
                            for (int i = 0; i < boardHeight; i++) {
                                for (int j = 0; j < boardWidth; j++) {
                                    obj.emplace_back(j * squareSize, i * squareSize, 0);
                                }
                            }

                            objectPoints.push_back(obj);
                            imagePoints.push_back(corners);
                            successfulDetections++;
                        }
                    }

                    // Print number of successful corner detections
                    std::cout << "✅ Corners found in " << successfulDetections << " out of " 
                              << images.size() << " images for this combination.\n";

                    if (successfulDetections == 0) {
                        std::cout << "❌ Skipping calibration due to zero successful detections.\n";
                        continue;
                    }

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

    std::cout << "\n🏆 Best Parameters Found:\n";
    std::cout << "  Square Size: " << bestSquareSize << "\n";
    std::cout << "  Apply Blur: " << (bestApplyBlur ? "Yes" : "No") << "\n";
    std::cout << "  Blur Kernel: " << bestBlurKernel << "\n";
    std::cout << "  SubPix Window: " << bestSubPixWinSize << "\n";
    std::cout << "  Best RMS Error: " << bestRMS << "\n";

    cv::FileStorage fs("./P6/calibration/intrinsics.yml", cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "CameraMatrix" << bestCameraMatrix;
        fs << "DistCoeffs" << bestDistCoeffs;
        fs << "RMS" << bestRMS;
        fs.release();
        std::cout << "✅ Calibration data saved to intrinsics.yml\n";
    } else {
        std::cerr << "❌ Failed to save calibration data to intrinsics.yml\n";
    }

    cv::destroyAllWindows();
    return 0;
}
