#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

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

void splitStereoImage(const cv::Mat &stereoImg, cv::Mat &leftImg, cv::Mat &rightImg) {
    int halfWidth = stereoImg.cols / 2;
    leftImg = stereoImg(cv::Rect(0, 0, halfWidth, stereoImg.rows)).clone();
    rightImg = stereoImg(cv::Rect(halfWidth, 0, halfWidth, stereoImg.rows)).clone();
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <output_file.yml>\n";
        return -1;
    }

    std::string imgDir     = argv[1];
    std::string outputFile = argv[2];

    cv::Size checkerboardSize = {7, 5};
    double squareSize = 0.02875;

    std::vector<std::string> stereoImages;
    readImages(imgDir, stereoImages);

    std::sort(stereoImages.begin(), stereoImages.end());

    if (stereoImages.empty()) {
        std::cerr << "❌ No images found in the specified directory.\n";
        return -1;
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePointsL, imagePointsR;

    cv::Size referenceImageSize;

    std::cout << "🔧 Starting stereo calibration process...\n";

    for (const auto &imagePath : stereoImages) {
        cv::Mat stereoImg = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (stereoImg.empty()) {
            std::cerr << "❌ Could not load image: " << imagePath << std::endl;
            continue;
        }

        cv::Mat grayL, grayR;
        splitStereoImage(stereoImg, grayL, grayR);

        if (referenceImageSize.empty()) {
            referenceImageSize = grayL.size();
        } else if (grayL.size() != referenceImageSize) {
            std::cerr << "❌ Size inconsistency in: " << imagePath << std::endl;
            continue;
        }

        std::vector<cv::Point2f> cornersL, cornersR;
        bool foundL = cv::findChessboardCorners(
            grayL, checkerboardSize, cornersL,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
        );
        bool foundR = cv::findChessboardCorners(
            grayR, checkerboardSize, cornersR,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
        );

        if (foundL && foundR) {
            cv::cornerSubPix(
                grayL,
                cornersL,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 60, 1e-6)
            );

            cv::cornerSubPix(
                grayR,
                cornersR,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 60, 1e-6)
            );

            std::vector<cv::Point3f> obj;
            obj.reserve(checkerboardSize.width * checkerboardSize.height);
            for (int i = 0; i < checkerboardSize.height; ++i) {
                for (int j = 0; j < checkerboardSize.width; ++j) {
                    obj.emplace_back(j * squareSize, i * squareSize, 0);
                }
            }

            objectPoints.push_back(obj);
            imagePointsL.push_back(cornersL);
            imagePointsR.push_back(cornersR);
        } else {
            std::cerr << "⚠️ Chessboard not found in image: " << imagePath << std::endl;
        }
    }

    if (objectPoints.empty()) {
        std::cerr << "❌ No chessboard corners detected in any image.\n";
        return -1;
    }

    cv::Mat cameraMatrixL = cv::initCameraMatrix2D(objectPoints, imagePointsL, referenceImageSize, 0);
    cv::Mat distCoeffsL   = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat cameraMatrixR = cv::initCameraMatrix2D(objectPoints, imagePointsR, referenceImageSize, 0);
    cv::Mat distCoeffsR   = cv::Mat::zeros(1, 5, CV_64F);

    cv::Mat R, T, E, F;

    std::cout << "🔄 Performing stereo calibration...\n";

    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 60, 1e-6);

    double rms = cv::stereoCalibrate(
        objectPoints,
        imagePointsL,
        imagePointsR,
        cameraMatrixL, distCoeffsL,
        cameraMatrixR, distCoeffsR,
        referenceImageSize,
        R, T, E, F,
        cv::CALIB_USE_INTRINSIC_GUESS,
        criteria
    );

    std::cout << "✅ Calibration completed. RMS: " << rms << std::endl;

    std::cout << "🔍 Left camera matrix (LEFT_K):\n" << cameraMatrixL << std::endl;
    std::cout << "🔍 Left distortion coefficients (LEFT_D):\n" << distCoeffsL << std::endl;
    std::cout << "🔍 Right camera matrix (RIGHT_K):\n" << cameraMatrixR << std::endl;
    std::cout << "🔍 Right distortion coefficients (RIGHT_D):\n" << distCoeffsR << std::endl;
    std::cout << "🔍 Rotation matrix (R):\n" << R << std::endl;
    std::cout << "🔍 Translation vector (T):\n" << T << std::endl;

    cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "❌ Error creating output file: " << outputFile << std::endl;
        return -1;
    }

    fs << "LEFT_K" << cameraMatrixL;
    fs << "LEFT_D" << distCoeffsL;
    fs << "RIGHT_K" << cameraMatrixR;
    fs << "RIGHT_D" << distCoeffsR;
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;

    fs.release();

    std::cout << "💾 Results saved in: " << outputFile << std::endl;
    return 0;
}
