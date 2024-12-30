#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <dirent.h>
#include <sys/types.h>
#include <algorithm>

// Preprocess images
cv::Mat preprocessImage(const cv::Mat &img, bool applyBlur, int blurKernel) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    if (applyBlur) {
        cv::GaussianBlur(gray, gray, cv::Size(blurKernel, blurKernel), 0);
    }

    return gray;
}

// Split stereo image into left and right images
void splitStereoImage(const cv::Mat &stereoImg, cv::Mat &leftImg, cv::Mat &rightImg) {
    int halfWidth = stereoImg.cols / 2;
    leftImg = stereoImg(cv::Rect(0, 0, halfWidth, stereoImg.rows)).clone();
    rightImg = stereoImg(cv::Rect(halfWidth, 0, halfWidth, stereoImg.rows)).clone();
}

// List files in a directory with a specific extension
std::vector<std::string> listFiles(const std::string &path, const std::string &extension) {
    std::vector<std::string> files;
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        perror("❌ Error opening directory");
        return files;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename(entry->d_name);
        if (filename.find(extension) != std::string::npos) {
            files.push_back(path + "/" + filename);
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./stereo_calibrate <image_dir> <output_path>\n";
        return -1;
    }

    std::string imgDir = argv[1];
    std::string outputFile = argv[2];

    cv::Size CheckerBoardSize = {7, 5}; // Checkerboard dimensions
    double SquareSize = 0.02875; // Size of each square in meters

    std::vector<std::string> stereoImages = listFiles(imgDir, ".jpg");

    if (stereoImages.empty()) {
        std::cerr << "❌ No images found in the given directory.\n";
        return -1;
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePointsL, imagePointsR;
    cv::Size referenceImageSize;

    double bestRMS = DBL_MAX;
    double currentRMS;
    cv::Mat bestCameraMatrixL = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat bestDistCoeffsL = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat bestCameraMatrixR = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat bestDistCoeffsR = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat bestR, bestT, bestE, bestF;

    // Debugging Outputs
    std::cout << "🔧 Starting Stereo Calibration Process...\n";

    for (const auto &imagePath : stereoImages) {
        cv::Mat stereoImg = cv::imread(imagePath);
        if (stereoImg.empty()) {
            std::cerr << "❌ Could not load image: " << imagePath << std::endl;
            continue;
        }

        cv::Mat imgL, imgR;
        splitStereoImage(stereoImg, imgL, imgR);

        if (referenceImageSize.empty()) referenceImageSize = imgL.size();
        else if (imgL.size() != referenceImageSize) {
            std::cerr << "❌ Image size mismatch: " << imagePath << std::endl;
            continue;
        }

        cv::Mat grayL = preprocessImage(imgL, true, 5);
        cv::Mat grayR = preprocessImage(imgR, true, 5);

        std::vector<cv::Point2f> cornersL, cornersR;
        bool foundL = cv::findChessboardCornersSB(grayL, CheckerBoardSize, cornersL);
        bool foundR = cv::findChessboardCornersSB(grayR, CheckerBoardSize, cornersR);

        if (foundL && foundR) {
            cv::cornerSubPix(
                grayL, cornersL, cv::Size(5, 5), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.01));
            cv::cornerSubPix(
                grayR, cornersR, cv::Size(5, 5), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.01));

            std::vector<cv::Point3f> obj;
            for (int i = 0; i < CheckerBoardSize.height; ++i)
                for (int j = 0; j < CheckerBoardSize.width; ++j)
                    obj.emplace_back(j * SquareSize, i * SquareSize, 0);

            objectPoints.push_back(obj);
            imagePointsL.push_back(cornersL);
            imagePointsR.push_back(cornersR);
        } else {
            std::cerr << "⚠️ Chessboard not found in image: " << imagePath << std::endl;
        }
    }

    std::cout << "🔄 Performing Stereo Calibration...\n";

    currentRMS = cv::stereoCalibrate(
        objectPoints, imagePointsL, imagePointsR,
        bestCameraMatrixL, bestDistCoeffsL,
        bestCameraMatrixR, bestDistCoeffsR,
        referenceImageSize, bestR, bestT, bestE, bestF,
        cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_SAME_FOCAL_LENGTH,
        cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-6));

    std::cout << "✅ Calibration Completed with RMS Error: " << currentRMS << std::endl;

    // Debugging Output for Calibration Matrices
    std::cout << "🔍 Left Camera Matrix:\n" << bestCameraMatrixL << std::endl;
    std::cout << "🔍 Left Distortion Coefficients:\n" << bestDistCoeffsL << std::endl;
    std::cout << "🔍 Right Camera Matrix:\n" << bestCameraMatrixR << std::endl;
    std::cout << "🔍 Right Distortion Coefficients:\n" << bestDistCoeffsR << std::endl;
    std::cout << "🔍 Rotation Matrix:\n" << bestR << std::endl;
    std::cout << "🔍 Translation Vector:\n" << bestT << std::endl;

    // Save Results
    cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
    fs << "LEFT_K" << bestCameraMatrixL;
    fs << "LEFT_D" << bestDistCoeffsL;
    fs << "RIGHT_K" << bestCameraMatrixR;
    fs << "RIGHT_D" << bestDistCoeffsR;
    fs << "R" << bestR;
    fs << "T" << bestT;
    fs << "E" << bestE;
    fs << "F" << bestF;
    fs.release();

    std::cout << "💾 Results saved to: " << outputFile << std::endl;

    return 0;
}
