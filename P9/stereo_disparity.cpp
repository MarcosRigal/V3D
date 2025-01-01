#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <iostream>
#include <vector>

// Stereo parameters structure
struct StereoParams {
    cv::Mat mtxL, distL, R_L, T_L;
    cv::Mat mtxR, distR, R_R, T_R;
    cv::Mat Rot, Trns, Emat, Fmat;
};

// Load stereo calibration parameters
bool loadStereoCalibration(const std::string &filename, StereoParams &params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "❌ Error opening stereo calibration file.\n";
        return false;
    }

    fs["LEFT_K"] >> params.mtxL;
    fs["LEFT_D"] >> params.distL;
    fs["RIGHT_K"] >> params.mtxR;
    fs["RIGHT_D"] >> params.distR;
    fs["R"] >> params.Rot;
    fs["T"] >> params.Trns;
    fs["E"] >> params.Emat;
    fs["F"] >> params.Fmat;

    fs.release();
    return true;
}

// Rectify stereo images
void rectifyStereoImages(const StereoParams &sti, cv::Mat &left, cv::Mat &right) {
    cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;
    cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
    cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

    cv::stereoRectify(
        sti.mtxL, sti.distL, sti.mtxR, sti.distR,
        left.size(), sti.Rot, sti.Trns,
        rect_l, rect_r, proj_mat_l, proj_mat_r,
        Q, cv::CALIB_ZERO_DISPARITY, 0);

    cv::initUndistortRectifyMap(
        sti.mtxL, sti.distL, rect_l, proj_mat_l,
        left.size(), CV_16SC2, Left_Stereo_Map1, Left_Stereo_Map2);

    cv::initUndistortRectifyMap(
        sti.mtxR, sti.distR, rect_r, proj_mat_r,
        right.size(), CV_16SC2, Right_Stereo_Map1, Right_Stereo_Map2);

    cv::remap(left, left, Left_Stereo_Map1, Left_Stereo_Map2,
              cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
    cv::remap(right, right, Right_Stereo_Map1, Right_Stereo_Map2,
              cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
}

// Calculate disparity using StereoBM
cv::Mat calculateDisparity(const cv::Mat &left, const cv::Mat &right) {
    // Ensure images are grayscale
    cv::Mat grayLeft, grayRight;
    if (left.channels() == 3) {
        cv::cvtColor(left, grayLeft, cv::COLOR_BGR2GRAY);
    } else {
        grayLeft = left.clone();
    }
    
    if (right.channels() == 3) {
        cv::cvtColor(right, grayRight, cv::COLOR_BGR2GRAY);
    } else {
        grayRight = right.clone();
    }

    // Create StereoBM object and compute disparity
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 15); // numDisparities, blockSize
    cv::Mat disparity;
    stereo->compute(grayLeft, grayRight, disparity);

    disparity.convertTo(disparity, CV_32F, 1.0 / 16.0); // Normalize disparity
    return disparity;
}

// Generate 3D Points
std::vector<cv::Point3f> generatePointCloud(const cv::Mat &disparity, const StereoParams &params) {
    std::vector<cv::Point3f> points;

    float fx = params.mtxL.at<double>(0, 0);
    float fy = params.mtxL.at<double>(1, 1);
    float cx = params.mtxL.at<double>(0, 2);
    float cy = params.mtxL.at<double>(1, 2);
    float baseline = cv::norm(params.Trns); // Distance between cameras

    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            float d = disparity.at<float>(y, x);
            if (d > 10.0f) { // Threshold disparity
                float Z = (fx * baseline) / d;
                float X = (x - cx) * Z / fx;
                float Y = (y - cy) * Z / fy;
                points.emplace_back(X, Y, Z);
            }
        }
    }

    return points;
}

// Save points to OBJ file
void writeToOBJ(const std::string &path, const std::vector<cv::Point3f> &points) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "❌ Error opening OBJ file for writing.\n";
        return;
    }

    for (const auto &p : points) {
        file << "v " << p.x << " " << p.y << " " << p.z << "\n";
    }

    file.close();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./stereo_disparity stereo_image.jpg calibration.yml out.obj\n";
        return -1;
    }

    std::string stereoImagePath = argv[1];
    std::string calibrationFile = argv[2];
    std::string outputOBJ = argv[3];

    // Load stereo image
    cv::Mat stereoImg = cv::imread(stereoImagePath);
    if (stereoImg.empty()) {
        std::cerr << "❌ Error loading stereo image.\n";
        return -1;
    }

    // Split stereo image into left and right
    cv::Mat leftImg = stereoImg(cv::Rect(0, 0, stereoImg.cols / 2, stereoImg.rows)).clone();
    cv::Mat rightImg = stereoImg(cv::Rect(stereoImg.cols / 2, 0, stereoImg.cols / 2, stereoImg.rows)).clone();

    // Load stereo calibration parameters
    StereoParams params;
    if (!loadStereoCalibration(calibrationFile, params)) {
        return -1;
    }

    // Rectify images
    rectifyStereoImages(params, leftImg, rightImg);

    // Compute disparity map
    cv::Mat disparity = calculateDisparity(leftImg, rightImg);

    // Generate point cloud
    auto points = generatePointCloud(disparity, params);

    // Save to OBJ file
    writeToOBJ(outputOBJ, points);

    std::cout << "✅ OBJ file saved successfully: " << outputOBJ << "\n";

    return 0;
}
