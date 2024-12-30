#include <opencv2/opencv.hpp>
#include <iostream>

// Stereo parameters structure
struct StereoParams {
    cv::Mat mtxL, distL, R_L, T_L;
    cv::Mat mtxR, distR, R_R, T_R;
    cv::Mat Rot, Trns, Emat, Fmat;
};

// Global variables for mouse callback
cv::Mat originalDisplay, rectifiedDisplay;
int mouseY = -1;

// Mouse callback for drawing dynamic horizontal line
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_MOUSEMOVE) {
        mouseY = y;

        // Redraw Original Images with Line
        cv::Mat originalCopy = originalDisplay.clone();
        cv::line(originalCopy, cv::Point(0, mouseY), cv::Point(originalCopy.cols, mouseY), cv::Scalar(0, 255, 0), 1);
        cv::imshow("Original Images", originalCopy);

        // Redraw Rectified Images with Line
        cv::Mat rectifiedCopy = rectifiedDisplay.clone();
        cv::line(rectifiedCopy, cv::Point(0, mouseY), cv::Point(rectifiedCopy.cols, mouseY), cv::Scalar(0, 255, 0), 1);
        cv::imshow("Rectified Images", rectifiedCopy);
    }
}

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

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./stereo_checkundistorted stereo_image.jpg stereocalibrationfile.yml\n";
        return -1;
    }

    std::string stereoImagePath = argv[1];
    std::string calibrationFile = argv[2];

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

    // Display original images
    cv::hconcat(leftImg, rightImg, originalDisplay);
    cv::imshow("Original Images", originalDisplay);

    // Rectify images
    rectifyStereoImages(params, leftImg, rightImg);

    // Display rectified images
    cv::hconcat(leftImg, rightImg, rectifiedDisplay);
    cv::imshow("Rectified Images", rectifiedDisplay);

    // Set mouse callback
    cv::setMouseCallback("Original Images", onMouse, nullptr);
    cv::setMouseCallback("Rectified Images", onMouse, nullptr);

    cv::waitKey(0);
    return 0;
}
