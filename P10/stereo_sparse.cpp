#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <vector>

// Stereo parameters structure
struct StereoParams {
    cv::Mat mtxL, distL;
    cv::Mat mtxR, distR;
    cv::Mat Rot, Trns;
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

    fs.release();
    return true;
}

// Rectify stereo images
void rectifyStereoImages(const StereoParams &sti, cv::Mat &left, cv::Mat &right) {
    cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;
    cv::stereoRectify(
        sti.mtxL, sti.distL, sti.mtxR, sti.distR,
        left.size(), sti.Rot, sti.Trns,
        rect_l, rect_r, proj_mat_l, proj_mat_r,
        Q, cv::CALIB_ZERO_DISPARITY, 0);

    cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
    cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

    cv::initUndistortRectifyMap(sti.mtxL, sti.distL, rect_l, proj_mat_l,
        left.size(), CV_16SC2, Left_Stereo_Map1, Left_Stereo_Map2);
    cv::initUndistortRectifyMap(sti.mtxR, sti.distR, rect_r, proj_mat_r,
        right.size(), CV_16SC2, Right_Stereo_Map1, Right_Stereo_Map2);

    cv::remap(left, left, Left_Stereo_Map1, Left_Stereo_Map2,
              cv::INTER_LINEAR);
    cv::remap(right, right, Right_Stereo_Map1, Right_Stereo_Map2,
              cv::INTER_LINEAR);
}

// AKAZE Feature Matching
std::vector<cv::DMatch> matchFeatures(const cv::Mat &left, const cv::Mat &right,
                                      std::vector<cv::KeyPoint> &keypointsLeft,
                                      std::vector<cv::KeyPoint> &keypointsRight) {
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    cv::Mat descriptorsLeft, descriptorsRight;
    detector->detectAndCompute(left, cv::noArray(), keypointsLeft, descriptorsLeft);
    detector->detectAndCompute(right, cv::noArray(), keypointsRight, descriptorsRight);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(descriptorsLeft, descriptorsRight, matches);

    return matches;
}

// Filter matches by horizontal alignment
std::vector<cv::DMatch> filterMatches(const std::vector<cv::DMatch> &matches,
                                      const std::vector<cv::KeyPoint> &keypointsLeft,
                                      const std::vector<cv::KeyPoint> &keypointsRight) {
    std::vector<cv::DMatch> filteredMatches;
    for (const auto &match : matches) {
        auto ptLeft = keypointsLeft[match.queryIdx].pt;
        auto ptRight = keypointsRight[match.trainIdx].pt;
        if (std::abs(ptLeft.y - ptRight.y) < 2.0) { // Check horizontal alignment
            filteredMatches.push_back(match);
        }
    }
    return filteredMatches;
}

// Draw Matches
void drawMatchesVisualization(const cv::Mat &left, const cv::Mat &right,
                              const std::vector<cv::KeyPoint> &keypointsLeft,
                              const std::vector<cv::KeyPoint> &keypointsRight,
                              const std::vector<cv::DMatch> &matches,
                              const std::string &windowName) {
    cv::Mat imgMatches;
    cv::drawMatches(left, keypointsLeft, right, keypointsRight, matches, imgMatches);
    cv::imshow(windowName, imgMatches);
    cv::waitKey(0);
}

// Triangulate Points
std::vector<cv::Point3f> triangulatePoints(const std::vector<cv::DMatch> &matches,
                                          const std::vector<cv::KeyPoint> &keypointsLeft,
                                          const std::vector<cv::KeyPoint> &keypointsRight,
                                          const StereoParams &params) {
    std::vector<cv::Point3f> points3D;
    for (const auto &match : matches) {
        auto ptLeft = keypointsLeft[match.queryIdx].pt;
        auto ptRight = keypointsRight[match.trainIdx].pt;
        cv::Point3f point;
        float disparity = ptLeft.x - ptRight.x;
        if (disparity > 0) {
            point.z = (params.mtxL.at<double>(0, 0) * cv::norm(params.Trns)) / disparity;
            point.x = (ptLeft.x - params.mtxL.at<double>(0, 2)) * point.z / params.mtxL.at<double>(0, 0);
            point.y = (ptLeft.y - params.mtxL.at<double>(1, 2)) * point.z / params.mtxL.at<double>(1, 1);
            points3D.push_back(point);
        }
    }
    return points3D;
}

// Save points to OBJ
void writeToOBJ(const std::string &path, const std::vector<cv::Point3f> &points) {
    std::ofstream file(path);
    for (const auto &p : points) {
        file << "v " << p.x << " " << p.y << " " << p.z << "\n";
    }
    file.close();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./stereo_sparse stereo_image.jpg calibration.yml out.obj\n";
        return -1;
    }

    cv::Mat stereoImg = cv::imread(argv[1]);
    cv::Mat left = stereoImg(cv::Rect(0, 0, stereoImg.cols / 2, stereoImg.rows));
    cv::Mat right = stereoImg(cv::Rect(stereoImg.cols / 2, 0, stereoImg.cols / 2, stereoImg.rows));

    StereoParams params;
    loadStereoCalibration(argv[2], params);
    rectifyStereoImages(params, left, right);

    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    auto matches = matchFeatures(left, right, keypointsLeft, keypointsRight);
    drawMatchesVisualization(left, right, keypointsLeft, keypointsRight, matches, "All Matches");
    
    auto filteredMatches = filterMatches(matches, keypointsLeft, keypointsRight);
    drawMatchesVisualization(left, right, keypointsLeft, keypointsRight, filteredMatches, "Filtered Matches");
    
    auto points3D = triangulatePoints(filteredMatches, keypointsLeft, keypointsRight, params);
    writeToOBJ(argv[3], points3D);
    
    std::cout << "✅ OBJ file saved successfully.";
    return 0;
}

