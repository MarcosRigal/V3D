#include <opencv2/opencv.hpp>
#include <iostream>

// Helper function to draw a cube
void drawCube(cv::Mat& frame, const std::vector<cv::Point2f>& points) {
    // Draw base of the cube
    cv::line(frame, points[0], points[1], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[1], points[2], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[2], points[3], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[3], points[0], cv::Scalar(255, 0, 0), 2);

    // Draw top of the cube
    cv::line(frame, points[4], points[5], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[5], points[6], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[6], points[7], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[7], points[4], cv::Scalar(255, 0, 0), 2);

    // Connect top and bottom
    cv::line(frame, points[0], points[4], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[1], points[5], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[2], points[6], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[3], points[7], cv::Scalar(255, 0, 0), 2);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: augReal size intrinsics.yml videofile" << std::endl;
        return -1;
    }

    // Parse arguments
    float squareSize = std::stof(argv[1]); // Size of axis and cubes
    std::string intrinsicsFile = argv[2];
    std::string videoFile = argv[3];

    // Load camera parameters
    cv::FileStorage fs(intrinsicsFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open intrinsics file: " << intrinsicsFile << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    fs["CameraMatrix"] >> cameraMatrix;
    fs["DistCoeffs"] >> distCoeffs;

    if (cameraMatrix.empty() || distCoeffs.empty()) {
        std::cerr << "Failed to load camera parameters from file!" << std::endl;
        return -1;
    }

    // Open video file
    cv::VideoCapture cap(videoFile);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video file!" << std::endl;
        return -1;
    }

    // Define chessboard pattern
    cv::Size patternSize(8, 5); // Adjust based on your calibration pattern
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            objectPoints.emplace_back(j * squareSize, i * squareSize, 0);
        }
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY); // Ensure grayscale input for cornerSubPix

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(grayFrame, patternSize, corners);

        if (found) {
            cv::cornerSubPix(grayFrame, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1));

            // Estimate pose
            cv::Mat rvec, tvec;
            cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

            // Define 3D axis
            std::vector<cv::Point3f> axis = {
                {0, 0, 0}, 
                {squareSize, 0, 0}, 
                {0, squareSize, 0}, 
                {0, 0, -squareSize}
            };

            std::vector<cv::Point2f> projectedPoints;
            cv::projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

            // Draw the 3D axis
            cv::line(frame, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 3); // X-axis
            cv::line(frame, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 3); // Y-axis
            cv::line(frame, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 3); // Z-axis

            // Draw cubes on black squares
            for (int i = 0; i < patternSize.height - 1; i++) {
                for (int j = 0; j < patternSize.width - 1; j++) {
                    if ((i + j) % 2 == 0) { // Black square
                        std::vector<cv::Point3f> cubePoints = {
                            {j * squareSize, i * squareSize, 0},
                            {(j + 1) * squareSize, i * squareSize, 0},
                            {(j + 1) * squareSize, (i + 1) * squareSize, 0},
                            {j * squareSize, (i + 1) * squareSize, 0},
                            {j * squareSize, i * squareSize, -squareSize},
                            {(j + 1) * squareSize, i * squareSize, -squareSize},
                            {(j + 1) * squareSize, (i + 1) * squareSize, -squareSize},
                            {j * squareSize, (i + 1) * squareSize, -squareSize}
                        };

                        std::vector<cv::Point2f> projectedCubePoints;
                        cv::projectPoints(cubePoints, rvec, tvec, cameraMatrix, distCoeffs, projectedCubePoints);
                        drawCube(frame, projectedCubePoints);
                    }
                }
            }
        }

        cv::imshow("Augmented Reality", frame);
        if (cv::waitKey(30) == 27) break; // Stop on ESC key
    }

    return 0;
}
