#include <opencv2/opencv.hpp>
#include <iostream>

void drawCube(cv::Mat& frame, const std::vector<cv::Point2f>& points) {
    cv::line(frame, points[0], points[1], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[1], points[2], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[2], points[3], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[3], points[0], cv::Scalar(255, 0, 0), 2);

    cv::line(frame, points[4], points[5], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[5], points[6], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[6], points[7], cv::Scalar(255, 0, 0), 2);
    cv::line(frame, points[7], points[4], cv::Scalar(255, 0, 0), 2);

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

    float squareSize = 1.0f;
    float axisScale = std::min(std::stof(argv[1]), 4.0f);

    std::string intrinsicsFile = argv[2];
    std::string videoFile = argv[3];

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

    cv::VideoCapture cap(videoFile);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video file!" << std::endl;
        return -1;
    }

    cv::Size patternSize(8, 5);
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            objectPoints.emplace_back(j * squareSize, i * squareSize, 0);
        }
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(grayFrame, patternSize, corners);

        if (found) {
            cv::cornerSubPix(grayFrame, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1));

            cv::Mat rvec, tvec;
            cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

            std::vector<cv::Point3f> axis = {
                {0, 0, 0}, 
                {squareSize*axisScale, 0, 0}, 
                {0, squareSize*axisScale, 0}, 
                {0, 0, -squareSize*axisScale}
            };

            std::vector<cv::Point2f> projectedPoints;
            cv::projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

            cv::line(frame, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 3);
            cv::line(frame, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 3);
            cv::line(frame, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 3);

            for (int i = 0; i < patternSize.height - 1; i++) {
                for (int j = 0; j < patternSize.width - 1; j++) {
                    if ((i + j) % 2 == 0) {
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
        if (cv::waitKey(30) == 27) break;
    }

    return 0;
}
