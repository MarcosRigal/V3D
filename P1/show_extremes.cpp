#include <iostream>
#include <exception>

//OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/calib3d.hpp> //Uncomment when it was appropiated.
//#include <opencv2/ml.hpp> //Uncomment when it was appropiated.


#include "common_code.hpp"

const char * keys =
    "{help h usage ? |      | print this message}"
    "{w              |20    | Wait time (miliseconds) between frames.}"
    "{v              |      | the input is a video file.}"
    "{c              |      | the input is a camera index.}"    
    "{@input         |<none>| input <fname|int>}"
    ;


void process_frame(cv::Mat &frame) {
    std::vector<double> min_v, max_v;
    std::vector<cv::Point> min_loc, max_loc;

    fsiv_find_min_max_loc_2(frame, min_v, max_v, min_loc, max_loc);

    for (size_t i = 0; i < min_v.size(); ++i) {
        cv::circle(frame, min_loc[i], 5, cv::Scalar(255, 0, 0), -1);
        cv::putText(frame, "Min: " + std::to_string(min_v[i]),
                    min_loc[i] + cv::Point(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 0, 0), 1);

        cv::circle(frame, max_loc[i], 5, cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, "Max: " + std::to_string(max_v[i]),
                    max_loc[i] + cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
    }
}

int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;
  
  try {    

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Show the extremes values and their locations.");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }
      bool is_video = parser.has("v");
      bool is_camera = parser.has("c");
      int wait = parser.get<int>("w");
      cv::String input = parser.get<cv::String>("@input");
      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

      cv::VideoCapture cap;
      if (is_camera) {
          int camera_index = std::stoi(input);
          if (!cap.open(camera_index)) {
              std::cerr << "Error: Could not open the camera." << std::endl;
              return EXIT_FAILURE;
          }
      } else {
          if (!cap.open(input)) {
              std::cerr << "Error: Could not open the file." << std::endl;
              return EXIT_FAILURE;
          }
      }
  
      cv::Mat frame;
      if (is_camera || is_video) {
          while (cap.read(frame)) {
              if (frame.empty()) {
                  std::cerr << "Error: Empty frame received." << std::endl;
                  break;
              }
  
              process_frame(frame);
  
              cv::imshow("Extremes", frame);
  
              char key = cv::waitKey(30);
              if (key == 'q' || key == 27)
                  break;
          }
      } else {
          cap >> frame;
          if (frame.empty()) {
              std::cerr << "Error: Could not read the image." << std::endl;
              return EXIT_FAILURE;
          }
  
          process_frame(frame);
  
          cv::imshow("Extremes", frame);
          cv::waitKey(0);
      }
  
      cap.release();
      cv::destroyAllWindows();
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception!" << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
