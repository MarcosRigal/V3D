#include <iostream>
#include <exception>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common_code.hpp"

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{i interactive  |      | use interactive mode.}"
    "{p              |0     | Percentage of brightest points used. Default 0 means use the classical white patch method. Values (0, 100) means to use this percentage of brighter pixels. Value 100 means use the gray world method.}"
    "{@input         |<none>| input image.}"
    "{@output        |<none>| output image.}";

/**
 * @brief Application State.
 * Use this structure to maintain the state of the application
 * that will be passed to the callbacks.
 */
struct UserData
{
    cv::Mat in;  // input image.
    cv::Mat out; // output image.
};

/** @brief Standard mouse callback
 * Use this function an argument for cv::setMouseCallback to control the
 * mouse interaction with a window.
 *
 * @arg event says which mouse event (move, push/release a mouse button ...)
 * @arg x and
 * @arg y says where the mouse is.
 * @arg flags give some keyboard state.
 * @arg user_data allow to pass user data to the callback.
 */
void on_mouse(int event, int x, int y, int flags, void *user_data_)
{
    UserData *user_data = static_cast<UserData *>(user_data_);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        user_data->out = fsiv_color_rescaling(user_data->in,
                                              user_data->in.at<cv::Vec3b>(y, x),
                                              cv::Scalar::all(255.0));
        cv::imshow("OUTPUT", user_data->out);
    }
}

/** @brief Standard trackbar callback
 * Use this function an argument for cv::createTrackbar to control
 * the trackbar changes.
 *
 * @arg v give the trackbar position.
 * @arg user_data allow to pass user data to the callback.
 */
void on_change(int v, void *user_data_)
{
    UserData *user_data = static_cast<UserData *>(user_data_);
    std::cout << "Setting p to " << v << "%" << std::endl;
    if (v < 100)
        user_data->out = fsiv_white_patch_color_balance(user_data->in, v);
    else
        user_data->out = fsiv_gray_world_color_balance(user_data->in);
    cv::imshow("OUTPUT", user_data->out);
}

int main(int argc, char *const *argv)
{
    int retCode = EXIT_SUCCESS;

    try
    {

        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Apply a color balance to an image.");
        if (parser.has("help"))
        {
            parser.printMessage();
            return EXIT_SUCCESS;
        }
        bool interactive_mode = parser.has("i");
        int p = parser.get<int>("p");
        if (p < 0 || p > 100)
        {
            std::cerr << "Error: p is out of range [0, 100]." << std::endl;
            return EXIT_FAILURE;
        }
        cv::String input_n = parser.get<cv::String>("@input");
        cv::String output_n = parser.get<cv::String>("@output");
        if (!parser.check())
        {
            parser.printErrors();
            return EXIT_FAILURE;
        }
        UserData user_data;
        user_data.in = cv::imread(input_n, cv::IMREAD_COLOR);
        if (user_data.in.empty())
        {
            std::cerr << "Error: could not open input image." << std::endl;
            return EXIT_FAILURE;
        }

        if (p < 100)
            user_data.out = fsiv_white_patch_color_balance(user_data.in, p);
        else
            user_data.out = fsiv_gray_world_color_balance(user_data.in);

        cv::namedWindow("INPUT");
        cv::namedWindow("OUTPUT");
        if (interactive_mode)
        {
            cv::setMouseCallback("INPUT", on_mouse, &user_data);
            cv::createTrackbar("P", "OUTPUT", &p, 100, on_change,
                               &user_data);
        }
        cv::imshow("INPUT", user_data.in);
        cv::imshow("OUTPUT", user_data.out);
        int k = cv::waitKey(0) & 0xff;
        if (k != 27)
            cv::imwrite(output_n, user_data.out);
    }
    catch (std::exception &e)
    {
        std::cerr << "Capturada excepcion: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
