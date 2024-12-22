#include "common_code.hpp"

cv::Mat
fsiv_convert_image_byte_to_float(const cv::Mat &img)
{
    CV_Assert(img.depth() == CV_8U);
    cv::Mat out;
    img.convertTo(out, CV_32F, 1.0 / 255.0);
    
    CV_Assert(out.rows == img.rows && out.cols == img.cols);
    CV_Assert(out.depth() == CV_32F);
    CV_Assert(img.channels() == out.channels());
    return out;
}

cv::Mat
fsiv_convert_image_float_to_byte(const cv::Mat &img)
{
    CV_Assert(img.depth() == CV_32F);
    cv::Mat out;
    img.convertTo(out, CV_8U, 255.0);
    
    CV_Assert(out.rows == img.rows && out.cols == img.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(img.channels() == out.channels());
    return out;
}

cv::Mat
fsiv_convert_bgr_to_hsv(const cv::Mat &img)
{
    CV_Assert(img.channels() == 3);
    cv::Mat out;
    cv::cvtColor(img, out, cv::COLOR_BGR2HSV);
    
    CV_Assert(out.channels() == 3);
    return out;
}

cv::Mat
fsiv_convert_hsv_to_bgr(const cv::Mat &img)
{
    CV_Assert(img.channels() == 3);
    cv::Mat out;
    cv::cvtColor(img, out, cv::COLOR_HSV2BGR);
    
    CV_Assert(out.channels() == 3);
    return out;
}

cv::Mat
fsiv_cbg_process(const cv::Mat &in,
                 double contrast, double brightness, double gamma,
                 bool only_luma)
{
    CV_Assert(in.depth() == CV_8U);
    cv::Mat out;

    if (in.channels() == 1) {
        // Grayscale Processing
        cv::Mat float_img = fsiv_convert_image_byte_to_float(in);

        // Gamma Correction
        cv::pow(float_img, gamma, float_img);

        // Contrast and Brightness
        float_img = float_img.mul(contrast) + brightness;

        // Clamp Values
        cv::min(cv::max(float_img, 0.0), 1.0, float_img);

        out = fsiv_convert_image_float_to_byte(float_img);
    }
    else if (!only_luma) {
        // Process All Channels
        std::vector<cv::Mat> channels;
        cv::split(fsiv_convert_image_byte_to_float(in), channels);

        for (auto &channel : channels) {
            cv::pow(channel, gamma, channel);
            channel = channel * contrast + brightness;
            cv::min(cv::max(channel, 0.0), 1.0, channel);
        }

        cv::merge(channels, out);
        out = fsiv_convert_image_float_to_byte(out);
    }
    else {
        // Process Only Luma (HSV)
        cv::Mat float_hsv;
        in.convertTo(float_hsv, CV_32F, 1.0 / 255.0);
        cv::cvtColor(float_hsv, float_hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> channels;
        cv::split(float_hsv, channels);

        // Process V channel
        cv::pow(channels[2], gamma, channels[2]);
        channels[2] = channels[2] * contrast + brightness;

        cv::min(cv::max(channels[2], 0.0), 1.0, channels[2]);

        cv::merge(channels, float_hsv);
        cv::cvtColor(float_hsv, float_hsv, cv::COLOR_HSV2BGR);
        float_hsv.convertTo(out, CV_8U, 255.0);
    }

    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(out.channels() == in.channels());

    return out;
}
