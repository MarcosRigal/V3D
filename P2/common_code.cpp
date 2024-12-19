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
    cv::Mat out = fsiv_convert_image_byte_to_float(in);

    if (in.channels() == 3 && only_luma) {
        cv::Mat hsv = fsiv_convert_bgr_to_hsv(out);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);

        // Process only the V (luma) channel.
        hsv_channels[2] = hsv_channels[2] * contrast + brightness;
        cv::pow(hsv_channels[2], gamma, hsv_channels[2]);

        // Merge channels and convert back to BGR.
        cv::merge(hsv_channels, hsv);
        out = fsiv_convert_hsv_to_bgr(hsv);
    } else {
        // Process all channels.
        out = out * contrast + brightness;
        cv::pow(out, gamma, out);
    }

    // Convert back to 8-bit.
    out = fsiv_convert_image_float_to_byte(out);

    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(out.channels() == in.channels());
    return out;
}
