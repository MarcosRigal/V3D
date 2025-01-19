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

    cv::Mat float_img = fsiv_convert_image_byte_to_float(in);

    if (in.channels() == 3 and only_luma) {
        cv::Mat hsv_img = fsiv_convert_bgr_to_hsv(float_img);
        std::vector<cv::Mat> channels;

        cv::split(hsv_img, channels);
        cv::pow(channels[2], gamma, channels[2]);

        channels[2] = contrast * channels[2] + brightness;
        cv::merge(channels, hsv_img);

        out = fsiv_convert_hsv_to_bgr(hsv_img);
    } else {
        cv::pow(float_img, gamma, out);
        cv::multiply(cv::Scalar::all(contrast), out, out);
        out += cv::Scalar::all(brightness);
    }

    out = fsiv_convert_image_float_to_byte(out);

    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(out.channels() == in.channels());
    return out;
}
