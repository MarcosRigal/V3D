#include "common_code.hpp"

cv::Mat
fsiv_convert_image_byte_to_float(const cv::Mat &img)
{
    CV_Assert(img.depth() == CV_8U);
    cv::Mat out;
    // Convert from [0,255] to [0,1] range
    img.convertTo(out, CV_32F, 1.0/255.0);
    
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
    // Convert from [0,1] to [0,255] range
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
    // Convert BGR to HSV color space
    cv::cvtColor(img, out, cv::COLOR_BGR2HSV);
    
    CV_Assert(out.channels() == 3);
    return out;
}

cv::Mat
fsiv_convert_hsv_to_bgr(const cv::Mat &img)
{
    CV_Assert(img.channels() == 3);
    cv::Mat out;
    // Convert HSV back to BGR color space
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
        // Process grayscale image
        cv::Mat float_img = fsiv_convert_image_byte_to_float(in);
        
        // Apply gamma correction first
        cv::pow(float_img, gamma, float_img);
        
        // Then apply contrast and brightness
        float_img = contrast * float_img + brightness;
        
        // Clamp values to [0,1] range
        cv::Mat tmp;
        cv::max(float_img, 0.0, tmp);
        cv::min(tmp, 1.0, float_img);
        
        // Convert back to byte range
        out = fsiv_convert_image_float_to_byte(float_img);
    }
    else if (!only_luma) {
        // Process all channels of color image
        cv::Mat float_img = fsiv_convert_image_byte_to_float(in);
        
        // Apply gamma correction first
        cv::pow(float_img, gamma, float_img);
        
        // Then apply contrast and brightness
        float_img = contrast * float_img + brightness;
        
        // Clamp values to [0,1] range
        cv::Mat tmp;
        cv::max(float_img, 0.0, tmp);
        cv::min(tmp, 1.0, float_img);
        
        // Convert back to byte range
        out = fsiv_convert_image_float_to_byte(float_img);
    }
    else {
        // Process only V channel in HSV space
        cv::Mat hsv = fsiv_convert_bgr_to_hsv(in);
        std::vector<cv::Mat> channels;
        cv::split(hsv, channels);
        
        // Convert V channel to float [0,1]
        cv::Mat v_float = fsiv_convert_image_byte_to_float(channels[2]);
        
        // Apply gamma correction first
        cv::pow(v_float, gamma, v_float);
        
        // Then apply contrast and brightness
        v_float = contrast * v_float + brightness;
        
        // Clamp values to [0,1] range
        cv::Mat tmp;
        cv::max(v_float, 0.0, tmp);
        cv::min(tmp, 1.0, v_float);
        
        // Convert back to byte range
        channels[2] = fsiv_convert_image_float_to_byte(v_float);
        
        // Merge channels and convert back to BGR
        cv::merge(channels, hsv);
        out = fsiv_convert_hsv_to_bgr(hsv);
    }
    
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    CV_Assert(out.depth() == CV_8U);
    CV_Assert(out.channels() == in.channels());
    return out;
}