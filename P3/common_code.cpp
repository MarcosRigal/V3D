#include "common_code.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

cv::Mat fsiv_color_rescaling(const cv::Mat &in, const cv::Scalar &from, const cv::Scalar &to)
{
    CV_Assert(in.type() == CV_8UC3);
    cv::Mat out = in.clone();
    // TODO
    // HINT: use cv:divide to compute the scaling factor.
    // HINT: use method cv::Mat::mul() to scale the input matrix.

    cv::Mat channels[3];
    cv::split(out, channels);
    for (int i = 0; i < 3; ++i)
    {
        double scale = to[i] / (from[i] + 1e-6); 
         channels[i].convertTo(channels[i], -1, scale);
    }

    cv::merge(channels, 3, out);

    //
    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}

cv::Mat fsiv_gray_world_color_balance(cv::Mat const &in)
{
    
 
    CV_Assert(in.type() == CV_8UC3);

    
    cv::Scalar mean_val = cv::mean(in);

    
    cv::Scalar scale(128.0 / mean_val[0], 128.0 / mean_val[1], 128.0 / mean_val[2]);

    cv::Mat out;
    cv::multiply(in, scale, out);

   
    cv::threshold(out, out, 255, 255, cv::THRESH_TRUNC);

    
    return out;
}



cv::Mat fsiv_convert_bgr_to_gray(const cv::Mat &img, cv::Mat &out)
{
    CV_Assert(img.channels() == 3);
    // TODO
    // HINT: use cv::cvtColor()
   cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);

    //
    CV_Assert(out.channels() == 1);
    return out;
}

cv::Mat fsiv_compute_image_histogram(cv::Mat const &img)
{
    CV_Assert(img.type() == CV_8UC1);
    cv::Mat hist;
    // TODO
    // Hint: use cv::calcHist().

    int histSize = 256; 
    float range[] = {0, 256}; 
    const float *ranges[] = {range};
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, ranges);

    hist.convertTo(hist, CV_32FC1);

    //
    CV_Assert(!hist.empty());
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    return hist;
}

float fsiv_compute_histogram_percentile(cv::Mat const &hist, float p_value)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.cols == 1);
    CV_Assert(0.0 <= p_value && p_value <= 1.0);

    int p = 0;

    // TODO
    // Remember: find the smaller index 'p' such that
    //           sum(h[0], h[1], ... , h[p]) >= p_value*area(hist)
    // Hint: use cv::sum() to compute the histogram area.

    //

    float total = cv::sum(hist)[0];
    float threshold = total * p_value;

    float cumulative_sum = 0.0f;
    for (int i = 0; i < hist.rows; ++i)
    {
        cumulative_sum += hist.at<float>(i);
        if (cumulative_sum >= threshold)
        {
            return static_cast<float>(i);
        }
    }
    return hist.rows - 1; 
}




cv::Mat fsiv_white_patch_color_balance(cv::Mat const &in, float p)
{
    CV_Assert(in.type() == CV_8UC3);
    CV_Assert(0.0f <= p && p <= 100.0f);

    cv::Mat out = in.clone();

    if (p == 0.0f)
    {
        cv::Mat gray;
        fsiv_convert_bgr_to_gray(in, gray);

        double min_val, max_val;
        cv::Point max_loc;
        cv::minMaxLoc(gray, &min_val, &max_val, nullptr, &max_loc);

        cv::Vec3b brightest_pixel = in.at<cv::Vec3b>(max_loc);
        cv::Scalar from(
            std::max(static_cast<double>(brightest_pixel[0]), 1e-6),
            std::max(static_cast<double>(brightest_pixel[1]), 1e-6),
            std::max(static_cast<double>(brightest_pixel[2]), 1e-6)
        );
        cv::Scalar to(255.0, 255.0, 255.0);

        out = fsiv_color_rescaling(out, from, to);
    }
    else
    {
        cv::Mat gray;
        fsiv_convert_bgr_to_gray(in, gray);

        cv::Mat hist = fsiv_compute_image_histogram(gray);
        float percentile = fsiv_compute_histogram_percentile(hist, 1.0f - (p / 100.0f));

        cv::Mat mask = gray >= percentile;

        if (cv::countNonZero(mask) < 10) {
            throw std::runtime_error("Insufficient valid pixels in mask.");
        }

        cv::Scalar mean_val = cv::mean(in, mask);
        cv::Scalar from(
            std::max(mean_val[0], 1e-6),
            std::max(mean_val[1], 1e-6),
            std::max(mean_val[2], 1e-6)
        );
        cv::Scalar to(255.0, 255.0, 255.0);

        out = fsiv_color_rescaling(out, from, to);
    }

    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}



