/**
 * @file common_code.cpp
 * @author Francisco José Madrid Cuevas (fjmadrid@uco.es)
 * @brief Utility module to do an Unsharp Mask image enhance.
 * @version 0.1
 * @date 2024-09-19
 *
 * @copyright Copyright (c) 2024-
 *
 */
#include "common_code.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat
fsiv_create_box_filter(const int r)
{
    CV_Assert(r > 0);
    cv::Mat ret_v;

    ret_v = cv::Mat::ones(2 * r + 1, 2 * r + 1, CV_32FC1); 
    ret_v /= (2 * r + 1) * (2 * r + 1);

    CV_Assert(ret_v.type() == CV_32FC1);
    CV_Assert(ret_v.rows == (2 * r + 1) && ret_v.rows == ret_v.cols);
    CV_Assert(std::abs(1.0 - cv::sum(ret_v)[0]) < 1.0e-6);
    return ret_v;
}

cv::Mat
fsiv_create_gaussian_filter(const int r)
{
    CV_Assert(r > 0);
    cv::Mat ret_v;

    cv::Mat kernel_1d = cv::getGaussianKernel(2 * r + 1, -1, CV_32FC1);
    ret_v = kernel_1d * kernel_1d.t();
    ret_v /= cv::sum(ret_v)[0];

    CV_Assert(ret_v.type() == CV_32FC1);
    CV_Assert(ret_v.rows == (2 * r + 1) && ret_v.rows == ret_v.cols);
    CV_Assert(std::abs(1.0 - cv::sum(ret_v)[0]) < 1.0e-6);
    return ret_v;
}

cv::Mat
fsiv_fill_expansion(cv::Mat const &in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r > 0);
    cv::Mat ret_v;

    cv::copyMakeBorder(in, ret_v, r, r, r, r, cv::BORDER_CONSTANT, cv::Scalar(0));

    CV_Assert(ret_v.type() == in.type());
    CV_Assert(ret_v.rows == in.rows + 2 * r);
    CV_Assert(ret_v.cols == in.cols + 2 * r);
    return ret_v;
}

cv::Mat
fsiv_circular_expansion(cv::Mat const &in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r > 0);
    cv::Mat ret_v;

    cv::copyMakeBorder(in, ret_v, r, r, r, r, cv::BORDER_WRAP);

    CV_Assert(ret_v.type() == in.type());
    CV_Assert(ret_v.rows == in.rows + 2 * r);
    CV_Assert(ret_v.cols == in.cols + 2 * r);
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(0, 0) == in.at<uchar>(in.rows - r, in.cols - r));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(0, ret_v.cols / 2) == in.at<uchar>(in.rows - r, in.cols / 2));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(0, ret_v.cols - 1) == in.at<uchar>(in.rows - r, r - 1));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows / 2, 0) == in.at<uchar>(in.rows / 2, in.cols - r));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows / 2, ret_v.cols / 2) == in.at<uchar>(in.rows / 2, in.cols / 2));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows - 1, 0) == in.at<uchar>(r - 1, in.cols - r));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows - 1, ret_v.cols / 2) == in.at<uchar>(r - 1, in.cols / 2));
    CV_Assert(!(in.type() == CV_8UC1) || ret_v.at<uchar>(ret_v.rows - 1, ret_v.cols - 1) == in.at<uchar>(r - 1, r - 1));
    return ret_v;
}

cv::Mat
fsiv_filter2D(cv::Mat const &in, cv::Mat const &filter)
{
    CV_Assert(!in.empty() && !filter.empty());
    CV_Assert(in.type() == CV_32FC1 && filter.type() == CV_32FC1);
    cv::Mat ret_v;

    ret_v = cv::Mat::zeros(in.rows - 2 * (filter.rows / 2), in.cols - 2 * (filter.cols / 2), CV_32FC1);

    for (int i = filter.rows / 2; i < in.rows - filter.rows / 2; i++) {
        for (int j = filter.cols / 2; j < in.cols - filter.cols / 2; j++) {
            float sum = 0.0f;
            for (int fi = 0; fi < filter.rows; fi++) {
                for (int fj = 0; fj < filter.cols; fj++) {
                    sum += in.at<float>(i + fi - filter.rows / 2, j + fj - filter.cols / 2) *
                           filter.at<float>(fi, fj);
                }
            }
            ret_v.at<float>(i - filter.rows / 2, j - filter.cols / 2) = sum;
        }
    }

    CV_Assert(ret_v.type() == CV_32FC1);
    CV_Assert(ret_v.rows == in.rows - 2 * (filter.rows / 2));
    CV_Assert(ret_v.cols == in.cols - 2 * (filter.cols / 2));
    return ret_v;
}

cv::Mat
fsiv_combine_images(const cv::Mat src1, const cv::Mat src2,
                    double a, double b)
{
    CV_Assert(src1.type() == src2.type());
    CV_Assert(src1.rows == src2.rows);
    CV_Assert(src1.cols == src2.cols);
    cv::Mat ret_v;

    cv::addWeighted(src1, a, src2, b, 0, ret_v);

    CV_Assert(ret_v.type() == src2.type());
    CV_Assert(ret_v.rows == src2.rows);
    CV_Assert(ret_v.cols == src2.cols);
    return ret_v;
}

cv::Mat
fsiv_usm_enhance(cv::Mat const &in, double g, int r,
                 int filter_type, bool circular, cv::Mat *unsharp_mask)
{
    CV_Assert(!in.empty());
    CV_Assert(in.type() == CV_32FC1);
    CV_Assert(r > 0);
    CV_Assert(filter_type >= 0 && filter_type <= 1);
    CV_Assert(g >= 0.0);
    cv::Mat ret_v;

    cv::Mat expanded_in;
    if (circular)
    {
        expanded_in = fsiv_circular_expansion(in, r);
    }
    else
    {
        expanded_in = fsiv_fill_expansion(in, r);
    }

    cv::Mat filter;
    if (filter_type == 0)
    {
        filter = fsiv_create_box_filter(r);
    }
    else
    {
        filter = fsiv_create_gaussian_filter(r);
    }

    cv::Mat blurred = fsiv_filter2D(expanded_in, filter);

    int crop_width = std::min(in.cols, blurred.cols);
    int crop_height = std::min(in.rows, blurred.rows);
    int crop_start_x = (blurred.cols - crop_width) / 2;
    int crop_start_y = (blurred.rows - crop_height) / 2;

    cv::Rect crop_region(crop_start_x, crop_start_y, crop_width, crop_height);

    CV_Assert(crop_region.x >= 0 && crop_region.y >= 0 &&
              crop_region.x + crop_region.width <= blurred.cols &&
              crop_region.y + crop_region.height <= blurred.rows);

    cv::Mat cropped_blurred = blurred(crop_region);

    cv::Mat mask = in - cropped_blurred;

    if (unsharp_mask != nullptr)
    {
        *unsharp_mask = mask.clone();
    }

    ret_v = fsiv_combine_images(in, mask, 1.0, g);

    CV_Assert(ret_v.rows == in.rows);
    CV_Assert(ret_v.cols == in.cols);
    CV_Assert(ret_v.type() == CV_32FC1);
    return ret_v;
}
