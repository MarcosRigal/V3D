#pragma once
#include <opencv2/core/core.hpp>

/**
 * @brief Scale the color of an image so an input color is transformed into an output color.
 * @param in is the image to be rescaled.
 * @param from is the input color.
 * @param to is the output color.
 * @return the color rescaled image.
 * @pre in.type()==CV_8UC3
 * @warning A BGR color space is assumed for the input image.
 */
cv::Mat fsiv_color_rescaling(const cv::Mat &in, const cv::Scalar &from,
                             const cv::Scalar &to);

/**
 * @brief Convert a BGR color image to Gray scale.
 * @param img is the input image.
 * @param out is the output image.
 * @return the output image.
 * @pre img.channels()==3
 * @post ret_v.channels()==1
 */
cv::Mat fsiv_convert_bgr_to_gray(const cv::Mat &img, cv::Mat &out);

/**
 * @brief Compute the histogram of an image.
 *
 * @param img the input image.
 * @return the histogram.
 * @pre in.type()==CV_8UC1
 *
 */
cv::Mat fsiv_compute_image_histogram(cv::Mat const &img);

/**
 * @brief Compute the percentile index given a p_value of an histogram.
 *
 * @param hist the histogram.
 * @param p_value the p_value
 * @return the percentile index.
 * @pre hist.type()==CV_32FC1
 * @pre 0<=p_value && p_value<=1.0
 * @post 0<=ret_v && ret_v<hist.rows
 */
float fsiv_compute_histogram_percentile(cv::Mat const &hist, float p_value);

/**
 * @brief Apply a "gray world" color balance operation to the image.
 * @param[in] in is the input image.
 * @return the color balanced image.
 * @pre in.type()==CV_8UC3
 * @warning A BGR color space is assumed for the input image.
 */
cv::Mat fsiv_gray_world_color_balance(cv::Mat const &in);

/**
 * @brief Apply a "white patch" color balance operation to the image.
 * @param[in] in is the input image.
 * @param[in] p use this percentage of brighter pixels. Value p=0 means use the most brighter.
 * @return the color balanced image.
 * @pre in.type()==CV_8UC3
 * @warning A BGR color space is assumed for the input image.
 */
cv::Mat fsiv_white_patch_color_balance(cv::Mat const &in, float p);
