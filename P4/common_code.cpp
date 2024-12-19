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

cv::Mat fsiv_create_box_filter(const int r)
{
    CV_Assert(r > 0);
    // Crear un filtro de caja (todos los valores son 1)
    cv::Mat ret_v = cv::Mat::ones(2 * r + 1, 2 * r + 1, CV_32FC1); 

    // Normalizar la suma a 1
    ret_v /= (2 * r + 1) * (2 * r + 1);

    // Verificación
    CV_Assert(ret_v.type() == CV_32FC1);
    CV_Assert(ret_v.rows == (2 * r + 1) && ret_v.rows == ret_v.cols);
    CV_Assert(std::abs(1.0 - cv::sum(ret_v)[0]) < 1.0e-6);
    return ret_v;
}

cv::Mat fsiv_create_gaussian_filter(const int r)
{
    CV_Assert(r > 0);
    // Crear un filtro gaussiano 1D
    cv::Mat kernel_1d = cv::getGaussianKernel(2 * r + 1, -1, CV_32FC1);
    // Convertirlo a 2D multiplicando el vector por su transposición
    cv::Mat ret_v = kernel_1d * kernel_1d.t();

    // Normalizar
    ret_v /= cv::sum(ret_v)[0];

    // Verificación
    CV_Assert(ret_v.type() == CV_32FC1);
    CV_Assert(ret_v.rows == (2 * r + 1) && ret_v.rows == ret_v.cols);
    CV_Assert(std::abs(1.0 - cv::sum(ret_v)[0]) < 1.0e-6);
    return ret_v;
}


cv::Mat fsiv_fill_expansion(cv::Mat const &in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r > 0);
    // Expandir la imagen con bordes de 0
    cv::Mat ret_v;
    cv::copyMakeBorder(in, ret_v, r, r, r, r, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Verificación
    CV_Assert(ret_v.type() == in.type());
    CV_Assert(ret_v.rows == in.rows + 2 * r);
    CV_Assert(ret_v.cols == in.cols + 2 * r);
    return ret_v;
}


cv::Mat fsiv_circular_expansion(cv::Mat const &in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r > 0);
    // Expandir la imagen con borde circular
    cv::Mat ret_v;
    cv::copyMakeBorder(in, ret_v, r, r, r, r, cv::BORDER_WRAP);

    // Verificación
    CV_Assert(ret_v.type() == in.type());
    CV_Assert(ret_v.rows == in.rows + 2 * r);
    CV_Assert(ret_v.cols == in.cols + 2 * r);
    return ret_v;
}


cv::Mat fsiv_filter2D(cv::Mat const &in, cv::Mat const &filter)
{
    CV_Assert(!in.empty() && !filter.empty());
    CV_Assert(in.type() == CV_32FC1 && filter.type() == CV_32FC1);
    cv::Mat ret_v = cv::Mat::zeros(in.rows - 2 * (filter.rows / 2), in.cols - 2 * (filter.cols / 2), CV_32FC1);

    // Convolución manual
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

    return ret_v;
}

cv::Mat fsiv_combine_images(const cv::Mat src1, const cv::Mat src2,
                            double a, double b)
{
    CV_Assert(src1.type() == src2.type());
    CV_Assert(src1.rows == src2.rows);
    CV_Assert(src1.cols == src2.cols);
    cv::Mat ret_v;

    // Combinar las imágenes usando addWeighted
    cv::addWeighted(src1, a, src2, b, 0, ret_v);

    // Verificación
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

    // Step 1: Expand the input image
    cv::Mat expanded_in;
    if (circular)
    {
        expanded_in = fsiv_circular_expansion(in, r);
    }
    else
    {
        expanded_in = fsiv_fill_expansion(in, r);
    }

    // Step 2: Create the filter
    cv::Mat filter;
    if (filter_type == 0)
    {
        filter = fsiv_create_box_filter(r);
    }
    else
    {
        filter = fsiv_create_gaussian_filter(r);
    }

    // Step 3: Convolve the expanded image with the filter
    cv::Mat blurred = fsiv_filter2D(expanded_in, filter);

    // Step 4: Dynamically adjust the crop region
    int crop_width = std::min(in.cols, blurred.cols);
    int crop_height = std::min(in.rows, blurred.rows);
    int crop_start_x = (blurred.cols - crop_width) / 2;
    int crop_start_y = (blurred.rows - crop_height) / 2;

    cv::Rect crop_region(crop_start_x, crop_start_y, crop_width, crop_height);

    CV_Assert(crop_region.x >= 0 && crop_region.y >= 0 &&
              crop_region.x + crop_region.width <= blurred.cols &&
              crop_region.y + crop_region.height <= blurred.rows);

    cv::Mat cropped_blurred = blurred(crop_region);

    // Step 5: Calculate the unsharp mask
    cv::Mat mask = in - cropped_blurred;

    // Step 6: Save the unsharp mask if requested
    if (unsharp_mask != nullptr)
    {
        *unsharp_mask = mask.clone();
    }

    // Step 7: Combine the original image with the unsharp mask
    ret_v = fsiv_combine_images(in, mask, 1.0, g);

    // Final verification
    CV_Assert(ret_v.rows == in.rows);
    CV_Assert(ret_v.cols == in.cols);
    CV_Assert(ret_v.type() == CV_32FC1);

    return ret_v;
}
