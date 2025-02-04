#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common_code.hpp"

void fsiv_compute_derivate(cv::Mat const &img, cv::Mat &dx, cv::Mat &dy, int g_r,
                           int s_ap)
{
    CV_Assert(img.type() == CV_8UC1);

    if (g_r > 0)
    {
        int kernel_size = 2 * g_r + 1;
        cv::GaussianBlur(img, img, cv::Size(kernel_size, kernel_size), 0);
    }

    cv::Sobel(img, dx, CV_32F, 1, 0, s_ap);
    cv::Sobel(img, dy, CV_32F, 0, 1, s_ap);

    CV_Assert(dx.size() == img.size());
    CV_Assert(dy.size() == dx.size());
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);
}

void fsiv_compute_gradient_magnitude(cv::Mat const &dx, cv::Mat const &dy,
                                     cv::Mat &gradient)
{
    CV_Assert(dx.size() == dy.size());
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);

    cv::magnitude(dx, dy, gradient);

    CV_Assert(gradient.size() == dx.size());
    CV_Assert(gradient.type() == CV_32FC1);
}

void fsiv_compute_gradient_histogram(cv::Mat const &gradient, int n_bins, cv::Mat &hist, float &max_gradient)
{
    CV_Assert(!gradient.empty());
    CV_Assert(gradient.type() == CV_32F || gradient.type() == CV_64F);
    CV_Assert(n_bins > 0);

    double min_val, max_val;
    cv::minMaxLoc(gradient, &min_val, &max_val);
    max_gradient = static_cast<float>(max_val);

    CV_Assert(max_gradient > 0.0); 

    float range[] = {0, max_gradient};
    const float *hist_range = {range};
    bool uniform = true;
    bool accumulate = false;

    cv::calcHist(&gradient,
                 1,         
                 0,
                 cv::Mat(), 
                 hist,      
                 1,         
                 &n_bins,
                 &hist_range,
                 uniform,
                 accumulate);

    CV_Assert(max_gradient > 0.0);
    CV_Assert(hist.rows == n_bins);
}

int fsiv_compute_histogram_percentile(cv::Mat const &hist, float percentile)
{
    CV_Assert(percentile >= 0.0 && percentile <= 1.0);
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.cols == 1);
    int idx = -1;

    float total_area = cv::sum(hist)[0];
    CV_Assert(total_area > 0.0);
    float cumulative_sum = 0.0f;

    for (int i = 0; i < hist.rows; ++i)
    {
        cumulative_sum += hist.at<float>(i);
        if (cumulative_sum / total_area >= percentile)
        {
            idx = i;
            break;
        }
    }

    if (percentile == 1.0) { idx = hist.rows - 1; }

    CV_Assert(idx >= 0 && idx < hist.rows);
    CV_Assert(idx == 0 || cv::sum(hist(cv::Range(0, idx), cv::Range::all()))[0] / cv::sum(hist)[0] < percentile);
    CV_Assert(cv::sum(hist(cv::Range(0, idx + 1), cv::Range::all()))[0] / cv::sum(hist)[0] >= percentile);
    return idx;
}

float fsiv_histogram_idx_to_value(int idx, int n_bins, float max_value,
                                  float min_value)
{
    CV_Assert(idx >= 0);
    CV_Assert(idx < n_bins);
    float value = 0.0;

    float bin_width = (max_value - min_value) / n_bins;
    value = min_value + idx * bin_width;

    CV_Assert(value >= min_value);
    CV_Assert(value < max_value);
    return value;
}

void fsiv_percentile_edge_detector(cv::Mat const &gradient, cv::Mat &edges,
                                   float th, int n_bins)
{
    CV_Assert(gradient.type() == CV_32FC1);
    CV_Assert(th >= 0.0 && th <= 1.0);
    CV_Assert(n_bins > 0);

    cv::Mat hist;
    float max_gradient;
    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_gradient);

    int idx = fsiv_compute_histogram_percentile(hist, th);

    float gradient_threshold = fsiv_histogram_idx_to_value(idx, n_bins, max_gradient, 0.0f);

    edges = cv::Mat::zeros(gradient.size(), CV_8UC1);
    cv::threshold(gradient, edges, gradient_threshold, 255, cv::THRESH_BINARY);

    edges.convertTo(edges, CV_8UC1);

    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == gradient.size());
}

void fsiv_otsu_edge_detector(cv::Mat const &gradient, cv::Mat &edges)
{
    CV_Assert(gradient.type() == CV_32FC1);

    cv::Mat normalized_gradient;
    double min_val, max_val;
    cv::minMaxLoc(gradient, &min_val, &max_val);

    gradient.convertTo(normalized_gradient, CV_8UC1, 255.0 / (max_val - min_val), -255.0 * min_val / (max_val - min_val));

    cv::threshold(normalized_gradient, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == gradient.size());
}

void fsiv_canny_edge_detector(cv::Mat const &dx, cv::Mat const &dy, cv::Mat &edges,
                              float th1, float th2, int n_bins)
{
    CV_Assert(dx.size() == dy.size());
    CV_Assert(th1 < th2);

    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);
    CV_Assert(th1 >= 0.0 && th1 <= 1.0);
    CV_Assert(th2 >= 0.0 && th2 <= 1.0);
    CV_Assert(n_bins > 0);

    cv::Mat gradient;
    cv::magnitude(dx, dy, gradient);

    cv::Mat hist;
    float max_gradient;
    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_gradient);

    int idx1 = fsiv_compute_histogram_percentile(hist, th1);
    int idx2 = fsiv_compute_histogram_percentile(hist, th2);

    float gradient_th1 = fsiv_histogram_idx_to_value(idx1, n_bins, max_gradient, 0.0f);
    float gradient_th2 = fsiv_histogram_idx_to_value(idx2, n_bins, max_gradient, 0.0f);

    cv::Mat dx_16s, dy_16s;
    dx.convertTo(dx_16s, CV_16SC1);
    dy.convertTo(dy_16s, CV_16SC1);

    cv::Canny(dx_16s, dy_16s, edges, gradient_th1, gradient_th2, true);

    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == dx.size());
}

void fsiv_compute_ground_truth_image(cv::Mat const &consensus_img,
                                     float min_consensus, cv::Mat &gt)
{

    cv::Mat normalized_img;
    cv::normalize(consensus_img, normalized_img, 0, 100, cv::NORM_MINMAX, CV_32F);

    gt = (normalized_img >= min_consensus);
    gt.convertTo(gt, CV_8UC1, 255.0);    

    CV_Assert(consensus_img.size() == gt.size());
    CV_Assert(gt.type() == CV_8UC1);
}

void fsiv_compute_confusion_matrix(cv::Mat const &gt, cv::Mat const &pred, cv::Mat &cm)
{
    CV_Assert(gt.type() == CV_8UC1);
    CV_Assert(pred.type() == CV_8UC1);
    CV_Assert(gt.size() == pred.size());

    float TP = 0.0f;
    float TN = 0.0f;
    float FP = 0.0f;
    float FN = 0.0f;

    for (int y = 0; y < gt.rows; ++y)
    {
        for (int x = 0; x < gt.cols; ++x)
        {
            bool gt_edge = gt.at<uchar>(y, x) != 0;
            bool pred_edge = pred.at<uchar>(y, x) != 0;

            if (gt_edge && pred_edge)
                TP += 1.0f;
            else if (!gt_edge && !pred_edge)
                TN += 1.0f;
            else if (!gt_edge && pred_edge)
                FP += 1.0f;
            else if (gt_edge && !pred_edge)
                FN += 1.0f;
        }
    }

    cm = (cv::Mat_<float>(2, 2) << TP, FN,
          FP, TN);

    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cv::abs(cv::sum(cm)[0] - (gt.rows * gt.cols)) < 1.0e-6);
}

float fsiv_compute_sensitivity(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float sensitivity = 0.0;

    float TP = cm.at<float>(0, 0);
    float FN = cm.at<float>(0, 1);

    sensitivity = (TP + FN) > 0 ? TP / (TP + FN) : 0.0f;
   
    return sensitivity;
}

float fsiv_compute_precision(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float precision = 0.0;

    float TP = cm.at<float>(0, 0);
    float FP = cm.at<float>(1, 0);

    precision = (TP + FP) > 0 ? TP / (TP + FP) : 0.0f;

    return precision;
}

float fsiv_compute_F1_score(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float F1 = 0.0;

    float precision = fsiv_compute_precision(cm);
    float sensitivity = fsiv_compute_sensitivity(cm);

    F1 = (precision + sensitivity) > 0 ? 
               2.0f * (precision * sensitivity) / (precision + sensitivity) : 0.0f;

    return F1;
}
