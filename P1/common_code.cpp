
#include "common_code.hpp"

void 
find_min_max_in_channel(const cv::Mat& img,
    cv::uint8_t& min_v, cv::uint8_t& max_v,
    cv::Point& min_loc, cv::Point& max_loc)
{
    CV_Assert( !img.empty() );
    CV_Assert( img.type() == CV_8UC1 );
    
    min_v = img.at<cv::uint8_t>(0, 0);
    max_v = img.at<cv::uint8_t>(0, 0);
    min_loc = cv::Point(0,0);
    max_loc = cv::Point(0,0);
    
    for (int row=0; row<img.rows; ++row)
    {
        for (int col=0; col<img.cols; ++col)
        {
            cv::uint8_t pixel_v = img.at<cv::uint8_t>(row, col);
            if(pixel_v > max_v){
                max_v = pixel_v;
                max_loc = cv::Point(col, row);
            }
            else if(pixel_v < min_v){
                min_v = pixel_v;
                min_loc = cv::Point(col, row);
            }
        }
    }
}

void 
fsiv_find_min_max_loc_1(cv::Mat const& input,
    std::vector<cv::uint8_t>& min_v, std::vector<cv::uint8_t>& max_v,
    std::vector<cv::Point>& min_loc, std::vector<cv::Point>& max_loc)
{
    CV_Assert(input.depth()==CV_8U);

    //! TODO: do a rows/cols scanning to find the first min/max values. 
    // Hint: use cv::split to get the input image channels.
    
    CV_Assert( !input.empty() );

    std::vector<cv::Mat> canales;
    cv::split(input, canales);

    for(size_t c = 0; c<canales.size(); ++c)
    {
        cv::uint8_t aux_min_v;
        cv::uint8_t aux_max_v;
        cv::Point aux_min_loc;
        cv::Point aux_max_loc;

        find_min_max_in_channel(canales[c], aux_min_v, aux_max_v, aux_min_loc, aux_max_loc);
              
        min_v.push_back(aux_min_v);
        max_v.push_back(aux_max_v);
        min_loc.push_back(aux_min_loc);
        max_loc.push_back(aux_max_loc);
 
    }
    //

    CV_Assert(input.channels()==min_v.size());
    CV_Assert(input.channels()==max_v.size());
    CV_Assert(input.channels()==min_loc.size());
    CV_Assert(input.channels()==max_loc.size());
}

void 
fsiv_find_min_max_loc_2(cv::Mat const& input,
    std::vector<double>& min_v, std::vector<double>& max_v,
    std::vector<cv::Point>& min_loc, std::vector<cv::Point>& max_loc)
{

    //! TODO: Find the first min/max values using cv::minMaxLoc function.    
    // Hint: use cv::split to get the input image channels.



    //

    CV_Assert(input.channels()==min_v.size());
    CV_Assert(input.channels()==max_v.size());
    CV_Assert(input.channels()==min_loc.size());
    CV_Assert(input.channels()==max_loc.size());

}

