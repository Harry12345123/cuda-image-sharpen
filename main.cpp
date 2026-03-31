#include <opencv2/opencv.hpp>
#include <iostream>
void sharpen_cuda(const unsigned char *in, unsigned char *out, int width, int height, int channels);
int main(int argc, char **argv)
{
    std::string input = argc > 1 ? argv[1] : "input.jpg";
    cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
    if (img.empty())
        return -1;
    cv::Mat out(img.rows, img.cols, img.type());
    sharpen_cuda(img.data, out.data, img.cols, img.rows, img.channels());
    cv::imwrite("output_sharpen.jpg", out);
    return 0;
}