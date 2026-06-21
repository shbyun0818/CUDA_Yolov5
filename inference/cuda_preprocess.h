#pragma once

#include <opencv2/core.hpp>

#include <vector>

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

void preprocess_image_cuda(
    const cv::Mat& src,
    int input_size,
    std::vector<float>& dst,
    LetterboxInfo& info
);
