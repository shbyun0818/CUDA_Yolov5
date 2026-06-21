#include "cuda_preprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(op) + " failed: " + cudaGetErrorString(status)
        );
    }
}

__device__ float bilinear_channel(const unsigned char* src, int src_w, int src_h, size_t src_step,
    float src_x, float src_y, int channel) 
{
    src_x = fminf(fmaxf(src_x, 0.0f), static_cast<float>(src_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), static_cast<float>(src_h - 1));

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x1 = (x0 + 1 < src_w) ? x0 + 1 : src_w - 1;
    int y1 = (y0 + 1 < src_h) ? y0 + 1 : src_h - 1;

    float dx = src_x - x0;
    float dy = src_y - y0;

    const unsigned char* p00 = src + y0 * src_step + x0 * 3;
    const unsigned char* p01 = src + y0 * src_step + x1 * 3;
    const unsigned char* p10 = src + y1 * src_step + x0 * 3;
    const unsigned char* p11 = src + y1 * src_step + x1 * 3;

    float top = (1.0f - dx) * p00[channel] + dx * p01[channel];
    float bottom = (1.0f - dx) * p10[channel] + dx * p11[channel];
    return (1.0f - dy) * top + dy * bottom;
}

__global__ void letterbox_kernel(const unsigned char* src, int src_w, int src_h, size_t src_step,
    float* dst, int dst_size, float scale, int pad_x, int pad_y) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_size || y >= dst_size) {
        return;
    }

    float rgb[3] = {
        114.0f / 255.0f,
        114.0f / 255.0f,
        114.0f / 255.0f,
    };

    float src_x = (static_cast<float>(x - pad_x) + 0.5f) / scale - 0.5f;
    float src_y = (static_cast<float>(y - pad_y) + 0.5f) / scale - 0.5f;

    if (src_x >= 0.0f && src_x <= src_w - 1 && src_y >= 0.0f && src_y <= src_h - 1) {
        float b = bilinear_channel(src, src_w, src_h, src_step, src_x, src_y, 0);
        float g = bilinear_channel(src, src_w, src_h, src_step, src_x, src_y, 1);
        float r = bilinear_channel(src, src_w, src_h, src_step, src_x, src_y, 2);

        rgb[0] = r / 255.0f;
        rgb[1] = g / 255.0f;
        rgb[2] = b / 255.0f;
    }

    int area = dst_size * dst_size;
    int dst_index = y * dst_size + x;
    dst[0 * area + dst_index] = rgb[0];
    dst[1 * area + dst_index] = rgb[1];
    dst[2 * area + dst_index] = rgb[2];
}

}  // namespace

void preprocess_image_cuda(const cv::Mat& src, int input_size, std::vector<float>& dst, LetterboxInfo& info) 
{
    if (src.empty()) {
        throw std::runtime_error("preprocess_image_cuda received an empty image");
    }
    if (src.type() != CV_8UC3) {
        throw std::runtime_error("preprocess_image_cuda expects CV_8UC3 BGR input");
    }

    int src_w = src.cols;
    int src_h = src.rows;

    info.scale = std::min(
        static_cast<float>(input_size) / src_w,
        static_cast<float>(input_size) / src_h
    );
    int resized_w = static_cast<int>(std::round(src_w * info.scale));
    int resized_h = static_cast<int>(std::round(src_h * info.scale));
    info.pad_x = (input_size - resized_w) / 2;
    info.pad_y = (input_size - resized_h) / 2;

    dst.resize(3 * input_size * input_size);

    unsigned char* d_src = nullptr;
    float* d_dst = nullptr;
    size_t src_bytes = src.step * src.rows;
    size_t dst_bytes = dst.size() * sizeof(float);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_src), src_bytes), "cudaMalloc d_src");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_dst), dst_bytes), "cudaMalloc d_dst");
        check_cuda(cudaMemcpy2D(d_src, src.step, src.data, src.step, src.cols * src.elemSize(),
                src.rows, cudaMemcpyHostToDevice),"cudaMemcpy2D src");

        dim3 block(16, 16);
        dim3 grid((input_size + block.x - 1) / block.x, (input_size + block.y - 1) / block.y);
        letterbox_kernel<<<grid, block>>>(d_src, src_w, src_h, src.step, d_dst,
            input_size, info.scale, info.pad_x, info.pad_y);
        check_cuda(cudaGetLastError(), "letterbox_kernel launch");
        check_cuda(cudaMemcpy(dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy dst");
    } catch (...) {
        cudaFree(d_dst);
        cudaFree(d_src);
        throw;
    }

    check_cuda(cudaFree(d_dst), "cudaFree d_dst");
    check_cuda(cudaFree(d_src), "cudaFree d_src");
}
