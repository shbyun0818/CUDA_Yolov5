#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#ifdef USE_CUDA_PREPROCESS
#include "cuda_preprocess.h"
#endif

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>


struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

cv::Mat letterbox(const cv::Mat& src, int new_size, float& scale, int& pad_x, int& pad_y) {
    int w = src.cols;
    int h = src.rows;

    scale = std::min((float)new_size / w, (float)new_size / h);
    int resized_w = std::round(w * scale);
    int resized_h = std::round(h * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_w, resized_h));

    pad_x = (new_size - resized_w) / 2;
    pad_y = (new_size - resized_h) / 2;

    cv::Mat out(new_size, new_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

    return out;
}

int main(int argc, char** argv) {
    constexpr int input_size = 640;
    const char* model_path = "/home/byun/Downloads/SH/GPU/model/yolov5s.onnx";
    std::string image_path = "/home/byun/Downloads/SH/GPU/test/test.jpg";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov5");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, model_path, session_options);

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "failed to read image: " << image_path << "\n";
        return 1;
    }

    float scale;
    int pad_x, pad_y;
    std::vector<float> input_tensor_values(1 * 3 * input_size * input_size);

#ifdef USE_CUDA_PREPROCESS
    try {
        LetterboxInfo letterbox_info;
        preprocess_image_cuda(image, input_size, input_tensor_values, letterbox_info);
        scale = letterbox_info.scale;
        pad_x = letterbox_info.pad_x;
        pad_y = letterbox_info.pad_y;
    } catch (const std::exception& e) {
        std::cerr << "CUDA preprocess failed: " << e.what() << "\n";
        return 1;
    }
#else
    cv::Mat input_img = letterbox(image, input_size, scale, pad_x, pad_y);

    cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
    input_img.convertTo(input_img, CV_32F, 1.0 / 255.0);

    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < input_size; y++) {
            for (int x = 0; x < input_size; x++) {
                input_tensor_values[idx++] = input_img.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
#endif

    std::array<int64_t, 4> input_shape{1, 3, input_size, input_size};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output"};

    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float* data = outputs[0].GetTensorMutableData<float>();

    const int num_boxes = 25200;
    const int num_classes = 80;
    const float conf_thres = 0.25f;
    const float nms_thres = 0.45f;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < num_boxes; i++) {
        float* row = data + i * 85;

        float obj_conf = row[4];
        if (obj_conf < conf_thres) continue;

        float max_class_score = 0.0f;
        int class_id = 0;

        for (int c = 0; c < num_classes; c++) {
            float score = row[5 + c];
            if (score > max_class_score) {
                max_class_score = score;
                class_id = c;
            }
        }

        float conf = obj_conf * max_class_score;
        if (conf < conf_thres) continue;

        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;

        x1 = (x1 - pad_x) / scale;
        y1 = (y1 - pad_y) / scale;
        w = w / scale;
        h = h / scale;

        cv::Rect box(std::round(x1), std::round(y1), std::round(w), std::round(h));

        boxes.push_back(box);
        scores.push_back(conf);
        class_ids.push_back(class_id);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thres, nms_thres, indices);

    for (int idx : indices) {
        cv::rectangle(image, boxes[idx], cv::Scalar(0, 255, 0), 2);

        std::string label = std::to_string(class_ids[idx]) + " " + std::to_string(scores[idx]);
        cv::putText(image, label, boxes[idx].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("result.jpg", image);
    std::cout << "saved result.jpg\n";

    return 0;
}
