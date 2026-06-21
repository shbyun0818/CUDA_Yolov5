cmake_minimum_required(VERSION 3.14)
project(yolov5_onnx_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include(CheckLanguage)
check_language(CUDA)

option(USE_CUDA_PREPROCESS "Use CUDA kernel for YOLOv5 letterbox preprocessing" ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(DEFAULT_ONNXRUNTIME_DIR "${REPO_ROOT}/onnxruntime-linux-x64-1.20.1")

set(ONNXRUNTIME_DIR "${DEFAULT_ONNXRUNTIME_DIR}" CACHE PATH "Path to the ONNX Runtime C++ package")
if(NOT ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${DEFAULT_ONNXRUNTIME_DIR}" CACHE PATH "Path to the ONNX Runtime C++ package" FORCE)
endif()

set(YOLO_SOURCES main.cpp)

if(USE_CUDA_PREPROCESS AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    list(APPEND YOLO_SOURCES cuda_preprocess.cu)
endif()

add_executable(yolo ${YOLO_SOURCES})

target_include_directories(yolo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(yolo PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

if(USE_CUDA_PREPROCESS AND CMAKE_CUDA_COMPILER)
    target_compile_definitions(yolo PRIVATE USE_CUDA_PREPROCESS)
    set_source_files_properties(cuda_preprocess.cu PROPERTIES LANGUAGE CUDA)
    set_target_properties(yolo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "CUDA compiler not found or USE_CUDA_PREPROCESS=OFF; using CPU preprocessing")
endif()
