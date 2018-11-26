#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../cpu_anim.h"

#define N 10
#define DIM 2048
#define PI 3.1415926535897923f
#define INF 2e10f

// for thermal conduction
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.00001f
#define SPEED 0.25f

#define MSIZE (100 * 1024 * 1024)


struct DataBlock {
    unsigned char *outputBitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start;
    cudaEvent_t stop;
    float totalTime;
    float frame;
};
// for thermal conduction

#define CUDA_CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t error, const char * file, const int line) {
    if (error != cudaSuccess) {
        std::cerr << "[CUDA FAILED]:" << file << "(" << line << ") - "
                  << cudaGetErrorName(error) << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif
