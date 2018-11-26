#include "tools.hpp"
#include "../book.h"

#include <cstring>
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using std::cerr;
using std::ios;

#define BLOCK_SIZE (100 * 1024 * 1024)

static bool HistogramCPU(unsigned char * block, unsigned int * histogram, bool check) {
    if (check) {
        for (int i = 0; i < BLOCK_SIZE; ++i)
            --histogram[block[i]];
        for (int i = 0; i < 256; ++i) {
            if (histogram[i] != 0)
                return false;
        }
    } else {
        for (int i = 0; i < BLOCK_SIZE; ++i)
            histogram[block[i]]++;
    }
    return true;
}


__global__ void  kernel(unsigned char *block, size_t size, unsigned int *histogram) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        atomicAdd(&histogram[block[i]], 1);
        i += stride;
    }
}

__global__ void kernel_shared(unsigned char *block, size_t size, unsigned *histogram) {
    // 共享内存每一个block会有一个副本存在
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd(&temp[block[i]], 1);
        i += stride;
    }

    __syncthreads();
    // 256个线程，每一个线程负责写入自己在本索引上得到的统计
    atomicAdd(&histogram[threadIdx.x], temp[threadIdx.x]);
}

void HistogramGPUWithGlobalMem() {
    unsigned char *buffer = (unsigned char*)big_random_block(BLOCK_SIZE);
    int device;
    cudaDeviceProp prop;

    prop.major = 1;
    prop.minor = 2;

    memset(&prop, 0, sizeof(prop));
    CUDA_CHECK_ERROR(cudaChooseDevice(&device, &prop));
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, device));

    unsigned char *dev_buffer;
    unsigned int *dev_histogram;

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    CUDA_CHECK_ERROR(cudaEventRecord(start, 0));


    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_buffer, BLOCK_SIZE));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_histogram, BLOCK_SIZE));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_buffer, buffer, BLOCK_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemset(dev_histogram, 0, 256 * sizeof(int)));

    int blocks = prop.multiProcessorCount;
    kernel_shared<<<blocks * 2, 256>>>(dev_buffer, BLOCK_SIZE, dev_histogram);

    unsigned int histogram[256];
    CUDA_CHECK_ERROR(cudaMemcpy(histogram, dev_histogram, 256 * sizeof(int),
                                cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout.setf(ios::fixed);
    cout << std::setw(4) << "Elapsed Time is " << elapsedTime << " ms" << endl;

    bool isCorrect = HistogramCPU(buffer, histogram, true);
    if (isCorrect)
        cout << "Computation Succeed!" << endl;
    else
        cerr << "Computation Failed!" << endl;

    cudaFree(dev_buffer);
    cudaFree(dev_histogram);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(buffer);
}
