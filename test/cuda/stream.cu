#include "tools.hpp"

#include <iostream>
#include <iomanip>

using std::cerr;
using std::cout;
using std::endl;
using std::ios;
using std::setw;


#define MSIZE (1024 * 1024)
#define FULL_DATA_SIZE (MSIZE * 20)

__global__ void test_kernel(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < MSIZE) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

void SingleStreamTest() {
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK_ERROR(cudaGetDevice(&device));
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, device));
    if (!prop.deviceOverlap) {
        cerr << "Device: " << prop.name << " don't support overlap for stream!"
             << endl;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

    cudaStream_t stream;
    cudaStream_t stream1;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream1));

    int *host_a;
    int *host_b;
    int *host_c;
    int *dev_a;
    int *dev_b;
    int *dev_c;
    int *dev_a1;
    int *dev_b1;
    int *dev_c1;

    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_a, MSIZE * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_b, MSIZE * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_c, MSIZE * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_a1, MSIZE * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_b1, MSIZE * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_c1, MSIZE * sizeof(int)));


    CUDA_CHECK_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int),
                                   cudaHostAllocDefault));
    CUDA_CHECK_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int),
                                   cudaHostAllocDefault));
    CUDA_CHECK_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int),
                                   cudaHostAllocDefault));

    for (int i = 0; i < FULL_DATA_SIZE; ++i) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    for (int i = 0; i < FULL_DATA_SIZE; i += MSIZE * 2) {
        CUDA_CHECK_ERROR(cudaMemcpyAsync(dev_a, host_a + i, MSIZE * sizeof(int),
                                         cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_ERROR(cudaMemcpyAsync(dev_a1, host_a + i + MSIZE, MSIZE * sizeof(int),
                                         cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK_ERROR(cudaMemcpyAsync(dev_b, host_b + i, MSIZE * sizeof(int),
                                         cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_ERROR(cudaMemcpyAsync(dev_b1, host_b + i + MSIZE, MSIZE * sizeof(int),
                                         cudaMemcpyHostToDevice, stream1));

        test_kernel<<<MSIZE / 256, 256>>>(dev_a, dev_b, dev_c);
        test_kernel<<<MSIZE / 256, 256>>>(dev_a1, dev_b1, dev_c1);
        // 第二个流


        CUDA_CHECK_ERROR(cudaMemcpyAsync(host_c + i, dev_c, MSIZE * sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK_ERROR(cudaMemcpyAsync(host_c + i + MSIZE, dev_c1, MSIZE * sizeof(int),
                                         cudaMemcpyDeviceToHost, stream1));

    }

    // 等待流同步执行
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream1));

    CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout.setf(ios::fixed);
    cout << setw(6) << "Time taken: " << elapsedTime << " ms" << endl;
    cout << setw(-1) << elapsedTime << endl;

    CUDA_CHECK_ERROR(cudaFreeHost(host_a));
    CUDA_CHECK_ERROR(cudaFreeHost(host_b));
    CUDA_CHECK_ERROR(cudaFreeHost(host_c));
    CUDA_CHECK_ERROR(cudaFree(dev_a));
    CUDA_CHECK_ERROR(cudaFree(dev_b));
    CUDA_CHECK_ERROR(cudaFree(dev_c));
    CUDA_CHECK_ERROR(cudaFree(dev_a1));
    CUDA_CHECK_ERROR(cudaFree(dev_b1));
    CUDA_CHECK_ERROR(cudaFree(dev_c1));

    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream1));

    CUDA_CHECK_ERROR(cudaDeviceReset());
}
