#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "../gl_helper.h"
#include "../cpu_bitmap.h"

#include "tools.hpp"


__global__ void add(int a, int b, int * c) {
    *c = a + b;
}

__global__ void addArray(int * a, int * b, int * c) {
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}


__global__ void addArrayLong(int * a, int * b, int * c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        //the while loop
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void doAdd(int a, int b) {
    int * dev_c;
    int c;
    cudaMalloc((void**)&dev_c, sizeof(int));
    add<<<1, 1>>>(a, b, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "The result of " << a << "+" << b << " is " << c << std::endl;
    cudaFree(dev_c);
}

void doAddArray(int * a, int * b) {
    int * dev_a;
    int * dev_b;
    int * dev_c;
    int c[N] = {0};
    const size_t size = N * sizeof(int);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_a, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_b, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_c, size));

    CUDA_CHECK_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

    addArray<<<N, 1>>>(dev_a, dev_b, dev_c);

    CUDA_CHECK_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;

    CUDA_CHECK_ERROR(cudaFree(dev_a));
    CUDA_CHECK_ERROR(cudaFree(dev_b));
    CUDA_CHECK_ERROR(cudaFree(dev_c));
}

void doAddArrayLong(int * a, int * b) {
    int * dev_a;
    int * dev_b;
    int * dev_c;
    int c[N] = {0};
    const size_t size = N * sizeof(int);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_a, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_b, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_c, size));

    CUDA_CHECK_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

    //can't exceed maxThreadsPerBlock
    addArrayLong<<<1024, 1024>>>(dev_a, dev_b, dev_c);

    CUDA_CHECK_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
            success = false;
        }
    }
    if (success)
        std::cout << "successfully compute!" << std::endl;

    CUDA_CHECK_ERROR(cudaFree(dev_a));
    CUDA_CHECK_ERROR(cudaFree(dev_b));
    CUDA_CHECK_ERROR(cudaFree(dev_c));

    CUDA_CHECK_ERROR(cudaDeviceReset());
}

struct cuComplex {
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void) {
        return r * r + i + i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y) {
    const float scale = 0.9;
    float jx = scale * (float)(DIM/2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM/2 - y) / (DIM / 2);

    cuComplex c(-0.91, 0.125);
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; ++i) {
        a = a * a  + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void JuliaKernel(unsigned char * ptr) {
    //int x = blockIdx.x;
    //int y = blockIdx.y;
    //int offset = x + y * gridDim.x;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int offset = x + y * blockDim.x;
    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 100 * juliaValue;
    ptr[offset * 4 + 1] = 250;
    ptr[offset * 4 + 2] = 20;
    ptr[offset * 4 + 3] = 1;
}

void DrawJulia(void) {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char * dev_bitmap;

    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);
    JuliaKernel<<<1, grid>>>(dev_bitmap);

    CUDA_CHECK_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    CUDA_CHECK_ERROR(cudaFree(dev_bitmap));
    CUDA_CHECK_ERROR(cudaDeviceReset());
}
