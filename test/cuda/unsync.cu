#include "tools.hpp"
#include "../cpu_bitmap.h"


__global__ void kernel(unsigned char * ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float shared[16][16];

    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] =
            255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
            (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

    __syncthreads();

    ptr[offset * 4 + 0] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

void DoUnsynced() {
    CPUBitmap bitmap(1920, 1080);
    unsigned char * dev_bitmap;

    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    dim3 grids(1920 / 16, 1080 / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);

    CUDA_CHECK_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
                                bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    CUDA_CHECK_ERROR(cudaFree(dev_bitmap));
    CUDA_CHECK_ERROR(cudaDeviceReset());
}
