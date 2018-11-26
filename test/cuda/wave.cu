#include <cuda.h>
#include <cuda_runtime.h>

#include "tools.hpp"
#include "../cpu_anim.h"

struct WaveDataBlock {
    unsigned char * dev_bitmap;
    CPUAnimBitmap * bitmap;
};

void CleanUp(WaveDataBlock * d) {
    CUDA_CHECK_ERROR(cudaFree(d->dev_bitmap));
}

__global__ void WaveKernel(unsigned char * ptr, int ticks) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d/100.0f - ticks/7.0f) /
                                         (d/10.0f + 1.0f));

    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey / 2;
    ptr[offset * 4 + 2] = - grey / 7;
    ptr[offset * 4 + 3] = 128;
}

void GenerateFrame(WaveDataBlock * d, int ticks) {
    dim3 blocks(DIM/16, DIM/16);
    dim3 threads(16, 16);

    WaveKernel<<<blocks, threads>>>(d->dev_bitmap, ticks);

    CUDA_CHECK_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
                                d->dev_bitmap,
                                d->bitmap->image_size(),
                                cudaMemcpyDeviceToHost));
}

void DrawWave() {
    WaveDataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));

    bitmap.anim_and_exit((void(*)(void*, int))GenerateFrame,
                         (void(*)(void*))CleanUp);
}
