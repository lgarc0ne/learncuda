#include "tools.hpp"
#include "../cpu_anim.h"
#include "../book.h"

#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using std::ios;


__global__ void copy_const_kernel(float *iptr, const float *cptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (cptr[offset] != 0) iptr[offset] = cptr[offset];
}

__global__ void blend_kernel(float *outSrc, float *inSrc) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0) ++left;
    if (x == DIM - 1) --right;

    int up = offset - DIM;
    int down = offset + DIM;
    if (y == 0) up += DIM;
    if (y == DIM - 1) down -= DIM;


    outSrc[offset] = inSrc[offset] + SPEED * (inSrc[up] + inSrc[down] +
                                              inSrc[left] + inSrc[right] -
                                              inSrc[offset] * 4);
}


static void AnimGPU(DataBlock *d, int  ticks) {
    CUDA_CHECK_ERROR(cudaEventRecord(d->start, 0));
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap *bitmap = d->bitmap;

    for (int i = 0; i < 99; ++i) {
        copy_const_kernel<<<grids, threads>>>(d->dev_inSrc, d->dev_constSrc);
        blend_kernel<<<grids, threads>>>(d->dev_outSrc, d->dev_inSrc);

       std::swap(d->dev_inSrc, d->dev_outSrc);
    }
    float_to_color<<<grids, threads>>>(d->outputBitmap, d->dev_inSrc);

    CUDA_CHECK_ERROR(cudaMemcpy(bitmap->get_ptr(), d->outputBitmap,
                                bitmap->image_size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaEventRecord(d->stop, 0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(d->stop));
    float elapsedTime;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime += elapsedTime;
    ++d->frame;
    cout.setf(ios::fixed);
    cout << "Average Time per frame: " << std::setw(4)
         << d->totalTime / d->frame << endl;
}

static void AnimExit(DataBlock *d) {
    CUDA_CHECK_ERROR(cudaFree(d->dev_constSrc));
    CUDA_CHECK_ERROR(cudaFree(d->dev_inSrc));
    CUDA_CHECK_ERROR(cudaFree(d->dev_outSrc));

    CUDA_CHECK_ERROR(cudaEventDestroy(d->start));
    CUDA_CHECK_ERROR(cudaEventDestroy(d->stop));
}


void NaiveThermalConductionAnim() {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frame = 0;
    CUDA_CHECK_ERROR(cudaEventCreate(&data.start));
    CUDA_CHECK_ERROR(cudaEventCreate(&data.stop));

    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.outputBitmap, bitmap.image_size()));

    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));



    float *temp = new float[bitmap.image_size()];
    for (int i = 0; i < DIM * DIM; ++i) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; ++y) {
        for (int x = 400; x < 500; ++x)
            temp[x + y * DIM] = MIN_TEMP;
    }

    CUDA_CHECK_ERROR(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(),
                                cudaMemcpyHostToDevice));
    for (int y = 800; y < DIM; ++y) {
        for (int x = 0; x < 200; ++x) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(),
                                cudaMemcpyHostToDevice));

    delete [] temp;

    bitmap.anim_and_exit((void (*)(void *, int))AnimGPU, ((void(*)(void*))AnimExit));
}
