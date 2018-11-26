#include "tools.hpp"
//#include "../cpu_anim.h"
#include "../gpu_anim.h"
//#include "../book.h"

#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using std::ios;

// 纹理内存必须声明为文件作用域的全局变量
// 并且不能作为参数传递
texture<float, 2> textureConstSrc;
texture<float, 2> textureIn;
texture<float, 2> textureOut;

__global__ void copy_const_kernel_tex2D(float *iptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(textureConstSrc, x, y);
    if (c != 0) iptr[offset] = c;
}

__global__ void blend_kernel_tex2D(float *dst, bool dstOut) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
// 2D纹理内存不需要
// 不用担心越界问题
// x < 0 ==> x = 0
// x >= DIM ==> x = DIM - 1
//    int left = offset - 1;
//    int right = offset + 1;
//    if (x == 0) ++left;
//    if (x == DIM - 1) --right;

//    int up = offset - DIM;
//    int down = offset + DIM;
//    if (y == 0) up += DIM;
//    if (y == DIM - 1) down -= DIM;

    float t;
    float d;
    float l;
    float r;
    float current;
    if (dstOut) {
        // 纹理内存的专用读取函数
        t = tex2D(textureIn, x, y - 1);
        d = tex2D(textureIn, x, y + 1);
        l = tex2D(textureIn, x - 1, y);
        r = tex2D(textureIn, x + 1, y);
        current = tex2D(textureIn, x, y);
    } else {
        t = tex2D(textureOut, x, y - 1);
        d = tex2D(textureOut, x, y + 1);
        l = tex2D(textureOut, x - 1, y);
        r = tex2D(textureOut, x + 1, y);
        current = tex2D(textureOut, x, y);
    }
    dst[offset]= current + SPEED * (t + d + l + r - 4 * current);
}


//static void AnimCPU(DataBlock *d, int  ticks) {
//    CUDA_CHECK_ERROR(cudaEventRecord(d->start, 0));
//    dim3 grids(DIM / 16, DIM / 16);
//    dim3 threads(16, 16);
//    CPUAnimBitmap *bitmap = d->bitmap;

//    volatile bool dstOut = true;
//    for (int i = 0; i < 99; ++i) {
//        float *in;
//        float *out;
//        if (dstOut) {
//            in = d->dev_inSrc;
//            out = d->dev_outSrc;
//        } else {
//            out = d->dev_inSrc;
//            in = d->dev_outSrc;
//        }
//        copy_const_kernel_tex2D<<<grids, threads>>>(in);
//        blend_kernel_tex2D<<<grids, threads>>>(out,dstOut);

//       dstOut = !dstOut;
//    }
//    float_to_color<<<grids, threads>>>(d->outputBitmap, d->dev_inSrc);

//    CUDA_CHECK_ERROR(cudaMemcpy(bitmap->get_ptr(), d->outputBitmap,
//                                bitmap->image_size(), cudaMemcpyDeviceToHost));
//    CUDA_CHECK_ERROR(cudaEventRecord(d->stop, 0));
//    CUDA_CHECK_ERROR(cudaEventSynchronize(d->stop));
//    float elapsedTime;
//    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
//    d->totalTime += elapsedTime;
//    ++d->frame;
//    cout.setf(ios::fixed);
//    cout << "Average Time per frame: " << std::setw(4)
//         << d->totalTime / d->frame  <<"ms" << endl;
//}

static void AnimGPU(uchar4 *outputBitmap, DataBlock *d, int  ticks) {
    CUDA_CHECK_ERROR(cudaEventRecord(d->start, 0));
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    volatile bool dstOut = true;
    for (int i = 0; i < 99; ++i) {
        float *in;
        float *out;
        if (dstOut) {
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
            in = d->dev_outSrc;
        }
        copy_const_kernel_tex2D<<<grids, threads>>>(in);
        blend_kernel_tex2D<<<grids, threads>>>(out,dstOut);

       dstOut = !dstOut;
    }
    float_to_color<<<grids, threads>>>(outputBitmap, d->dev_inSrc);

    //CUDA_CHECK_ERROR(cudaMemcpy(bitmap->get_ptr(), d->outputBitmap,
    //                           bitmap->image_size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaEventRecord(d->stop, 0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(d->stop));
    float elapsedTime;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime += elapsedTime;
    ++d->frame;
    cout.setf(ios::fixed);
    cout << "Average Time per frame: " << std::setw(4)
         << d->totalTime / d->frame  <<"ms" << endl;
}



static void AnimExit(DataBlock *d) {
    // 解绑定纹理内存
    CUDA_CHECK_ERROR(cudaUnbindTexture(textureIn));
    CUDA_CHECK_ERROR(cudaUnbindTexture(textureOut));
    CUDA_CHECK_ERROR(cudaUnbindTexture(textureConstSrc));

    CUDA_CHECK_ERROR(cudaFree(d->dev_constSrc));
    CUDA_CHECK_ERROR(cudaFree(d->dev_inSrc));
    CUDA_CHECK_ERROR(cudaFree(d->dev_outSrc));

    CUDA_CHECK_ERROR(cudaEventDestroy(d->start));
    CUDA_CHECK_ERROR(cudaEventDestroy(d->stop));
}


//void NaiveThermalConductionAnimWith2DTexture() {
//    DataBlock data;
//    CPUAnimBitmap bitmap(DIM, DIM, &data);
//    data.bitmap = &bitmap;
//    data.totalTime = 0;
//    data.frame = 0;
//    CUDA_CHECK_ERROR(cudaEventCreate(&data.start));
//    CUDA_CHECK_ERROR(cudaEventCreate(&data.stop));

//    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.outputBitmap, bitmap.image_size()));

//    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
//    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
//    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));


//    // 绑定纹理内存
//    // 注意2D texture必须使用的通道描述符
//    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
//    CUDA_CHECK_ERROR(cudaBindTexture2D(NULL, textureConstSrc,
//                                       data.dev_constSrc, desc,
//                                       DIM, DIM, sizeof(float) * DIM));
//    CUDA_CHECK_ERROR(cudaBindTexture2D(NULL, textureIn,
//                                       data.dev_inSrc, desc,
//                                       DIM, DIM, sizeof(float) * DIM));
//    CUDA_CHECK_ERROR(cudaBindTexture2D(NULL, textureOut,
//                                       data.dev_outSrc, desc,
//                                       DIM, DIM, sizeof(float) * DIM));

//    float *temp = new float[bitmap.image_size()];
//    for (int i = 0; i < DIM * DIM; ++i) {
//        temp[i] = 0;
//        int x = i % DIM;
//        int y = i / DIM;
//        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
//            temp[i] = MAX_TEMP;
//    }
//    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
//    temp[DIM * 700 + 100] = MIN_TEMP;
//    temp[DIM * 300 + 300] = MIN_TEMP;
//    temp[DIM * 200 + 700] = MIN_TEMP;
//    for (int y = 800; y < 900; ++y) {
//        for (int x = 400; x < 500; ++x)
//            temp[x + y * DIM] = MIN_TEMP;
//    }

//    // 设置默认const数据
//    CUDA_CHECK_ERROR(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(),
//                                cudaMemcpyHostToDevice));
//    for (int y = 800; y < DIM; ++y) {
//        for (int x = 0; x < 200; ++x) {
//            temp[x + y * DIM] = MAX_TEMP;
//        }
//    }
//    CUDA_CHECK_ERROR(cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(),
//                                cudaMemcpyHostToDevice));

//    delete [] temp;

//    bitmap.anim_and_exit((void (*)(void *, int))AnimCPU, ((void(*)(void*))AnimExit));
//}

void     NaiveThermalConductionAnimWith2DTextureGPU() {
    DataBlock data;
    //CPUAnimBitmap bitmap(DIM, DIM, &data);
    GPUAnimBitmap bitmap(DIM, DIM, &data);
   //data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frame = 0;
    CUDA_CHECK_ERROR(cudaEventCreate(&data.start));
    CUDA_CHECK_ERROR(cudaEventCreate(&data.stop));

    //CUDA_CHECK_ERROR(cudaMalloc((void**)&data.outputBitmap, bitmap.image_size()));

    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));


    // 绑定纹理内存
    // 注意2D texture必须使用的通道描述符
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    CUDA_CHECK_ERROR(cudaBindTexture2D(NULL, textureConstSrc,
                                       data.dev_constSrc, desc,
                                       DIM, DIM, sizeof(float) * DIM));
    CUDA_CHECK_ERROR(cudaBindTexture2D(NULL, textureIn,
                                       data.dev_inSrc, desc,
                                       DIM, DIM, sizeof(float) * DIM));
    CUDA_CHECK_ERROR(cudaBindTexture2D(NULL, textureOut,
                                       data.dev_outSrc, desc,
                                       DIM, DIM, sizeof(float) * DIM));

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

    // 设置默认const数据
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

    bitmap.anim_and_exit((void (*)(uchar4 *, void *, int))AnimGPU, (void (*)(void*))AnimExit);
}
