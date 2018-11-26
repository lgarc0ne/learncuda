#include "tools.hpp"
#include "../cpu_bitmap.h"
#include <math.h>
#include <iostream>
#include <iomanip>
#include <utility>

using std::cout;
using std::endl;

#define rnd(x) (x * rand() / RAND_MAX)

constexpr int SPHERES = 200;


//struct Sphere {
//    float r;
//    float b;
//    float g;
//    float radius;
//    float x;
//    float y;
//    float z;
//    __device__ float hit(float ox, float oy, float * n) {
//        float dx = ox - x;
//        float dy = oy - y;
//        if (dx * dx + dy * dy <  radius * radius) {
//            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
//            // dz是球面上该点的投影到xy水平面的距离
//            // n是dz和半径的比例，最高点为1,边缘为0,控制颜色强度
//            *n = dz / sqrtf(radius * radius);
//            // 投影点的实际z值
//            return dz + z;
//        }
//        return -INF;
//    }

//};

struct Point {
    float x_;
    float y_;
    //__host__ __device__ Point() : x_(0), y_(0) {}
    //__host__ __device__ Point(float x, float y) : x_(x), y_(y) {}
    __host__ __device__ float dot(Point& rhs) {
        return x_ * rhs.x_ + y_ * rhs.y_;
    }
    __host__ __device__ Point outter(Point& rhs) {
        // 此处不能用new分配内存，因为会在kernel中访问
        Point p;
        p.x_ = x_ * rhs.y_;
        p.y_ = - y_ * rhs.x_;
        return std::move(p);
    }
    __host__ __device__ Point vectorTo(Point& rhs) {
        Point p;
        p.x_ = rhs.x_ - x_;
        p.y_ = rhs.y_ - y_;
        return std::move(p);

    }
};

struct Triangle {
    float r;
    float b;
    float g;
    Point pa;
    Point pb;
    Point pc;

    __host__ __device__ float hit(float ox, float oy, float *n) {
        Point p;
        p.x_ = ox;
        p.y_ = oy;
        Point vecAB = pa.vectorTo(pb);
        Point vecAC = pa.vectorTo(pc);
        Point vecAP = pa.vectorTo(p);

        Point v1 = vecAB.outter(vecAC);
        Point v2 = vecAB.outter(vecAP);

        *n = 1;

        return v1.dot(v2) > 0? 1.0:0;
    }
};

// totalConstMem是常量内存的总量
// tx2为64K，超出直接编译时报错
//__constant__ Sphere s[SPHERES];

__constant__ Triangle s[SPHERES];

__global__ void kernel(unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // 标准化像素点（x,y)使得原点在屏幕中心
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0;
    float g = 0;
    float b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; ++i) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        // 被投射位置（圆上）不在无限远处
        if (t > 0) {
            // 根据距离调整光照强度
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}


void NaiveRaytrace() {

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

    CPUBitmap bitmap(DIM, DIM);
    unsigned char * dev_bitmap;

    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    //CUDA_CHECK_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));

    Triangle * tmp = new Triangle[SPHERES];
    for (int i = 0; i < SPHERES; ++i) {
        tmp[i].r = rnd(1.0f);
        tmp[i].b = rnd(1.0f);
        tmp[i].g = rnd(1.0f);
//        tmp[i].x = rnd(1000.0f) - 500;
//        tmp[i].y = rnd(1000.0f) - 500;
//        tmp[i].z = rnd(1000.0f) - 500;
//        tmp[i].radius = rnd(100.0f) + 20;
        tmp[i].pa.x_ = rnd(10.0f);
        tmp[i].pa.y_ = rnd(50.0f);
        tmp[i].pb.x_ = rnd(200.0f);
        tmp[i].pb.y_ = rnd(120.0f);
        tmp[i].pc.x_ = rnd(70.0f);
        tmp[i].pc.y_ = rnd(100.0f);

    }

    //CUDA_CHECK_ERROR(cudaMemcpy(s, tmp, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
    // 常量内存复制
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(s, tmp, sizeof(Triangle) * SPHERES));
    delete [] tmp;

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    cout << bitmap.image_size() << endl;

    // kernel
    kernel<<<grids, threads>>>(dev_bitmap);


    CUDA_CHECK_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                                cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&ms, start, stop));
    cout.setf(std::ios::fixed);
    cout << ms << "ms" << endl;
    bitmap.display_and_exit();

    CUDA_CHECK_ERROR(cudaFree(dev_bitmap));
    CUDA_CHECK_ERROR(cudaEventDestroy(start));
    CUDA_CHECK_ERROR(cudaEventDestroy(stop));

    // 常量内存不必显式释放
    //CUDA_CHECK_ERROR(cudaFree(s));
    CUDA_CHECK_ERROR(cudaDeviceReset());
}




