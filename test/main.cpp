#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "camera/camera.h"

using namespace std;

extern void doAdd(int a, int b);
extern void doAddArray(int * a, int * b);
extern void doAddArrayLong(int * a, int * b);
extern void DrawJulia(void);
extern void DrawWave();
extern bool TestDotProduct();
extern void DoUnsynced();
extern void NaiveRaytrace();
extern void NaiveThermalConductionAnim();
extern void NaiveThermalConductionAnimWith1DTexture();
extern void NaiveThermalConductionAnimWith2DTexture();
extern void NaiveThermalConductionAnimWith2DTextureGPU();
extern void BasicGLOperation(int argc, char *argv[]);
extern void HistogramGPUWithGlobalMem();
extern void SingleStreamTest();
extern void NaiveZeroCopyMemoryTest();
extern int DotProductOnBook( void );

int main(int argc, char *argv[])
{
//    constexpr int N = 10;

//    int a[N] = {0};
//    int b[N] = {0};
//    for (int i = 0; i < N; ++i) {
//        a[i] = i;
//        b[i] = i + 1;
//    }
    //DrawJulia();
    //Capture();
    //DrawWave();
    //doAddArrayLong(a, b);
    //DoUnsynced();
//    bool success = TestDotProduct();
//    if (success)
//        cout << "successed" << endl;
//    else
//        cerr << "failed" << endl;
    NaiveRaytrace();
    //NaiveThermalConductionAnim();
    //NaiveThermalConductionAnimWith1DTexture();
    //NaiveThermalConductionAnimWith2DTexture();
    //BasicGLOperation(argc, argv);
    //NaiveThermalConductionAnimWith2DTextureGPU();
    //HistogramGPUWithGlobalMem();
    //SingleStreamTest();
    //TestDotProduct();
    //NaiveZeroCopyMemoryTest();
    //DotProductOnBook();
//    int count = 0;
//    cudaGetDeviceCount(&count);
//    cudaDeviceProp prop;
//    for (int i = 0; i < count; ++i) {
//        cudaGetDeviceProperties(&prop, i);
//        cout << prop.name << endl;
//        cout << prop.totalGlobalMem / 1024 / 1024 / 1024 << "GB" << endl;
//        cout << prop.maxThreadsPerBlock << endl;
//        cout << prop.totalConstMem / 1024.0 << "KB" << endl;
//        cout << "Support Mapped Host Memory: "
//             <<  ((prop.canMapHostMemory == 1)? "true":"false") << endl;
//    }

    return 0;
}

