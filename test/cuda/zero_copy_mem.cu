#include "tools.hpp"

#include <cstring> // for memset
#include <iomanip> // for setw, setf

using std::cout;
using std::endl;
using std::ios;

#define imax(a, b) (a > b? a : b)

const int NUM_COUNT = 10240;
const int threadsPerBlock = 256;
const int blocksPerGrid = imax(32, (NUM_COUNT + threadsPerBlock - 1) / threadsPerBlock);

__global__ void kernel(long long  *a, long long *b, long long  *c) {
    __shared__ long long  cache[threadsPerBlock];
    int stride = gridDim.x * blockDim.x;
    int tid = threadIdx.x;
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    long long  temp = 0;
    while (index < NUM_COUNT) {
        temp += a[index] * b[index];
        index += stride;
    }
    cache[tid] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (tid < i)
            cache[tid] += cache[tid + i];
        __syncthreads();
        i /= 2;
    }

    if (tid == 0)
        c[blockIdx.x] = cache[0];
}


void NaiveZeroCopyMemoryTest() {
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.canMapHostMemory = 1;
    int device;
    CUDA_CHECK_ERROR(cudaChooseDevice(&device, &prop));
    cudaSetDevice(device);
    cudaSetDeviceFlags(cudaHostAllocMapped);
    cudaGetDeviceProperties(&prop, device);


    long long  *a;
    long long  *b;
    long long  *c;
    long long  *dev_a;
    long long  *dev_b;
    long long  *dev_c;

    // cudaHostAllocWriteCombined 合并式写入
    // 对于GPU读取内存的性能具有显著的提升
    // 但对于CPU读取内存的性能具有显著的负面效果
    CUDA_CHECK_ERROR(cudaHostAlloc((void**)&a, NUM_COUNT * sizeof(long long ),
                                   cudaHostAllocWriteCombined |
                                   cudaHostAllocMapped));
    CUDA_CHECK_ERROR(cudaHostAlloc((void**)&b, NUM_COUNT * sizeof(long long ),
                                   cudaHostAllocWriteCombined |
                                   cudaHostAllocMapped));
    CUDA_CHECK_ERROR(cudaHostAlloc((void**)&c, blocksPerGrid * sizeof(long long ),
                                   cudaHostAllocMapped));

    for (int i = 0; i < NUM_COUNT; ++i) {
        a[i] = i;
        b[i] = i;
    }

    CUDA_CHECK_ERROR(cudaHostGetDevicePointer((void**)&dev_a, a, 0));
    CUDA_CHECK_ERROR(cudaHostGetDevicePointer((void**)&dev_b, b, 0));
    CUDA_CHECK_ERROR(cudaHostGetDevicePointer((void**)&dev_c, c, 0));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    // 同步CPU与GPU
    CUDA_CHECK_ERROR(cudaThreadSynchronize());

    CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
    float  elapsedTime;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout.setf(ios::fixed);
    cout << std::setw(6) << "Elapsed Time : " << elapsedTime << "ms" << endl;

    long long  result = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        result += c[i];
    }
    cout << "The result on GPU is : " << result << endl;
    result = 0;
    for (int i = 0; i < NUM_COUNT; ++i) {
        result += i * i;
    }
    cout << "The result on CPU is : " << result  << endl;


    CUDA_CHECK_ERROR(cudaFreeHost(a));
    CUDA_CHECK_ERROR(cudaFreeHost(b));
    CUDA_CHECK_ERROR(cudaFreeHost(c));

    CUDA_CHECK_ERROR(cudaDeviceReset());
}
