#include "tools.hpp"

#include <iomanip> // for setw setf

using std::cout;
using std::endl;
using std::ios;

#define imin(a, b) (a<b?a:b)

const int NUM = 10240;
const int threadPerBlock = 256;
const int blocksPerGrid = 1;//imin(32, (NUM + threadPerBlock - 1) / threadPerBlock);

__global__ void dot(long long *a, long long *b, long long *c) {
    __shared__ long long cache[threadPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    long long temp = 0;
    while (tid < NUM) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();

        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

long long DotProduct(long long *a, long long *b) {
    long long *dev_a;
    long long *dev_b;
    long long *dev_c;
    long long * partial_c = new long long[blocksPerGrid];

    const size_t size = NUM * sizeof(long long);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_a, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_b, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dev_c, blocksPerGrid * sizeof(long long)));

    CUDA_CHECK_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

    dot<<<blocksPerGrid, threadPerBlock>>>(dev_a, dev_b, dev_c);

    float elapsedTime;
    CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout.setf(ios::fixed);
    cout << std::setw(6) << "Elapsed Time without Mapped Memory : " << elapsedTime << "ms" << endl;

    CUDA_CHECK_ERROR(cudaMemcpy(partial_c, dev_c, blocksPerGrid * sizeof(long long), cudaMemcpyDeviceToHost));
    long long c = 0;
    for (int i = 0; i < blocksPerGrid; ++i)
        c += partial_c[i];

    CUDA_CHECK_ERROR(cudaFree(dev_a));
    CUDA_CHECK_ERROR(cudaFree(dev_b));
    CUDA_CHECK_ERROR(cudaFree(dev_c));

    CUDA_CHECK_ERROR(cudaDeviceReset());
    return c;
}

bool TestDotProduct() {
    long long * a = new long long[NUM];
    long long * b = new long long[NUM];
    long long tmp = NUM - 1;
    const long long c = tmp * (tmp + 1) * (tmp * 2 + 1) / 6;

    for (int i = 0; i < NUM; ++i) {
        a[i] = i;
        b[i] = i;
    }
    long long result = DotProduct(a, b);
    std::cout << "c      = " << (long long)c << std::endl;
    std::cout << "result = " << (long long)result << std::endl;
    std::cout << "The difference is " << c - result << std::endl;
    return (result - c == 0.0);
}
