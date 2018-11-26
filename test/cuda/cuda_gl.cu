#define GL_GLEXT_PROTOTYPES

#include <iostream>
#include <cstring>
#include <cmath>

#include <GL/glut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "../book.h"
#include "../cpu_bitmap.h"
#include "tools.hpp"

#define IMG_SIZE 1024

GLuint bufferObject;
cudaGraphicsResource * cudaResource;

__global__ void kernel(uchar4 *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x / (float)IMG_SIZE - 0.5f;
    float fy = y / (float)IMG_SIZE - 0.5f;
    unsigned char green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

    ptr[offset].x = 0;
    ptr[offset].y = green;
    ptr[offset].z = 0;
    ptr[offset].w = 255;
}

static void KeyFunc(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
        // 释放CUDA和 OpenGL
        CUDA_CHECK_ERROR(cudaGraphicsUnregisterResource(cudaResource));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &bufferObject);
        exit(0);
    }
}

static void DisplayFunc() {
    // 已绑定缓冲区，所以最后一个参数为0
    glDrawPixels(IMG_SIZE, IMG_SIZE, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

void BasicGLOperation(int argc, char *argv[]) {
    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    // 获取该设备ID
    CUDA_CHECK_ERROR(cudaChooseDevice(&dev, &prop));
    // 使用该设备执行CUDA和OpenGL操作
    CUDA_CHECK_ERROR(cudaGLSetGLDevice(dev));
    // glut初始化操作
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(IMG_SIZE, IMG_SIZE);
    glutCreateWindow("bitmap");

    // gl数据缓冲区
    glGenBuffers(1, &bufferObject);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObject);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, IMG_SIZE * IMG_SIZE * 4,
                 NULL, GL_DYNAMIC_DRAW_ARB);

    // 将缓冲区在CUDA和OpenGL之间共享
    CUDA_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResource,
                                                  bufferObject,
                                                  cudaGraphicsMapFlagsNone));

    uchar4 *dev_ptr;
    size_t size;
    CUDA_CHECK_ERROR(cudaGraphicsMapResources(1, &cudaResource, NULL));
    CUDA_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr,
                                                          &size,
                                                          cudaResource));

    dim3 grids(IMG_SIZE /16, IMG_SIZE / 16);
    dim3 threads(16, 16);

    kernel<<<grids, threads>>>(dev_ptr);

    // 解除映射，为保证绘图操作的同步，该句之前的CUDA操作完成后，才开始图形调用
    CUDA_CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaResource, NULL));

    // glut绘图操作
    glutKeyboardFunc(KeyFunc);
    glutDisplayFunc(DisplayFunc);
    glutMainLoop();
}
