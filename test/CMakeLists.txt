CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
project(foo_cuda CUDA CXX)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# highgui and imgproc are for cv::imshow cv::waitKey
find_package(OpenCV REQUIRED core videoio highgui imgproc)
include_directories(/usr/local/cuda-9.0/include)
include_directories(/usr/include/c++/5)
INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}/../)
file(GLOB_RECURSE cxx_files *.cpp)
file(GLOB_RECURSE cxx_headers *.h)
file(GLOB_RECURSE cuda_files *.cu *.hpp)

#exclude CMakeFiles directory
list(FILTER cxx_files EXCLUDE REGEX ".*/CMakeFiles/.*")
list(FILTER cxx_headers EXCLUDE REGEX ".*/CMakeFiles/.*")
list(FILTER cuda_files EXCLUDE REGEX ".*/CMakeFiles/.*")


LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing-lineinfo -user_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
add_library(cuda_lib STATIC ${cuda_files})
# c++11支持,必须添加给cuda静态库
set_property(TARGET cuda_lib PROPERTY CUDA_STANDARD 11)
CUDA_ADD_EXECUTABLE(foo_cuda ${cxx_files} ${cxx_headers})

target_link_libraries(foo_cuda
                    /usr/local/cuda-9.0/lib64/libcudart.so
                    cuda_lib
                    ${OPENGL_LIBRARIES}
                    ${GLUT_LIBRARIES}
                    ${OpenCV_LIBS})

