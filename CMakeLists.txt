cmake_minimum_required(VERSION 3.12)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_COMMON "/usr/local/cuda/samples/common/inc")

project(JuliaFractal-CUDA C CXX CUDA)

#set(CMAKE_CXX_STANDARD 14)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(CUDA 10.0 REQUIRED)

# set this according to your GPU architecture
# (see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)
set(GPU_ARCH arch=compute_52,code=sm_52)
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 -gencode ${GPU_ARCH}
)

add_executable(JuliaFractal julia.c)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${CUDA_COMMON})
target_link_libraries(JuliaFractal ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} m)

add_executable(JuliaFractal-CUDA julia.cu)
target_link_libraries(JuliaFractal-CUDA ${CUDA_LIBRARY} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
