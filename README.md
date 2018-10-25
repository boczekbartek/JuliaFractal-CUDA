# JuliaFractal-CUDA
Julia Fractal implemented in two versions: CPU and GPU in CUDA

# Installation

Requirements:

* cuda
* cmake v3.12
* GLUT library (delivered with CUDA)

```bash
mkdir build
cd build
cmake ..
make -j
```
### Other GPU architectures
As a defualt code is compiled to work with MAXWELL-arch GPUs.

To use different architecture run *cmake* with proper *GPU_ARCH* flag.
Find your params at: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list 
```bash
cmake -DGPU_ARCH=arch=compute_52,code=sm_52 ..
```

# Run

## CPU version
```bash
./JuliaFractal
```

## GPU version
```bash
./JuliaFractal-CUDA
```

## Picture controlling
* z - zoom in
* Shift-z - zoom out
* p - set picture center to mouse cursor