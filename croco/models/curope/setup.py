# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile for all possible CUDA architectures
all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()
print("CUDA architectures: ", all_cuda_archs)

# alternatively, you can list cuda archs that you want, eg: 
# https://developer.nvidia.com/cuda-gpus#compute
# all_cuda_archs = all_cuda_archs[-2:]  # keep only the last 3 architectures
# print("pick CUDA architectures: ", all_cuda_archs)


setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=[
                "curope.cpp",
                "kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
