#ifndef COMMON_CUH
#define COMMON_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_HOST __host__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_HOST
#endif 

#endif
