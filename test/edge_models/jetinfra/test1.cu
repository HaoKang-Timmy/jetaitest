#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(512, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_accum[1];
  half_t A_local[8];
  half_t B_local[8];
  float C_squared[1];
  float sum_reg[1];
  if ((((int)threadIdx.x) % 4) == 0) {
    ((float*)buf_dyn_shmem)[((((int)threadIdx.x) >> 2) + 512)] = 0x0p+0f/*0.000000e+00*/;
  }
  C_accum[0] = 0x0p+0f/*0.000000e+00*/;
  for (int bk = 0; bk < 48; ++bk) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((bk * 32) + ((((int)threadIdx.x) & 3) * 8)));
    *(uint4*)(B_local + 0) = *(uint4*)(B + ((((((int)threadIdx.x) >> 2) * 1536) + (bk * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    for (int k = 0; k < 8; ++k) {
      C_accum[0] = (C_accum[0] + (((float)A_local[k]) * ((float)B_local[k])));
    }
  }
  tl::fence_proxy_async();
  __syncthreads();
  AtomicAdd((&(((float*)buf_dyn_shmem)[((((int)threadIdx.x) >> 2) + 512)])), C_accum[0]);
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    #pragma unroll
    for (int i = 0; i < 1; ++i) {
      ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 512)] = (((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 512)] / (0x1p+0f/*1.000000e+00*/ + __expf((((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 512)] * -0x1p+0f/*-1.000000e+00*/))));
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 1; ++i_1) {
    C_squared[0] = (((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 127) + 512)] * ((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 127) + 512)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 1; ++i_2) {
    sum_reg[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 1; ++rv) {
      sum_reg[0] = (sum_reg[0] + C_squared[0]);
    }
    sum_reg[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(sum_reg[0], (&(((float*)buf_dyn_shmem)[0])));
  }
  __syncthreads();
  C[(((int)threadIdx.x) >> 2)] = ((half_t)((float*)buf_dyn_shmem)[((((int)threadIdx.x) >> 2) + 512)]);
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 2560);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 2560, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
        main_kernel<<<dim3(1, 1, 1), dim3(512, 1, 1), 2560, stream>>>(A, B, C);
        TILELANG_CHECK_LAST_ERROR("main_kernel");

        return 0;
}