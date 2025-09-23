#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_fp16_prefill_kernel(__grid_constant__ const CUtensorMap Cache_desc, __grid_constant__ const CUtensorMap Cache_desc_1, __grid_constant__ const CUtensorMap Cache_desc_2, __grid_constant__ const CUtensorMap Input_desc, __grid_constant__ const CUtensorMap Kernel_input_desc, __grid_constant__ const CUtensorMap Output_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_fp16_prefill_kernel(__grid_constant__ const CUtensorMap Cache_desc, __grid_constant__ const CUtensorMap Cache_desc_1, __grid_constant__ const CUtensorMap Cache_desc_2, __grid_constant__ const CUtensorMap Input_desc, __grid_constant__ const CUtensorMap Kernel_input_desc, __grid_constant__ const CUtensorMap Output_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float Input_reg[8];
  float Kernel_reg[8];
  float Output_reg[8];
  float Output_reduced[4];
  __shared__ uint64_t mbarrier_mem[1];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Kernel_input_desc);
    tl::prefetch_tma_descriptor(Cache_desc);
    tl::prefetch_tma_descriptor(Input_desc);
    tl::prefetch_tma_descriptor(Output_desc);
    tl::prefetch_tma_descriptor(Cache_desc_1);
    tl::prefetch_tma_descriptor(Cache_desc_2);
    mbarrier[0].init(128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[0].expect_transaction(2048);
      tl::tma_load(Kernel_input_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[0])), 0, 0, 0, 0);
      mbarrier[0].expect_transaction(1536);
      tl::tma_load(Cache_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[1024])), 0, 1, 0);
      mbarrier[0].expect_transaction(512);
      tl::tma_load(Input_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[1792])), 0, 0, 0);
    }
    mbarrier[0].arrive();
  } else {
    tl::warpgroup_reg_alloc<240>();
    mbarrier[0].wait(0);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      float4 __1;
      uint2 v_ = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 1024));
      __1.x = (float)(((nv_bfloat162*)(&(v_.x)))->x);
      __1.y = (float)(((nv_bfloat162*)(&(v_.x)))->y);
      __1.z = (float)(((nv_bfloat162*)(&(v_.y)))->x);
      __1.w = (float)(((nv_bfloat162*)(&(v_.y)))->y);
      *(float4*)(Input_reg + (i * 4)) = __1;
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      float4 __2;
      uint2 v__1 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
      __2.x = (float)(((nv_bfloat162*)(&(v__1.x)))->x);
      __2.y = (float)(((nv_bfloat162*)(&(v__1.x)))->y);
      __2.z = (float)(((nv_bfloat162*)(&(v__1.y)))->x);
      __2.w = (float)(((nv_bfloat162*)(&(v__1.y)))->y);
      *(float4*)(Kernel_reg + (i_1 * 4)) = __2;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      Output_reg[i_2] = (Input_reg[i_2] * Kernel_reg[i_2]);
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      Output_reduced[i_3] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        Output_reduced[i_3] = (Output_reduced[i_3] + Output_reg[((rv * 4) + i_3)]);
      }
      Output_reduced[i_3] = tl::AllReduce<tl::SumOp, 128, 64, 0, 128>::run_hopper(Output_reduced[i_3], (&(((float*)buf_dyn_shmem)[0])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      Output_reduced[i_4] = (Output_reduced[i_4] / (0x1p+0f/*1.000000e+00*/ + __expf((Output_reduced[i_4] * -0x1p+0f/*-1.000000e+00*/))));
    }
    if ((((int)threadIdx.x) >> 6) == 0) {
      uint2 __3;
      float4 v__2 = *(float4*)(Output_reduced + 0);
      ((nv_bfloat162*)(&(__3.x)))->x = (bfloat16_t)(v__2.x);
      ((nv_bfloat162*)(&(__3.x)))->y = (bfloat16_t)(v__2.y);
      ((nv_bfloat162*)(&(__3.y)))->x = (bfloat16_t)(v__2.z);
      ((nv_bfloat162*)(&(__3.y)))->y = (bfloat16_t)(v__2.w);
      *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((int)threadIdx.x) & 63) * 4)) = __3;
    }
    tl::fence_proxy_async();
    tl::__sync_thread_partial<3, 128>();
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 0, 0, 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Cache_desc_1, (&(((bfloat16_t*)buf_dyn_shmem)[1024])), 0, 0, 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Cache_desc_2, (&(((bfloat16_t*)buf_dyn_shmem)[1792])), 0, 3, 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_fp16_prefill_kernel = cudaFuncSetAttribute(main_fp16_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);
    if (result_main_fp16_prefill_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 4096, cudaGetErrorString(result_main_fp16_prefill_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ Input, bfloat16_t* __restrict__ Cache, bfloat16_t* __restrict__ Kernel_input, bfloat16_t* __restrict__ Output, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap Cache_desc;
	CUtensorMapDataType Cache_desc_type= (CUtensorMapDataType)9;
	cuuint32_t Cache_desc_tensorRank= 3;
	void *Cache_desc_globalAddress= Cache;
	cuuint64_t Cache_desc_globalDim[3]= {128,4,1};
	cuuint64_t Cache_desc_globalStride[3]= {2,256,1024};
	cuuint32_t Cache_desc_boxDim[3]= {256,3,1};
	cuuint32_t Cache_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Cache_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Cache_desc_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion Cache_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Cache_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Cache_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Cache_desc, Cache_desc_type, Cache_desc_tensorRank, Cache_desc_globalAddress, Cache_desc_globalDim, Cache_desc_globalStride + 1, Cache_desc_boxDim, Cache_desc_elementStrides, Cache_desc_interleave, Cache_desc_swizzle, Cache_desc_l2Promotion, Cache_desc_oobFill);

	if (Cache_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Cache_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Cache_desc_1;
	CUtensorMapDataType Cache_desc_1_type= (CUtensorMapDataType)9;
	cuuint32_t Cache_desc_1_tensorRank= 3;
	void *Cache_desc_1_globalAddress= Cache;
	cuuint64_t Cache_desc_1_globalDim[3]= {128,4,1};
	cuuint64_t Cache_desc_1_globalStride[3]= {2,256,1024};
	cuuint32_t Cache_desc_1_boxDim[3]= {256,4,1};
	cuuint32_t Cache_desc_1_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Cache_desc_1_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Cache_desc_1_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion Cache_desc_1_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Cache_desc_1_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Cache_desc_1_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Cache_desc_1, Cache_desc_1_type, Cache_desc_1_tensorRank, Cache_desc_1_globalAddress, Cache_desc_1_globalDim, Cache_desc_1_globalStride + 1, Cache_desc_1_boxDim, Cache_desc_1_elementStrides, Cache_desc_1_interleave, Cache_desc_1_swizzle, Cache_desc_1_l2Promotion, Cache_desc_1_oobFill);

	if (Cache_desc_1_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Cache_desc_1";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Cache_desc_2;
	CUtensorMapDataType Cache_desc_2_type= (CUtensorMapDataType)9;
	cuuint32_t Cache_desc_2_tensorRank= 3;
	void *Cache_desc_2_globalAddress= Cache;
	cuuint64_t Cache_desc_2_globalDim[3]= {128,4,1};
	cuuint64_t Cache_desc_2_globalStride[3]= {2,256,1024};
	cuuint32_t Cache_desc_2_boxDim[3]= {256,1,1};
	cuuint32_t Cache_desc_2_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Cache_desc_2_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Cache_desc_2_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion Cache_desc_2_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Cache_desc_2_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Cache_desc_2_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Cache_desc_2, Cache_desc_2_type, Cache_desc_2_tensorRank, Cache_desc_2_globalAddress, Cache_desc_2_globalDim, Cache_desc_2_globalStride + 1, Cache_desc_2_boxDim, Cache_desc_2_elementStrides, Cache_desc_2_interleave, Cache_desc_2_swizzle, Cache_desc_2_l2Promotion, Cache_desc_2_oobFill);

	if (Cache_desc_2_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Cache_desc_2";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Input_desc;
	CUtensorMapDataType Input_desc_type= (CUtensorMapDataType)9;
	cuuint32_t Input_desc_tensorRank= 3;
	void *Input_desc_globalAddress= Input;
	cuuint64_t Input_desc_globalDim[3]= {128,1,1};
	cuuint64_t Input_desc_globalStride[3]= {2,256,256};
	cuuint32_t Input_desc_boxDim[3]= {256,1,1};
	cuuint32_t Input_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Input_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Input_desc_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion Input_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Input_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Input_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Input_desc, Input_desc_type, Input_desc_tensorRank, Input_desc_globalAddress, Input_desc_globalDim, Input_desc_globalStride + 1, Input_desc_boxDim, Input_desc_elementStrides, Input_desc_interleave, Input_desc_swizzle, Input_desc_l2Promotion, Input_desc_oobFill);

	if (Input_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Input_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Kernel_input_desc;
	CUtensorMapDataType Kernel_input_desc_type= (CUtensorMapDataType)9;
	cuuint32_t Kernel_input_desc_tensorRank= 4;
	void *Kernel_input_desc_globalAddress= Kernel_input;
	cuuint64_t Kernel_input_desc_globalDim[4]= {128,4,1,1};
	cuuint64_t Kernel_input_desc_globalStride[4]= {2,256,1024,1024};
	cuuint32_t Kernel_input_desc_boxDim[4]= {256,4,1,1};
	cuuint32_t Kernel_input_desc_elementStrides[4]= {1,1,1,1};
	CUtensorMapInterleave Kernel_input_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Kernel_input_desc_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion Kernel_input_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Kernel_input_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Kernel_input_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Kernel_input_desc, Kernel_input_desc_type, Kernel_input_desc_tensorRank, Kernel_input_desc_globalAddress, Kernel_input_desc_globalDim, Kernel_input_desc_globalStride + 1, Kernel_input_desc_boxDim, Kernel_input_desc_elementStrides, Kernel_input_desc_interleave, Kernel_input_desc_swizzle, Kernel_input_desc_l2Promotion, Kernel_input_desc_oobFill);

	if (Kernel_input_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Kernel_input_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Output_desc;
	CUtensorMapDataType Output_desc_type= (CUtensorMapDataType)9;
	cuuint32_t Output_desc_tensorRank= 3;
	void *Output_desc_globalAddress= Output;
	cuuint64_t Output_desc_globalDim[3]= {128,1,1};
	cuuint64_t Output_desc_globalStride[3]= {2,256,256};
	cuuint32_t Output_desc_boxDim[3]= {256,1,1};
	cuuint32_t Output_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Output_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Output_desc_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion Output_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Output_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Output_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Output_desc, Output_desc_type, Output_desc_tensorRank, Output_desc_globalAddress, Output_desc_globalDim, Output_desc_globalStride + 1, Output_desc_boxDim, Output_desc_elementStrides, Output_desc_interleave, Output_desc_swizzle, Output_desc_l2Promotion, Output_desc_oobFill);

	if (Output_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Output_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	main_fp16_prefill_kernel<<<dim3(1, 1, 1), dim3(256, 1, 1), 4096, stream>>>(Cache_desc, Cache_desc_1, Cache_desc_2, Input_desc, Kernel_input_desc, Output_desc);
	TILELANG_CHECK_LAST_ERROR("main_fp16_prefill_kernel");

	return 0;
}
