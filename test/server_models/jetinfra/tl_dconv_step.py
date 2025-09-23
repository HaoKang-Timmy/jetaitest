import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
import time
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triton
import triton.language as tl
import torch.nn as nn

dtype_dict = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3",
    torch.float8_e5m2: "float8_e5m2",
}
def get_configs():
    block_D = [64 ,128, 256, 512]
    threads = [128]
    _configs = list(itertools.product(block_D, threads))
    configs = [{
        'block_D': c[0],
        'threads': c[1]
    } for c in _configs]
    return configs

# @autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx = [-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def _dconv_step_kernel(
    Batch,
    Token,
    Kernel_size,
    Indim,

    block_D,
    threads,
    dtype,
    reduce_type = "float32"
):
    @T.prim_func
    def main_fp16_prefill(
        Input: T.Tensor([Batch, Token, Indim], dtype),
        Cache: T.Tensor([Batch, Kernel_size, Indim], dtype),
        Kernel_input: T.Tensor([Batch, Token, Kernel_size, Indim], dtype),
        Output: T.Tensor([Batch, Token, Indim], dtype),
    ):
        with T.Kernel(Batch * Token, T.ceildiv(Indim, block_D), threads=threads) as (bx, by):
            Kernel_shared = T.alloc_shared([Kernel_size, block_D], dtype)
            Input_shared = T.alloc_shared([Kernel_size, block_D], dtype)

            Output_shared = T.alloc_shared([block_D], dtype)

            Input_reg = T.alloc_fragment([Kernel_size, block_D], reduce_type)
            Output_reg = T.alloc_fragment([Kernel_size, block_D], reduce_type)
            Output_reduced = T.alloc_fragment([block_D], reduce_type)
            Kernel_reg = T.alloc_fragment([Kernel_size, block_D], reduce_type)
            batch_id = bx // Token
            token_id = bx % Token
            # for i, j in T.Parallel(block_D, Kernel_size):
            #     Kernel_shared[i, j] = Kernel_input[bx, token_id, i + by * block_D, j]
            T.copy(Kernel_input[batch_id, token_id, 0:Kernel_size, by * block_D : by * block_D + block_D], Kernel_shared)

            ###
            
            T.copy(Cache[batch_id, 1:Kernel_size, by * block_D : by * block_D + block_D], Input_shared[0:Kernel_size - 1, 0:block_D])
            T.copy(Input[batch_id, token_id, by * block_D : by * block_D + block_D], Input_shared[Kernel_size - 1, 0:block_D])


            
            

            T.copy(Input_shared, Input_reg)
            T.copy(Kernel_shared, Kernel_reg)
            for i, j in T.Parallel(Kernel_size, block_D):
                Output_reg[i, j] = Input_reg[i, j] * Kernel_reg[i, j] 
            T.reduce_sum(Output_reg, Output_reduced, dim=0) 
            # for i in T.Parallel(block_D):
            #     Output_reduced[i] = Output_reduced[i] / (1 + T.exp(-Output_reduced[i]))
            T.copy(Output_reduced, Output_shared)
            
            # 正确的输出复制 - 将block_D大小的输出复制到正确的位置
            T.copy(Output_shared, Output[bx, token_id, by * block_D : by * block_D + block_D])
            
            # ### renew cache - 将当前输入复制到缓存的最后位置
            T.copy(Input_shared[0:Kernel_size, 0:block_D], Cache[batch_id, 0:Kernel_size, by * block_D : by * block_D + block_D])
            T.copy(Input_shared[Kernel_size - 1, 0:block_D], Cache[batch_id, Kernel_size - 1, by * block_D : by * block_D + block_D])
    return main_fp16_prefill

# Helper function to ensure tensors are contiguous for Triton
def ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    # Ensure tensor is contiguous in memory.
    return t if t.is_contiguous() else t.contiguous()

@triton.jit
def _causal_conv_step_kernel(
    # --- Input/Output Pointers ---
    X_ptr,         # Pointer to current input x [B, D] (after squeeze)
    Cache_ptr,     # Pointer to cache [B, D, W], updated IN-PLACE
    Kernels_ptr,   # Pointer to generated kernels [B, D, W]
    Out_ptr,       # Pointer to output tensor [B, D]

    # --- Tensor Dimensions ---
    B, D,          # Batch size, Feature dimension

    # --- Tensor Strides ---
    X_stride_b, X_stride_d,
    Cache_stride_b, Cache_stride_d, Cache_stride_w,
    Kernels_stride_b, Kernels_stride_d, Kernels_stride_w,
    Out_stride_b, Out_stride_d,

    # --- Kernel Meta-Parameters ---
    W: tl.constexpr,               # Kernel width (Cache size), passed as compile-time constant (1 < W <= 4)
    BLOCK_SIZE_D: tl.constexpr,    # Block size for D dimension (tuning parameter)
):
    """
    Triton kernel for a single step (T=1) of causal dynamic convolution.
    Updates the cache in-place and computes the output (without activation).
    Optimized for small W (1 < W <= 4) by manually unrolling the W dimension.
    Does NOT handle separate static bias.

    Grid: (B, cdiv(D, BLOCK_SIZE_D))
    Updates Cache[b, d, :] and computes Out[b, d].
    """
    # 1. --- Get Program IDs and Calculate Indices ---
    pid_b = tl.program_id(0)       # Program ID for batch dimension
    pid_d_block = tl.program_id(1) # Program ID for dimension block

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D # Shape: [BLOCK_SIZE_D]

    # 2. --- Load Current Input X ---
    x_ptrs = X_ptr + pid_b * X_stride_b + offs_d * X_stride_d
    x_curr = tl.load(x_ptrs, mask=d_mask, other=0.0) # Shape: [BLOCK_SIZE_D]

    # --- Initialize Accumulator ---
    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=x_curr.dtype) # Use input dtype

    # --- Manually Unroll Operations for W ---
    # We will load kernel values and cache values step-by-step
    # and perform the calculation and cache update.

    # --- Step w = 0 ---
    # Compute: cache_val_1 * k_val_0 (part 1)
    # Cache Update: store cache_val_1 at index 0
    if tl.constexpr(W > 1):
        # Load k_val_0
        k_ptr_0 = Kernels_ptr + pid_b * Kernels_stride_b + offs_d * Kernels_stride_d + 0 * Kernels_stride_w
        k_val_0 = tl.load(k_ptr_0, mask=d_mask, other=0.0)

        # Load cache_val_1 (needed for computation and storing at index 0)
        cache_ptr_1 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 1 * Cache_stride_w
        cache_val_1 = tl.load(cache_ptr_1, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_1 * k_val_0

        # Cache Update: Store cache_val_1 -> cache_ptr_0
        cache_ptr_0 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 0 * Cache_stride_w
        tl.store(cache_ptr_0, cache_val_1, mask=d_mask)

    # --- Step w = 1 ---
    # Compute: cache_val_2 * k_val_1 (part 1)
    # Cache Update: store cache_val_2 at index 1
    if tl.constexpr(W > 2):
        # Load k_val_1
        k_ptr_1 = Kernels_ptr + pid_b * Kernels_stride_b + offs_d * Kernels_stride_d + 1 * Kernels_stride_w
        k_val_1 = tl.load(k_ptr_1, mask=d_mask, other=0.0)

        # Load cache_val_2
        cache_ptr_2 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 2 * Cache_stride_w
        cache_val_2 = tl.load(cache_ptr_2, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_2 * k_val_1

        # Cache Update: Store cache_val_2 -> cache_ptr_1
        cache_ptr_1 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 1 * Cache_stride_w
        tl.store(cache_ptr_1, cache_val_2, mask=d_mask)

    # --- Step w = 2 ---
    # Compute: cache_val_3 * k_val_2 (part 1)
    # Cache Update: store cache_val_3 at index 2
    if tl.constexpr(W > 3):
        # Load k_val_2
        k_ptr_2 = Kernels_ptr + pid_b * Kernels_stride_b + offs_d * Kernels_stride_d + 2 * Kernels_stride_w
        k_val_2 = tl.load(k_ptr_2, mask=d_mask, other=0.0)

        # Load cache_val_3
        cache_ptr_3 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 3 * Cache_stride_w
        cache_val_3 = tl.load(cache_ptr_3, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_3 * k_val_2

        # Cache Update: Store cache_val_3 -> cache_ptr_2
        cache_ptr_2 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 2 * Cache_stride_w
        tl.store(cache_ptr_2, cache_val_3, mask=d_mask)

    # --- Final Step (Part 2 and Final Cache Update) ---
    # Compute: x_curr * k_val_{W-1} (part 2)
    # Cache Update: store x_curr at index W-1

    # Load k_val_{W-1}
    k_ptr_last = Kernels_ptr + pid_b * Kernels_stride_b + offs_d * Kernels_stride_d + (W - 1) * Kernels_stride_w
    k_val_last = tl.load(k_ptr_last, mask=d_mask, other=0.0)

    # Accumulate Part 2
    accumulator += x_curr * k_val_last

    # Final Cache Update: Store x_curr -> cache_ptr_{W-1}
    cache_ptr_last = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + (W - 1) * Cache_stride_w
    tl.store(cache_ptr_last, x_curr, mask=d_mask)

    # 6. --- Store Output ---
    out_ptrs = Out_ptr + pid_b * Out_stride_b + offs_d * Out_stride_d
    tl.store(out_ptrs, accumulator, mask=d_mask) # Store result without activation

    # Cache update is now fully handled within the unrolled steps.


# --- Python Wrapper Function ---
def causal_conv_step_triton(
    x: torch.Tensor,           # Input tensor [B, 1, D]
    cache: torch.Tensor,       # Cache tensor [B, D, W], modified in-place
    kernels: torch.Tensor,     # Kernels tensor [B, D, W]
) -> torch.Tensor:             # Returns output tensor [B, D] (before activation)
    """
    Performs one step of causal dynamic convolution using Triton.
    Updates the cache in-place. Does NOT fuse activation. Assumes 1 < W <= 4.
    Uses manually unrolled kernel for W dimension.

    Args:
        x: Current input token tensor of shape [B, 1, D].
        cache: Cache tensor of shape [B, D, W]. Will be updated in-place.
        kernels: Dynamically generated kernels tensor of shape [B, D, W].

    Returns:
        Output tensor of shape [B, D] for the current step (before activation).
    """
    # --- Input Validation and Preparation ---
    assert x.dim() == 3 and x.shape[1] == 1, "Input x must have shape [B, 1, D]"
    assert cache.dim() == 3, "Cache must have shape [B, D, W]"
    assert kernels.dim() == 3, "Kernels must have shape [B, D, W]"
    B, _, D = x.shape
    W = cache.shape[2]
    # Updated assertion: W must be > 1 and <= 4
    assert 1 < W <= 4, f"Kernel W={W}, this optimized version assumes 1 < W <= 4"
    assert cache.shape[0] == B and cache.shape[1] == D, f"Cache shape mismatch: {cache.shape}"
    assert kernels.shape == cache.shape, f"Kernels shape mismatch: {kernels.shape}"
    assert x.is_cuda and cache.is_cuda and kernels.is_cuda, "Inputs must be CUDA tensors"

    # Squeeze the time dimension from input x
    x_squeezed = x.squeeze(1) # Shape [B, D]

    # Ensure tensors are contiguous for correct stride calculations in Triton
    x_squeezed = ensure_contiguous(x_squeezed)
    # Cache MUST be contiguous for in-place updates and loads/stores to work reliably
    cache = ensure_contiguous(cache)
    kernels = ensure_contiguous(kernels)

    # Create output tensor with the same dtype as input x
    out = torch.empty_like(x_squeezed) # Shape [B, D]

    # --- Triton Kernel Launch ---
    grid = lambda meta: (B, triton.cdiv(D, meta['BLOCK_SIZE_D']))
    BLOCK_SIZE_D = 64 # Example, tune this value

    # Launch the kernel
    _causal_conv_step_kernel[grid](
        x_squeezed, cache, kernels, out,   # Tensor pointers
        B, D,                              # Dimensions
        x_squeezed.stride(0), x_squeezed.stride(1), # x strides
        cache.stride(0), cache.stride(1), cache.stride(2), # cache strides
        kernels.stride(0), kernels.stride(1), kernels.stride(2), # kernels strides
        out.stride(0), out.stride(1),      # out strides
        # --- Meta-parameters ---
        W=W,                               # Pass W as constexpr
        BLOCK_SIZE_D=BLOCK_SIZE_D,         # Pass BLOCK_SIZE_D as constexpr
    )

    return out # Return the computed output [B, D] (before activation)

def tilelang_dconv_step(input, cache, kernel_input):
    B, Token, Indim = input.shape
    Kernel_size = kernel_input.shape[-2]
    dtype = dtype_dict[input.dtype]
    kernel = _dconv_step_kernel(B, Token, Kernel_size, Indim, block_D = 128, threads = 128, dtype=dtype)
    output = kernel(input, cache, kernel_input)
    return output, cache
def determin_error(output_tilelang, output_triton_raw):
    diff = torch.abs(output_tilelang - output_triton_raw)
    diff_max = torch.max(diff).item()
    dfff_relative_max = torch.max(diff / (torch.abs(output_triton_raw) + 1e-6)).item()
    dfff_relative_mean = torch.mean(diff / (torch.abs(output_triton_raw) + 1e-6)).item()
    return dfff_relative_max, dfff_relative_mean
if __name__ == "__main__":
    batch = 1
    token = 1
    indim = 3072
    kernel_size = 4
    block_D = 64
    threads = 128
    dtype = "bfloat16"

    input1 = torch.randn(batch, token, indim).to(torch.bfloat16).cuda()
    kernel_input = torch.randn(batch, token,  kernel_size, indim).to(torch.bfloat16).cuda()
    cache = torch.randn(batch, kernel_size, indim).to(torch.bfloat16).cuda()
    output_triton = torch.randn(batch, token, indim).to(torch.bfloat16).cuda()

    input_tl = input1.clone()
    cache_tl = cache.clone()
    kernel_input_tl = kernel_input.clone()
    output_tilelang = output_triton.clone()

    # 检查输入数据
    print("=== 输入数据检查 ===")
    print(f"input1形状: {input1.shape}, 非零元素: {torch.count_nonzero(input1).item()}")
    print(f"kernel_input形状: {kernel_input.shape}, 非零元素: {torch.count_nonzero(kernel_input).item()}")
    print(f"cache形状: {cache.shape}, 非零元素: {torch.count_nonzero(cache).item()}")
    print(f"input1样本值: {input1[0, 0, :5]}")
    print(f"kernel样本值: {kernel_input[0, 0, :5, 0]}")
    
    # 创建TileLang kernel
    # kernel = _dconv_step_kernel(batch, token, kernel_size, indim, dtype=dtype)
    
    # 运行TileLang kernel
    print("\n=== 运行TileLang kernel ===")
    # print(f"tilelang cache before: {cache_tl.squeeze()}")
    # output, cache = tilelang_dconv_step(input_tl, cache_tl, kernel_input_tl)
    # print(f"tilelang cache before: {cache_tl.squeeze()}")
    output_tilelang, cache_tl = tilelang_dconv_step(input_tl, cache_tl, kernel_input_tl)
    print(f"TileLang输出: {output_tilelang.squeeze()}")
    # print(f"tilelang cache: {cache_tl.squeeze()}")
    # print(f"tilelang input: {input_tl.squeeze()}")
    
    # print(f"输出非零元素: {torch.count_nonzero(output_tilelang).item()}")
    
    # 准备并运行Triton kernel
    # print("\n=== 运行Triton kernel ===")
    x_triton = input1  # [B, 1, D]
    cache_triton = cache.permute(0, 2, 1).contiguous()  # [B, D, W]
    kernels_triton = kernel_input.permute(0, 1, 3, 2).contiguous()[:, 0, :, :]  # [B, D, W]
    
    output_triton_raw = causal_conv_step_triton(x_triton, cache_triton, kernels_triton)
    # output_triton_raw = nn.SiLU()(output_triton_raw)
    print(f"Triton输出: {output_triton_raw.squeeze()}")
    # print(f"Triton cache: {cache_triton.permute(0, 2, 1)}")
    # print(f"输出非零元素: {torch.count_nonzero(output_triton_raw).item()}")
    print(f"TileLang和Triton的相对误差(output): {determin_error(output_tilelang, output_triton_raw)}")
    print(f"TileLang和Triton的相对误差(cache): {determin_error(cache_tl, cache_triton.permute(0, 2, 1))}")
    # # 比较结果
    # if output_tilelang.squeeze().shape == output_triton_raw.squeeze().shape:
    #     diff = torch.abs(output_tilelang.squeeze() - output_triton_raw.squeeze())
    #     print(f"\n=== 比较结果 ===")
    #     print(f"差异: {diff}")
    #     print(f"最大差异: {torch.max(diff).item():.8f}")
    #     print(f"平均差异: {torch.mean(diff).item():.8f}")
    # for i in range(10):
    #     kernel(input1, cache, kernel_input, output_triton)
    # torch.cuda.synchronize()
    # start_time = time.time()
    # for i in range(10):
    #     kernel(input1, cache, kernel_input, output_triton)
    # torch.cuda.synchronize()
    # print(f"Time taken: {time.time() - start_time} seconds")

            
            


