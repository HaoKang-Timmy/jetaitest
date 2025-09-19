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
from dconv_step import causal_conv_step_triton

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
    # out_idx = [-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def _dconv_step_kernel(
    Batch,
    Token,
    Indim,
    Kernel_size,
    block_D,
    threads,
    dtype,
    reduce_type = "float32"
):
    @T.prim_func
    def main_fp16_prefill(
        Input: T.Tensor([Batch, Token, Indim], dtype),
        Cache: T.Tensor([Batch, Indim, Kernel_size], dtype),
        Kernel_input: T.Tensor([Batch, Token, Indim, Kernel_size], dtype),
        Output: T.Tensor([Batch, Token, Indim], dtype),
    ):
        with T.Kernel(Batch * Token, T.ceildiv(Indim, block_D), threads=threads) as (bx, by):
            Kernel_shared = T.alloc_shared([block_D, Kernel_size], dtype)
            Input_shared = T.alloc_shared([block_D, Kernel_size], dtype)
            Output_shared = T.alloc_shared([block_D], dtype)

            Input_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            Output_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            Output_reduced = T.alloc_fragment([block_D], reduce_type)
            Kernel_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            batch_id = bx // Token
            token_id = bx % Token
            # for i, j in T.Parallel(block_D, Kernel_size):
            #     Kernel_shared[i, j] = Kernel_input[bx, token_id, i + by * block_D, j]
            T.copy(Kernel_input[bx, token_id, by * block_D, 0], Kernel_shared)
            # T.copy(Input[batch_id, token_id, by * block_D],Input_shared[])
            for i, j in T.Parallel(block_D, Kernel_size):
                Input_shared[i, j] = T.if_then_else(token_id + 1 + j - Kernel_size >= 0, 
                    Input[bx, token_id, i + by * block_D], Cache[bx, i + by * block_D, j+1])

            T.copy(Input_shared, Input_reg)
            T.copy(Kernel_shared, Kernel_reg)
            for i, j in T.Parallel(block_D, Kernel_size):
                Output_reg[i, j] = Input_reg[i, j] * Kernel_reg[i, j] 
            T.reduce_sum(Output_reg, Output_reduced, dim=1) 
            # for i in T.Parallel(block_D):
            #     Output_reduced[i] = Output_reduced[i] / (1 + T.exp(-Output_reduced[i]))
            T.copy(Output_reduced, Output_shared)
            # 正确的输出复制 - 将block_D大小的输出复制到正确的位置
            T.copy(Output_shared, Output[bx, token_id, by * block_D : by * block_D + block_D])
            
            ### renew cache - 将当前输入复制到缓存的最后位置
            for i, j in T.Parallel(block_D, Kernel_size):
                Cache[bx, i + by * block_D, j] = Input_shared[i, j]
    return main_fp16_prefill

if __name__ == "__main__":
    batch = 1
    token = 1
    indim = 64
    kernel_size = 4
    block_D = 64
    threads = 128
    dtype = "float16"

    input1 = torch.randn(batch, token, indim).to(torch.float16).cuda()
    kernel_input = torch.randn(batch, token, indim, kernel_size).to(torch.float16).cuda()
    cache = torch.randn(batch, indim, kernel_size).to(torch.float16).cuda()
    output_triton = torch.randn(batch, token, indim).to(torch.float16).cuda()

    input2 = input1.clone()
    cache2 = cache.clone()
    kernel_input2 = kernel_input.clone()
    output_tilelang = output_triton.clone()

    # 检查输入数据
    print("=== 输入数据检查 ===")
    print(f"input1形状: {input1.shape}, 非零元素: {torch.count_nonzero(input1).item()}")
    print(f"kernel_input形状: {kernel_input.shape}, 非零元素: {torch.count_nonzero(kernel_input).item()}")
    print(f"cache形状: {cache.shape}, 非零元素: {torch.count_nonzero(cache).item()}")
    print(f"input1样本值: {input1[0, 0, :5]}")
    print(f"kernel样本值: {kernel_input[0, 0, :5, 0]}")
    
    # 创建TileLang kernel
    kernel = _dconv_step_kernel(batch, token, indim, kernel_size, block_D, threads, dtype=dtype)
    
    # 运行TileLang kernel
    print("\n=== 运行TileLang kernel ===")
    kernel(input2, cache2, kernel_input2, output_tilelang)
    print(f"TileLang输出: {output_tilelang.squeeze()}")
    print(f"tilelang cache: {cache2.squeeze()}")
    # print(f"输出非零元素: {torch.count_nonzero(output_tilelang).item()}")
    
    # 准备并运行Triton kernel
    print("\n=== 运行Triton kernel ===")
    x_triton = input1  # [B, 1, D]
    cache_triton = cache.permute(0, 1, 2).contiguous()  # [B, D, W]
    kernels_triton = kernel_input[:, 0, :, :].contiguous()  # [B, D, W]
    
    output_triton_raw = causal_conv_step_triton(x_triton, cache_triton, kernels_triton)
    print(f"Triton输出: {output_triton_raw.squeeze()}")
    print(f"Triton cache: {cache_triton.squeeze()}")
    # print(f"输出非零元素: {torch.count_nonzero(output_triton_raw).item()}")
    
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

            
            


