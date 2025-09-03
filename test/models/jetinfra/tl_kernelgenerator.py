import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
import time
import torch.nn as nn

def get_configs():
    block_M = [32, 64, 128, 256]  
    block_K = [64, 128, 256] 
    block_N = [64, 128, 256]  
    num_stages = [1, 2, 3, 4]  
    threads = [128, 256]
    _configs = list(itertools.product(block_M, block_K, block_N, num_stages, threads))
    configs = [{
        'block_M': c[0],
        'block_K': c[1],
        'block_N': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def tl_linear_w_silu(
    Batch,
    Token,
    Indim,
    outdim,
    dtype,
    block_M,
    block_K,
    block_N,
    num_stages,
    threads,
    reduce_dtype = "float",
    out_dtype = None,
):
    @ T.macro
    def silu(
        buffer: T.FragmentBuffer([block_M, block_N], reduce_dtype),
    ):
        for i, j in T.Parallel(block_M, block_N):
            buffer[i, j] = buffer[i, j] / (1 + T.exp(-buffer[i, j]))

    from tilelang.carver.arch import driver
    sm_num = driver.get_num_sms()

    @T.prim_func
    def main_fp16(
        Input: T.Tensor([Batch * Token, Indim], dtype),
        W_T: T.Tensor([outdim, Indim], dtype),
        Output: T.Tensor([Batch * Token, outdim], dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            Input_shared = T.alloc_shared((block_M, block_K), dtype)
            W_T_shared = T.alloc_shared((block_N, block_K), dtype)
            output_reg = T.alloc_fragment((block_M, block_N), reduce_dtype)
            output_shared = T.alloc_shared((block_M, block_N), dtype)

            T.annotate_layout({
                output_shared: tilelang.layout.make_swizzled_layout(output_shared)
            })

            for bx, by in T.Persistent(
                [T.ceildiv(Batch * Token, block_M), T.ceildiv(outdim, block_N)], sm_num, block_id):
                T.clear(output_reg)

                for k in T.Pipelined(T.ceildiv(Indim, block_K), num_stages=num_stages):
                    T.copy(Input[bx * block_M, k * block_K], Input_shared)
                    T.copy(W_T[by * block_N, k * block_K], W_T_shared)
                    T.gemm(Input_shared, W_T_shared, output_reg, transpose_B=True)
    
                silu(output_reg)
                T.copy(output_reg, output_shared)
                T.copy(output_shared, Output[bx * block_M, by * block_N])

    @T.prim_func
    def main_fp8(
        Input: T.Tensor([Batch * Token, Indim], dtype),
        W_T: T.Tensor([outdim, Indim], dtype),
        Output: T.Tensor([Batch * Token, outdim], out_dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            Input_shared = T.alloc_shared((block_M, block_K), dtype)
            W_T_shared = T.alloc_shared((block_N, block_K), dtype)
            output_reg = T.alloc_fragment((block_M, block_N), reduce_dtype)
            output_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.annotate_layout({
                output_shared: tilelang.layout.make_swizzled_layout(output_shared)
            })

            for bx, by in T.Persistent(
                [T.ceildiv(Batch * Token, block_M), T.ceildiv(outdim, block_N)], sm_num, block_id):
                T.clear(output_reg)

                for k in T.Pipelined(T.ceildiv(Indim, block_K), num_stages=num_stages):
                    T.copy(Input[bx * block_M, k * block_K], Input_shared)
                    T.copy(W_T[by * block_N, k * block_K], W_T_shared)
                    T.gemm(Input_shared, W_T_shared, output_reg, transpose_B=True)
    
                silu(output_reg)
                T.copy(output_reg, output_shared)
                T.copy(output_shared, Output[bx * block_M, by * block_N])
    
    if dtype == "float16":
        return main_fp16
    elif "float8" in dtype:
        return main_fp8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}") 



@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit()
def tl_linear(
    Batch,
    Token,
    Indim,
    outdim,
    dtype,
    block_M,
    block_K,
    block_N,
    num_stages,
    threads,
    reduce_dtype = "float",
    out_dtype = None,
):

    from tilelang.carver.arch import driver
    sm_num = driver.get_num_sms()

    @T.prim_func
    def main_fp16(
        Input: T.Tensor([Batch * Token, Indim], dtype),
        W_T: T.Tensor([outdim, Indim], dtype),
        Output: T.Tensor([Batch * Token, outdim], dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            Input_shared = T.alloc_shared((block_M, block_K), dtype)
            W_T_shared = T.alloc_shared((block_N, block_K), dtype)
            output_reg = T.alloc_fragment((block_M, block_N), reduce_dtype)
            output_shared = T.alloc_shared((block_M, block_N), dtype)

            T.annotate_layout({
                output_shared: tilelang.layout.make_swizzled_layout(output_shared)
            })

            for bx, by in T.Persistent(
                [T.ceildiv(Batch * Token, block_M), T.ceildiv(outdim, block_N)], sm_num, block_id):
                T.clear(output_reg)

                for k in T.Pipelined(T.ceildiv(Indim, block_K), num_stages=num_stages):
                    T.copy(Input[bx * block_M, k * block_K], Input_shared)
                    T.copy(W_T[by * block_N, k * block_K], W_T_shared)
                    T.gemm(Input_shared, W_T_shared, output_reg, transpose_B=True)
    
                T.copy(output_reg, output_shared)
                T.copy(output_shared, Output[bx * block_M, by * block_N])

    @T.prim_func
    def main_fp8(
        Input: T.Tensor([Batch * Token, Indim], dtype),
        W_T: T.Tensor([outdim, Indim], dtype),
        Output: T.Tensor([Batch * Token, outdim], out_dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            Input_shared = T.alloc_shared((block_M, block_K), dtype)
            W_T_shared = T.alloc_shared((block_N, block_K), dtype)
            output_reg = T.alloc_fragment((block_M, block_N), reduce_dtype)
            output_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.annotate_layout({
                output_shared: tilelang.layout.make_swizzled_layout(output_shared)
            })

            for bx, by in T.Persistent(
                [T.ceildiv(Batch * Token, block_M), T.ceildiv(outdim, block_N)], sm_num, block_id):
                T.clear(output_reg)

                for k in T.Pipelined(T.ceildiv(Indim, block_K), num_stages=num_stages):
                    T.copy(Input[bx * block_M, k * block_K], Input_shared)
                    T.copy(W_T[by * block_N, k * block_K], W_T_shared)
                    T.gemm(Input_shared, W_T_shared, output_reg, transpose_B=True)
    
                T.copy(output_reg, output_shared)
                T.copy(output_shared, Output[bx * block_M, by * block_N])
    
    if dtype == "float16":
        return main_fp16
    elif "float8" in dtype:
        return main_fp8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
def linear_w_silu(input, weight_t, dtype="float16", reduced_dtype="float", out_dtype="float16"):
    B, Token, D_in = input.shape
    D_out = weight_t.shape[0]
    device = input.device
    kernel = tl_linear_w_silu(B, Token, D_in, D_out, dtype=dtype, reduce_dtype=reduced_dtype, out_dtype=out_dtype)

    # Map tilelang dtype string to torch dtype string for output tensor creation
    if out_dtype == "float8_e4m3":
        torch_out_dtype_str = "float8_e4m3fn"
    else:
        torch_out_dtype_str = out_dtype

    torch_dtype = getattr(torch, torch_out_dtype_str)
    output = torch.empty(B, Token, D_out, device=device, dtype=torch_dtype)
    kernel(input.view(B * Token, D_in), weight_t, output.view(B * Token, D_out))
    return output

def conv_kernel_generator(input, weight1, weight2, dtype="float16"):
    B, Token, D_in = input.shape
    hidden_size = weight1.shape[0]
    D_out = weight2.shape[0]
    device = input.device
    torch_dtype = torch.float16 if dtype == "float16" else torch.float8_e4m3fn
    kernel1 = tl_linear_w_silu(B, Token, D_in, hidden_size, dtype=dtype, out_dtype=dtype)
    kernel2 = tl_linear(B, Token, hidden_size, D_out, dtype=dtype, out_dtype=dtype)
    output1 = torch.empty(B, Token, hidden_size, device=device, dtype=torch_dtype)
    output2 = torch.empty(B, Token, D_out, device=device, dtype=torch_dtype)
    kernel1(input.view(B * Token, D_in), weight1, output1.view(B * Token, hidden_size))
    kernel2(output1.view(B * Token, hidden_size), weight2, output2.view(B * Token, D_out))
    return output2


if __name__ == "__main__":
    input = torch.randn(1, 1024, 1024, dtype=torch.float16).to("cuda")
    weight1 = torch.randn(1024, 1024, dtype=torch.float16).to("cuda")
    weight2 = torch.randn(1024, 1024, dtype=torch.float16).to("cuda")
    output = conv_kernel_generator(input, weight1, weight2, dtype="float16")
    print(output.shape)
    print(output)