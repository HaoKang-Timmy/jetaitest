import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
import time
import torch.nn as nn

def get_configs():
    block_M = [32,64, 128]  # 因为 batch_size * token = 32 * 1 = 32
    block_K = [64, 128, 256]  # 对应 D_in = 1024
    block_N = [64, 128]  # 对应 D_out = 512
    num_stages = [1, 2, 3, 4]  # 减少配置数量，专注于更可能成功的配置
    threads = [128, 256]
    split_k = [1, 2, 4, 8]
    _configs = list(itertools.product(block_M, block_K, block_N, num_stages, threads, split_k))
    configs = [{
        'block_M': c[0],
        'block_K': c[1],
        'block_N': c[2],
        'num_stages': c[3],
        'threads': c[4],
        'split_k': c[5]
    } for c in _configs]
    return configs

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def linear_w_silu(
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
    split_k,
    reduce_dtype = "float",
    out_dtype = None,
):
    splitK = Indim // split_k
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

            for bx, by, bz in T.Persistent(
                [T.ceildiv(Batch * Token, block_M), T.ceildiv(outdim, block_N), split_k], sm_num, block_id):
                T.clear(output_reg)

                for k in T.Pipelined(T.ceildiv(Indim, block_K), num_stages=num_stages):
                    T.copy(Input[bx * block_M, bz * splitK + k * block_K], Input_shared)
                    T.copy(W_T[bz * splitK + by * block_N, k * block_K], W_T_shared)
                    T.gemm(Input_shared, W_T_shared, output_reg, transpose_B=True)
    
                silu(output_reg)
                T.copy(output_reg, output_shared)
                # T.copy(output_shared, Output[bx * block_M, by * block_N])
                T.atomic_add(Output[bx * block_M, by * block_N], output_shared)

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
    # @T.prim_func
    # def main_fp8_splitk(
    #     Input: T.Tensor([Batch * Token, Indim], dtype),
    #     W_T: T.Tensor([outdim, Indim], dtype),
    #     Output: T.Tensor([Batch * Token, outdim], out_dtype),
    # ):
    #     with T.Kernel(sm_num, threads=threads) as (block_id):
    #         Input_shared = T.alloc_shared((block_M, block_K), dtype)
    #         W_T_shared = T.alloc_shared((block_N, block_K), dtype)
    #         output_reg = T.alloc_fragment((block_M, block_N), reduce_dtype)
    #         output_shared = T.alloc_shared((block_M, block_N), out_dtype)

    #         T.annotate_layout({
    #             output_shared: tilelang.layout.make_swizzled_layout(output_shared)
    #         })

    #         for bx, by in T.Persistent(
    #             [T.ceildiv(Batch * Token, block_M), T.ceildiv(outdim, block_N)], sm_num, block_id):
    #             T.clear(output_reg)

    #             for k in T.Pipelined(T.ceildiv(Indim, block_K), num_stages=num_stages):
    #                 T.copy(Input[bx * block_M, k * block_K], Input_shared)
    #                 T.copy(W_T[by * block_N, k * block_K], W_T_shared)
    #                 T.gemm(Input_shared, W_T_shared, output_reg, transpose_B=True)
    
    #             silu(output_reg)
    #             T.copy(output_reg, output_shared)
    #             T.copy(output_shared, Output[bx * block_M, by * block_N])
    if dtype == "float16":
        return main_fp16
    elif "float8" in dtype:
        return main_fp8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def fused_linear_w_silu(
    Input: torch.Tensor,
    W_T: torch.Tensor,
    Output: torch.Tensor,
):
    Output = linear_w_silu(Input, W_T, Output)
    Output = nn.functional.silu(Output)
    return Output

if __name__ == "__main__":
    B, Token, D_in, D_out = 32, 1, 1536, 1152

    print("Benchmarking FP16...")
    # Tile-lang implementation
    Input_fp16 = torch.randn(B, Token, D_in).cuda().half()
    W_T_fp16 = torch.randn(D_out, D_in).cuda().half()
    Output_fp16 = torch.empty(B, Token, D_out).cuda().half()
    
    kernel_fp16 = linear_w_silu(B, Token, D_in, D_out, "float16", out_dtype="float16")
    # warm up
    for _ in range(10):
        kernel_fp16(Input_fp16.view(B * Token, D_in), W_T_fp16, Output_fp16.view(B * Token, D_out))
    # benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        kernel_fp16(Input_fp16.view(B * Token, D_in), W_T_fp16, Output_fp16.view(B * Token, D_out))
    torch.cuda.synchronize()
    end = time.time()
    print(f"Tilelang FP16 Time taken: {(end - start) * 1000 / 20} ms")

    # PyTorch implementation
    # warm up
    for _ in range(10):
        output = Input_fp16.view(B * Token, D_in) @ W_T_fp16.T
        output = nn.functional.silu(output)
    # benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        output = Input_fp16.view(B * Token, D_in) @ W_T_fp16.T
        output = nn.functional.silu(output)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Pytorch FP16 Time taken: {(end - start) * 1000 / 20} ms")

    print("\nBenchmarking FP8...")
    # Tile-lang implementation
    Input_fp8 = torch.randn(B, Token, D_in).cuda().to(torch.float8_e4m3fn)
    W_T_fp8 = torch.randn(D_out, D_in).cuda().to(torch.float8_e4m3fn)
    Output_fp8 = torch.empty(B, Token, D_out).cuda().to(torch.float8_e4m3fn)

    kernel_fp8 = linear_w_silu(B, Token, D_in, D_out, "float8_e4m3", reduce_dtype="float", out_dtype="float8_e4m3")
    # warm up
    for _ in range(10):
        kernel_fp8(Input_fp8.view(B * Token, D_in), W_T_fp8, Output_fp8.view(B * Token, D_out))
    # benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        kernel_fp8(Input_fp8.view(B * Token, D_in), W_T_fp8, Output_fp8.view(B * Token, D_out))
    torch.cuda.synchronize()
    end = time.time()
    print(f"Tilelang FP8 Time taken: {(end - start) * 1000 / 20} ms")