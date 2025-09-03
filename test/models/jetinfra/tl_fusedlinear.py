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


def linear_w_mul(
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
    # @ T.macro
    # def silu(
    #     buffer: T.FragmentBuffer([block_M, block_N], reduce_dtype),
    # ):
    #     for i, j in T.Parallel(block_M, block_N):
    #         buffer[i, j] = buffer[i, j] / (1 + T.exp(-buffer[i, j]))

    from tilelang.carver.arch import driver
    sm_num = driver.get_num_sms()

    @T.prim_func
    def main_fp16(
        Input: T.Tensor([Batch * Token, Indim], dtype),
        W_T: T.Tensor([outdim, Indim], dtype),
        Mask: T.Tensor([Batch * Token], dtype),
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
    
                # silu(output_reg)
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
def linear_w_silu(input, weight_t, dtype="float16", reduced_dtype="float", out_dtype="float16"):
    B, Token, D_in = input.shape
    D_out = weight_t.shape[0]
    device = input.device
    kernel = linear_w_silu(B, Token, D_in, D_out, dtype=dtype, reduce_dtype=reduced_dtype, out_dtype=out_dtype)

    # Map tilelang dtype string to torch dtype string for output tensor creation
    if out_dtype == "float8_e4m3":
        torch_out_dtype_str = "float8_e4m3fn"
    else:
        torch_out_dtype_str = out_dtype

    torch_dtype = getattr(torch, torch_out_dtype_str)
    output = torch.empty(B, Token, D_out, device=device, dtype=torch_dtype)
    kernel(input.view(B * Token, D_in), weight_t, output.view(B * Token, D_out))
    return output


if __name__ == "__main__":
    B, Token, D_in, D_out = 40, 1, 1536, 1152

    # =================
    # FP16 Benchmarking
    # =================
    print("Benchmarking FP16...")
    Input_fp16 = torch.randn(B, Token, D_in).cuda().half()
    W_T_fp16 = torch.randn(D_out, D_in).cuda().half()

    # --- Tile-lang implementation ---
    # Warm up
    for _ in range(10):
        output_tl_fp16 = linear_w_silu_prefill(Input_fp16, W_T_fp16, dtype="float16", out_dtype="float16")
    torch.cuda.synchronize()
    # Benchmark
    start = time.time()
    for _ in range(20):
        output_tl_fp16 = linear_w_silu_prefill(Input_fp16, W_T_fp16, dtype="float16", out_dtype="float16")
    torch.cuda.synchronize()
    end = time.time()
    tl_fp16_time = (end - start) * 1000 / 20
    print(f"Tilelang FP16 Time taken: {tl_fp16_time:.4f} ms")

    # --- PyTorch implementation ---
    # Warm up
    for _ in range(10):
        output_pt_fp16 = Input_fp16.view(B * Token, D_in) @ W_T_fp16.T
        output_pt_fp16 = nn.functional.silu(output_pt_fp16)
    torch.cuda.synchronize()
    # Benchmark
    start = time.time()
    for _ in range(20):
        output_pt_fp16 = Input_fp16.view(B * Token, D_in) @ W_T_fp16.T
        output_pt_fp16 = nn.functional.silu(output_pt_fp16)
    torch.cuda.synchronize()
    end = time.time()
    pt_fp16_time = (end - start) * 1000 / 20
    print(f"Pytorch FP16 Time taken: {pt_fp16_time:.4f} ms")

    # --- Correctness Check ---
    output_tl_fp16_reshaped = output_tl_fp16.view(B * Token, D_out)
    torch.testing.assert_close(output_tl_fp16_reshaped, output_pt_fp16, rtol=1e-2, atol=1e-2)
    relative_error_fp16 = torch.mean(torch.abs(output_tl_fp16_reshaped - output_pt_fp16) / (torch.abs(output_pt_fp16) + 1e-6))
    print(f"FP16 Average Relative Error: {relative_error_fp16.item():.6f}")

    # =================
    print("\nBenchmarking FP8...")
    Input_fp8 = torch.randn(B, Token, D_in).cuda().to(torch.float8_e4m3fn)
    W_T_fp8 = torch.randn(D_out, D_in).cuda().to(torch.float8_e4m3fn)

    # --- Tile-lang implementation ---
    # Warm up
    for _ in range(10):
        output_tl_fp8 = linear_w_silu_prefill(Input_fp8, W_T_fp8, dtype="float8_e4m3", out_dtype="float8_e4m3")
    torch.cuda.synchronize()
    # Benchmark
    start = time.time()
    for _ in range(20):
        output_tl_fp8 = linear_w_silu_prefill(Input_fp8, W_T_fp8, dtype="float8_e4m3", out_dtype="float8_e4m3")
    torch.cuda.synchronize()
    end = time.time()
    tl_fp8_time = (end - start) * 1000 / 20
    print(f"Tilelang FP8 Time taken: {tl_fp8_time:.4f} ms")

    # --- PyTorch FP16 implementation for reference ---
    output_pt_ref_for_fp8 = Input_fp8.view(B * Token, D_in).to(torch.float16) @ W_T_fp8.T.to(torch.float16)
    output_pt_ref_for_fp8 = nn.functional.silu(output_pt_ref_for_fp8)

    # --- Correctness Check ---
    output_tl_fp8_reshaped = output_tl_fp8.view(B*Token, D_out).to(torch.float16)
    # Using a higher tolerance for FP8 vs FP16 comparison
    torch.testing.assert_close(output_tl_fp8_reshaped, output_pt_ref_for_fp8, rtol=0.1, atol=0.1)
    relative_error_fp8_vs_fp16 = torch.mean(torch.abs(output_tl_fp8_reshaped - output_pt_ref_for_fp8) / (torch.abs(output_pt_ref_for_fp8) + 1e-6))
    print(f"FP8 (Tilelang) vs FP16 (PyTorch) Average Relative Error: {relative_error_fp8_vs_fp16.item():.6f}")