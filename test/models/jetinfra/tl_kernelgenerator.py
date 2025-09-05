import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
import time
import torch.nn as nn

def get_configs():
    block_M = [32, 64, 128]  
    block_K = [64, 128] 
    block_N = [64, 128]  
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
    in_mat = input.view(B * Token, D_in).contiguous()
    # w1 = weight1.contiguous()
    out1_mat = output1.view(B * Token, hidden_size).contiguous()
    kernel1(in_mat, weight1, out1_mat)

    # w2 = weight2.contiguous()
    # out2_mat = output2.view(B * Token, D_out).contiguous()
    # kernel2(out1_mat, weight2, out2_mat)
    # return output2
    return output1


if __name__ == "__main__":
    # Problem size
    B, Token, D_in = 40, 1, 1536
    hidden_size = 384
    D_out = 12288

    # =================
    # FP16 Benchmarking
    # =================
    print("Benchmarking FP16...")
    Input_fp16 = torch.randn(B, Token, D_in, dtype=torch.float16).to("cuda")
    W1_fp16 = torch.randn(hidden_size, D_in, dtype=torch.float16).to("cuda")
    W2_fp16 = torch.randn(D_out, hidden_size, dtype=torch.float16).to("cuda")

    # --- Tile-lang implementation ---
    for _ in range(10):
        out_tl_fp16 = conv_kernel_generator(Input_fp16, W1_fp16, W2_fp16, dtype="float16")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        out_tl_fp16 = conv_kernel_generator(Input_fp16, W1_fp16, W2_fp16, dtype="float16")
    torch.cuda.synchronize()
    end = time.time()
    tl_fp16_time = (end - start) * 1000 / 20
    print(f"Tilelang FP16 Time taken: {tl_fp16_time:.4f} ms")

    # --- PyTorch implementation ---
    for _ in range(10):
        out1 = Input_fp16.view(B * Token, D_in) @ W1_fp16.T
        out1 = nn.functional.silu(out1)
        out_pt_fp16 = out1 @ W2_fp16.T
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        out1 = Input_fp16.view(B * Token, D_in) @ W1_fp16.T
        out1 = nn.functional.silu(out1)
        out_pt_fp16 = out1 @ W2_fp16.T
    torch.cuda.synchronize()
    end = time.time()
    pt_fp16_time = (end - start) * 1000 / 20
    print(f"PyTorch FP16 Time taken: {pt_fp16_time:.4f} ms")

    # --- Correctness Check ---
    # out_tl_fp16_2d = out_tl_fp16.view(B * Token, D_out)
    # torch.testing.assert_close(out_tl_fp16_2d, out_pt_fp16, rtol=1e-2, atol=1e-2)
    # rel_err_fp16 = torch.mean(torch.abs(out_tl_fp16_2d - out_pt_fp16) / (torch.abs(out_pt_fp16) + 1e-6))
    # print(f"FP16 Average Relative Error: {rel_err_fp16.item():.6f}")
    out_tl_fp16_2d = out_tl_fp16.view(B * Token, -1)
    torch.testing.assert_close(out_tl_fp16_2d, out1, rtol=1e-2, atol=1e-2)
    rel_err_fp16 = torch.mean(torch.abs(out_tl_fp16_2d - out1) / (torch.abs(out1) + 1e-6))
    print(f"FP16 Average Relative Error: {rel_err_fp16.item():.6f}")

    # # =================
    # # FP8 Benchmarking
    # # =================
    # print("\nBenchmarking FP8...")
    # Input_fp8 = torch.randn(B, Token, D_in, dtype=torch.float16).to("cuda").to(torch.float8_e4m3fn)
    # W1_fp8 = torch.randn(hidden_size, D_in, dtype=torch.float16).to("cuda").to(torch.float8_e4m3fn)
    # W2_fp8 = torch.randn(D_out, hidden_size, dtype=torch.float16).to("cuda").to(torch.float8_e4m3fn)

    # # --- Tile-lang implementation ---
    # for _ in range(10):
    #     out_tl_fp8 = conv_kernel_generator(Input_fp8, W1_fp8, W2_fp8, dtype="float8_e4m3")
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(20):
    #     out_tl_fp8 = conv_kernel_generator(Input_fp8, W1_fp8, W2_fp8, dtype="float8_e4m3")
    # torch.cuda.synchronize()
    # end = time.time()
    # tl_fp8_time = (end - start) * 1000 / 20
    # print(f"Tilelang FP8 Time taken: {tl_fp8_time:.4f} ms")

    # # --- PyTorch reference (FP16) ---
    # out1_ref = Input_fp8.view(B * Token, D_in).to(torch.float16) @ W1_fp8.T.to(torch.float16)
    # out1_ref = nn.functional.silu(out1_ref)
    # out_pt_ref_fp8 = out1_ref @ W2_fp8.T.to(torch.float16)

    # # --- Correctness Check ---
    # out_tl_fp8_2d = out_tl_fp8.view(B * Token, D_out).to(torch.float16)
    # torch.testing.assert_close(out_tl_fp8_2d, out_pt_ref_fp8, rtol=0.1, atol=0.1)
    # rel_err_fp8 = torch.mean(torch.abs(out_tl_fp8_2d - out_pt_ref_fp8) / (torch.abs(out_pt_ref_fp8) + 1e-6))
    # print(f"FP8 (Tilelang) vs FP16 (PyTorch) Average Relative Error: {rel_err_fp8.item():.6f}")