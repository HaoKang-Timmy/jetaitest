import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
import time
import torch.nn as nn
from torch.nn import functional as F
dtype_dict = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3",
    torch.float8_e5m2: "float8_e5m2",
}
def get_configs():
    block_M = [64, 128]
    block_K = [64, 128]
    block_N = [128]
    num_stages = [1, 2, 3, 4]
    threads = [128, 256]
    # block_M = [64, 128]  
    # block_K = [64, 128] 
    # block_N = [64, 128]  
    # num_stages = [ 3, 4]  
    # threads = [128, 256]
    _configs = list(itertools.product(block_M, block_K, block_N, num_stages, threads))
    configs = [{
        'block_M': c[0],
        'block_K': c[1],
        'block_N': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs

# @autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx = [-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    }
)
def _linear_kernel(
    Batch,
    Token,
    Indim,
    outdim,
    dtype,
    block_M = 64,
    block_K = 64,
    block_N = 128,
    num_stages = 2,
    threads = 128,
    reduce_dtype = "float32",
):
    @T.macro
    def L2Norm_QK(
        QK: T.FragmentBuffer([block_M, block_N],reduce_dtype),
    ):
        squared_reg = T.alloc_fragment([block_M, block_N], reduce_dtype)
        sum_reg = T.alloc_fragment([block_M], reduce_dtype)
        
        
        for i, j in T.Parallel(block_M, block_N):
            squared_reg[i, j] = QK[i, j] * QK[i, j]
        T.reduce_sum(squared_reg, sum_reg, dim=1)
        # for i in T.Parallel(block_M):
        #     sum_reg[i] = T.sqrt(sum_reg[i]) + 1e-6
        
        
        # for i, j in T.Parallel(block_M, block_N):
        #     QK[i, j] = QK[i, j] / sum_reg[i]
        for i, j in T.Parallel(block_M, block_N):
            QK[i, j] = QK[i, j] * T.rsqrt(sum_reg[i]) 
    @ T.macro
    def silu(
        buffer: T.FragmentBuffer([block_M, block_N], reduce_dtype),
    ):
        for i, j in T.Parallel(block_M, block_N):
            buffer[i, j] = buffer[i, j] / (1 + T.exp(-buffer[i, j]))
    from tilelang.carver.arch import driver
    sm_num = driver.get_num_sms()
    @T.prim_func
    def main(
        Input: T.Tensor([Batch * Token, Indim], dtype),
        W_T: T.Tensor([outdim, Indim], dtype),
        Output: T.Tensor([Batch * Token, outdim], dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            Input_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            W_T_shared = T.alloc_shared((block_N, block_K), dtype, scope="shared")
            output_reg = T.alloc_fragment((block_M, block_N), reduce_dtype)
            output_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared")

            T.annotate_layout({
                Input_shared: tilelang.layout.make_swizzled_layout(Input_shared),
                W_T_shared: tilelang.layout.make_swizzled_layout(W_T_shared),
                output_shared: tilelang.layout.make_swizzled_layout(output_shared)
            })
            T.disable_warp_group_reg_alloc()
            for bx, by in T.Persistent(
                [T.ceildiv(Batch * Token, block_M), T.ceildiv(outdim, block_N)], sm_num, block_id):
                T.clear(output_reg)

                for k in T.Pipelined(T.ceildiv(Indim, block_K), num_stages=num_stages):
                    T.copy(Input[bx * block_M, k * block_K], Input_shared)
                    T.copy(W_T[by * block_N, k * block_K], W_T_shared)
                    T.gemm(Input_shared, W_T_shared, output_reg, transpose_B=True)
    
                silu(output_reg)
                L2Norm_QK(output_reg)
                T.copy(output_reg, output_shared)
                T.copy(output_shared, Output[bx * block_M, by * block_N])
        
    return main

def fused_linear_silu_l2norm(Input, W_T):
    B, Token, D_in = Input.shape
    D_out = W_T.shape[0]
    dtype = Input.dtype
    dtype = dtype_dict[dtype]
    # output = torch.empty(B, Token, D_out).cuda().half()
    kernel = _linear_kernel(B, Token, D_in, D_out, dtype)
    output = kernel(Input.view(-1, D_in), W_T)
    output = output.view(B, Token, D_out)
    return output

def pytorch_impl(Input, W_T):
    B, Token, D_in = Input.shape
    D_out = W_T.shape[0]
    
    # output = torch.empty(B, Token, D_out).cuda().half()
    Input = Input.view(-1, D_in)
    # output = output.view(-1, D_out)
    output = torch.matmul(Input, W_T.T)
    output = output / (1 + torch.exp(-output))
    output = output.view(B, Token, D_out)
    return output
if __name__ == "__main__":
    B, Token, D_in, D_out = 40, 1, 1536, 1152
    Input_fp16 = torch.randn(B, Token, D_in).cuda().bfloat16()
    W_T_fp16 = torch.randn(D_out, D_in).cuda().bfloat16()
    W_T_fp16_2 = torch.randn(D_out, D_in).cuda().bfloat16()  # 第二个权重矩阵，模拟k_proj
    layer_q = nn.Linear(D_in, D_out, bias=False).cuda().bfloat16()
    layer_k = nn.Linear(D_in, D_out, bias=False).cuda().bfloat16()
    # 分配一些额外的GPU内存来模拟模型环境
    dummy_tensors = [torch.randn(100, 1536, 1536).cuda().bfloat16() for _ in range(5)]
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # 测试：模拟模型中的调用模式（连续调用两次不同权重）
    print("\n=== Test 1: 模拟模型中的调用模式 (两次连续调用) ===")
    for _ in range(10):
        _ = fused_linear_silu_l2norm(Input_fp16, layer_q.weight)
        _ = fused_linear_silu_l2norm(Input_fp16, layer_k.weight)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(20):
        q = fused_linear_silu_l2norm(Input_fp16, layer_q.weight)
        k = fused_linear_silu_l2norm(Input_fp16, layer_k.weight)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time taken tilelang (2 calls): {(end - start) / 20} seconds")
    # print(f"Time per call: {(end - start) / 40} seconds")
    
    # # 测试单次调用
    # print("\n=== Test 2: 单次调用测试 ===")
    # for _ in range(10):
    #     fused_linear_silu_l2norm(Input_fp16, layer_q.weight)
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(20):
    #     fused_linear_silu_l2norm(Input_fp16, W_T_fp16)
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"Time taken tilelang (single): {(end - start) / 20} seconds")
    
    # PyTorch对比
    print("\n=== Test 3: PyTorch baseline ===")
    for _ in range(10):
        # pytorch_impl(Input_fp16, W_T_fp16)
        _ = F.silu(layer_q(Input_fp16))
        _ = F.silu(layer_k(Input_fp16))
        # _ = layer_q(Input_fp16)
        # _ = layer_k(Input_fp16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        _ = F.silu(layer_q(Input_fp16))
        _ = F.silu(layer_k(Input_fp16))
        # _ = layer_q(Input_fp16)
        # _ = layer_k(Input_fp16)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time taken pytorch: {(end - start) / 20} seconds")
    # pytorch matmul
    for _ in range(10):
        _ = F.silu(torch.matmul(Input_fp16, W_T_fp16.T))
        _ = F.silu(torch.matmul(Input_fp16, W_T_fp16_2.T))
        # _ = torch.matmul(Input_fp16, W_T_fp16.T)
        # _ = torch.matmul(Input_fp16, W_T_fp16_2.T)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        _ = F.silu(torch.matmul(Input_fp16, W_T_fp16.T))
        _ = F.silu(torch.matmul(Input_fp16, W_T_fp16_2.T))
        # _ = torch.matmul(Input_fp16, W_T_fp16.T)
        # _ = torch.matmul(Input_fp16, W_T_fp16_2.T)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time taken pytorch matmul: {(end - start) / 20} seconds")
    # Input_fp8 = torch.randn(B, Token, D_in).cuda().to(torch.float8_e4m3fn)
    # W_T_fp8 = torch.randn(D_out, D_in).cuda().to(torch.float8_e4m3fn)
    # for _ in range(10):
    #     fused_linear_silu(Input_fp8, W_T_fp8)
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(20):
    #     fused_linear_silu(Input_fp8, W_T_fp8)
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"Time taken: {end - start} seconds")


