import tilelang
import tilelang.language as T
import sys  # noqa: F401
import itertools
from tilelang.autotuner import autotune
import time
import torch
from typing import Optional
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.utils.solve_tril import solve_tril_16x16_kernel, merge_16x16_to_64x64_inverse_kernel
def cdiv(a, b):
    return (a + b - 1) // b

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    chunk_lens = cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    indices = torch.cat([torch.arange(n) for n in chunk_lens])
    chunk_indices = torch.cat([torch.full((n,), i) for i, n in enumerate(chunk_lens)])
    return torch.stack([chunk_indices, indices], 1).to(cu_seqlens)



@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
})
def solve_tril_16x16(        
    B: int,
    Token: int,
    H: int,
    BT: int,
    NT: int,
    input_dtype = "bfloat16",
    accum_dtype = "float32",
    output_dtype = "bfloat16",
    ):
    assert BT in [16, 32, 64]
    # B, Token, H, BT = A.shape
    # chunk_indices = prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    # NT = len(chunk_indices) if cu_seqlens is not None else cdiv(Token, 16)
    @T.prim_func
    def main(
        A: T.Tensor([B, Token, H, BT], input_dtype),
        Ad: T.Tensor([B, Token, H, 16], output_dtype),
    
    ):
        with T.Kernel(NT, B * H) as (i_t, i_bh):
            i_b, ih = i_bh // H, i_bh % H
            # i_n, i_tc = chunk_indices[i_t, 0], chunk_indices[i_t, 1]
            # bos, eos = cu_seqlens[i_n], cu_seqlens[i_n + 1]
            # seq = eos - bos
            offset = (i_t * 16) % BT
            # offset = i_t * 16
            

            A_shared = T.alloc_shared((16, 16), dtype=accum_dtype, scope="shared")
            # A_fragment = T.alloc_fragment((16, 16), dtype=accum_dtype)
            reduce_fragment1 = T.alloc_fragment((16, 16), dtype=accum_dtype)
            reduce_fragment2 = T.alloc_fragment((16), dtype=accum_dtype)
            a_shared = T.alloc_shared((16), dtype=accum_dtype, scope="shared")
            a_fragment = T.alloc_fragment((16), dtype=accum_dtype)

            T.copy(A[i_b, i_t * 16:(i_t + 1) * 16, ih, offset:offset + 16], A_shared)
            # T.copy(A_shared, A_fragment)
            for i, j in T.Parallel(16, 16):
                A_shared[i, j] = T.if_then_else(i >j, -A_shared[i, j], 0)
                
            for i in T.serial(1, T.min(16, Token - i_t * 16)):
                T.copy(A[i_b, i_t * 16 + i, ih, offset:offset + 16], a_shared)
                T.copy(a_shared, a_fragment)
                for j in T.Parallel(16):
                    a_fragment[j] = -a_fragment[j]
                # for j in T.Parallel(16):
                #     a_fragment[j] = A[i_b, i_t * 16 + i, ih, offset:offset + j]
                for ii, j in T.Parallel(16, 16):
                    reduce_fragment1[ii, j] = A_shared[ii, j] * a_fragment[ii]
                T.reduce_sum(reduce_fragment1, reduce_fragment2, dim = 0)
                for j in T.Parallel(16):
                    a_fragment[j] = a_fragment[j] + reduce_fragment2[j]
                for j in T.Parallel(16):
                    A_shared[i, j] = a_fragment[j]

            # T.copy(A_fragment, A_shared)
            for i in T.Parallel(16):
                A_shared[i, i] = A_shared[i, i] + 1.0
            
            T.copy(A_shared, Ad[i_b, i_t * 16:(i_t + 1) * 16, ih, :])
    return main

@tilelang.jit(
pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    })
def tl_merge_16x16_to_64x64_inverse_kernel(
    B: int,
    Token: int,
    H: int,
    BT: int,
    NT: int,
    input_dtype = "bfloat16",
    accum_dtype = "float32",
    output_dtype = "bfloat16",
):

    @T.macro
    def gemm_inverse(
        mat_a: T.SharedBuffer([16, 16], input_dtype),
        mat_b: T.SharedBuffer([16, 16], input_dtype),
        mat_c: T.SharedBuffer([16, 16], input_dtype),
        tmp_reg1: T.FragmentBuffer([16, 16], accum_dtype),
        tmp_reg2: T.FragmentBuffer([16, 16], accum_dtype),
        tmp_shared: T.SharedBuffer([16, 16], input_dtype),
        out_shared: T.SharedBuffer([16, 16], input_dtype),
    ):
        # tmp_reg1 = mat_a @ mat_b
        T.clear(tmp_reg1)
        T.gemm(mat_a, mat_b, tmp_reg1, transpose_B=False)
        # Write back to shared memory to break layout dependency (manual copy)
        for i, j in T.Parallel(16, 16):
            tmp_shared[i, j] = tmp_reg1[i, j]
        # tmp_reg2 = tmp_shared @ mat_c = (mat_a @ mat_b) @ mat_c
        T.clear(tmp_reg2)
        T.gemm(tmp_shared, mat_c, tmp_reg2, transpose_B=False)
        # out_shared = -(mat_a @ mat_b @ mat_c)
        for i, j in T.Parallel(16, 16):
            tmp_reg2[i, j] = -tmp_reg2[i, j] 
        T.copy(tmp_reg2, out_shared)
    @T.macro
    def multigemm1(
        mat_a: T.SharedBuffer([16, 16], input_dtype),
        mat_b: T.SharedBuffer([16, 16], input_dtype),
        mat_c: T.SharedBuffer([16, 16], input_dtype),
        mat_d: T.SharedBuffer([16, 16], input_dtype),
        mat_e: T.SharedBuffer([16, 16], input_dtype),
        tmp_reg1: T.FragmentBuffer([16, 16], accum_dtype),
        tmp_reg2: T.FragmentBuffer([16, 16], accum_dtype),
        tmp_shared: T.SharedBuffer([16, 16], input_dtype),
        out_shared: T.SharedBuffer([16, 16], input_dtype),
    ):
        T.clear(tmp_reg1)
        T.gemm(mat_a, mat_b, tmp_reg1, transpose_B=False)
        T.gemm(mat_c, mat_d, tmp_reg1, transpose_B=False)
        # Write back to shared memory to break layout dependency (manual copy)
        for i, j in T.Parallel(16, 16):
            tmp_shared[i, j] = tmp_reg1[i, j]
        T.clear(tmp_reg2)
        T.gemm(mat_e, tmp_shared, tmp_reg2, transpose_B=False)
        for i, j in T.Parallel(16, 16):
            tmp_reg2[i, j] = -tmp_reg2[i, j] 
        T.copy(tmp_reg2, out_shared)
    
    @T.macro
    def multigemm2(
        mat_a: T.SharedBuffer([16, 16], input_dtype),
        mat_b: T.SharedBuffer([16, 16], input_dtype),
        mat_c: T.SharedBuffer([16, 16], input_dtype),
        mat_d: T.SharedBuffer([16, 16], input_dtype),
        mat_e: T.SharedBuffer([16, 16], input_dtype),
        mat_f: T.SharedBuffer([16, 16], input_dtype),
        mat_g: T.SharedBuffer([16, 16], input_dtype),
        tmp_reg1: T.FragmentBuffer([16, 16], accum_dtype),
        tmp_reg2: T.FragmentBuffer([16, 16], accum_dtype),
        tmp_shared: T.SharedBuffer([16, 16], input_dtype),
        out_shared: T.SharedBuffer([16, 16], input_dtype),
    ):
        T.clear(tmp_reg1)
        T.gemm(mat_a, mat_b, tmp_reg1, transpose_B=False)
        T.gemm(mat_c, mat_d, tmp_reg1, transpose_B=False)
        T.gemm(mat_e, mat_f, tmp_reg1, transpose_B=False)
        # Write back to shared memory to break layout dependency (manual copy)
        for i, j in T.Parallel(16, 16):
            tmp_shared[i, j] = tmp_reg1[i, j]
        T.clear(tmp_reg2)
        T.gemm(mat_g, tmp_shared, tmp_reg2, transpose_B=False)
        for i, j in T.Parallel(16, 16):
            tmp_reg2[i, j] = -tmp_reg2[i, j] 
        T.copy(tmp_reg2, out_shared)
    @T.prim_func
    def main(
        A: T.Tensor([B, Token, H, BT], input_dtype),
        Ad: T.Tensor([B, Token, H, 16], input_dtype),
        Ai: T.Tensor([B, Token, H, 64], output_dtype),
    ):
        with T.Kernel(NT, B * H,threads=32) as (i_t, i_bh):
            i_b, ih = i_bh // H, i_bh % H
            
            # Allocate shared memory for all blocks
            A_21_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            A_32_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            A_31_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            A_43_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            A_42_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            A_41_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")

            Ad_11_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_22_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_33_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_44_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_21_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_32_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_43_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_31_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_42_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            Ad_41_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            
            # Temporary shared buffers for intermediate results
            gemm_inv_tmp_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            multigemm1_tmp_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            multigemm2_tmp_shared = T.alloc_shared((16, 16), dtype=input_dtype, scope="shared")
            
            # Fragment buffers for gemm_inverse macro
            gemm_inv_reg1 = T.alloc_fragment((16, 16), dtype=accum_dtype)
            gemm_inv_reg2 = T.alloc_fragment((16, 16), dtype=accum_dtype)
            
            # Fragment buffers for multigemm1 macro
            multigemm1_reg1 = T.alloc_fragment((16, 16), dtype=accum_dtype)
            multigemm1_reg2 = T.alloc_fragment((16, 16), dtype=accum_dtype)
            
            # Fragment buffers for multigemm2 macro
            multigemm2_reg1 = T.alloc_fragment((16, 16), dtype=accum_dtype)
            multigemm2_reg2 = T.alloc_fragment((16, 16), dtype=accum_dtype)
            # Load blocks from A tensor
            # p_A_21: (i_t * 64 + 16, 0), shape (16, 16) - rows [i_t*64+16:i_t*64+32], cols [0:16]
            T.copy(A[i_b, i_t * 64 + 16:i_t * 64 + 32, ih, 0:16], A_21_shared)
            
            # p_A_32: (i_t * 64 + 32, 16), shape (16, 16) - rows [i_t*64+32:i_t*64+48], cols [16:32]
            T.copy(A[i_b, i_t * 64 + 32:i_t * 64 + 48, ih, 16:32], A_32_shared)
            
            # p_A_31: (i_t * 64 + 32, 0), shape (16, 16) - rows [i_t*64+32:i_t*64+48], cols [0:16]
            T.copy(A[i_b, i_t * 64 + 32:i_t * 64 + 48, ih, 0:16], A_31_shared)
            
            # p_A_43: (i_t * 64 + 48, 32), shape (16, 16) - rows [i_t*64+48:i_t*64+64], cols [32:48]
            T.copy(A[i_b, i_t * 64 + 48:i_t * 64 + 64, ih, 32:48], A_43_shared)
            
            # p_A_42: (i_t * 64 + 48, 16), shape (16, 16) - rows [i_t*64+48:i_t*64+64], cols [16:32]
            T.copy(A[i_b, i_t * 64 + 48:i_t * 64 + 64, ih, 16:32], A_42_shared)
            
            # p_A_41: (i_t * 64 + 48, 0), shape (16, 16) - rows [i_t*64+48:i_t*64+64], cols [0:16]
            T.copy(A[i_b, i_t * 64 + 48:i_t * 64 + 64, ih, 0:16], A_41_shared)
            
            # Load blocks from Ad tensor (diagonal blocks)
            # p_Ad_11: (i_t * 64, 0), shape (16, 16) - rows [i_t*64:i_t*64+16], cols [0:16]
            T.copy(Ad[i_b, i_t * 64:i_t * 64 + 16, ih, 0:16], Ad_11_shared)
            
            # p_Ad_22: (i_t * 64 + 16, 0), shape (16, 16) - rows [i_t*64+16:i_t*64+32], cols [0:16]
            T.copy(Ad[i_b, i_t * 64 + 16:i_t * 64 + 32, ih, 0:16], Ad_22_shared)
            
            # p_Ad_33: (i_t * 64 + 32, 0), shape (16, 16) - rows [i_t*64+32:i_t*64+48], cols [0:16]
            T.copy(Ad[i_b, i_t * 64 + 32:i_t * 64 + 48, ih, 0:16], Ad_33_shared)
            
            # p_Ad_44: (i_t * 64 + 48, 0), shape (16, 16) - rows [i_t*64+48:i_t*64+64], cols [0:16]
            T.copy(Ad[i_b, i_t * 64 + 48:i_t * 64 + 64, ih, 0:16], Ad_44_shared)

            # Ai_21 = -(Ai_22 @ A_21 @ Ai_11)
            gemm_inverse(Ad_22_shared, A_21_shared, Ad_11_shared, gemm_inv_reg1, gemm_inv_reg2, gemm_inv_tmp_shared, Ad_21_shared)
            # Ai_32 = -(Ai_33 @ A_32 @ Ai_22)
            gemm_inverse(Ad_33_shared, A_32_shared, Ad_22_shared, gemm_inv_reg1, gemm_inv_reg2, gemm_inv_tmp_shared, Ad_32_shared)
            # Ai_43 = -(Ai_44 @ A_43 @ Ai_33)
            gemm_inverse(Ad_44_shared, A_43_shared, Ad_33_shared, gemm_inv_reg1, gemm_inv_reg2, gemm_inv_tmp_shared, Ad_43_shared)

            multigemm1(A_31_shared, Ad_11_shared, A_32_shared, Ad_21_shared, Ad_33_shared, multigemm1_reg1, multigemm1_reg2, multigemm1_tmp_shared, Ad_31_shared)

            multigemm1(A_42_shared, Ad_22_shared, A_43_shared, Ad_32_shared, Ad_44_shared, multigemm1_reg1, multigemm1_reg2, multigemm1_tmp_shared, Ad_42_shared)

            multigemm2(A_41_shared, Ad_11_shared, A_42_shared, Ad_21_shared, A_43_shared, Ad_31_shared, Ad_44_shared, multigemm2_reg1, multigemm2_reg2, multigemm2_tmp_shared, Ad_41_shared)

            # Store results back to Ai tensor
            # p_Ai_11: (i_t * 64, 0), shape (16, 16) - rows [i_t*64:i_t*64+16], cols [0:16]
            T.copy(Ad_11_shared, Ai[i_b, i_t * 64:i_t * 64 + 16, ih, 0:16])
            
            # p_Ai_22: (i_t * 64 + 16, 16), shape (16, 16) - rows [i_t*64+16:i_t*64+32], cols [16:32]
            T.copy(Ad_22_shared, Ai[i_b, i_t * 64 + 16:i_t * 64 + 32, ih, 16:32])
            
            # p_Ai_33: (i_t * 64 + 32, 32), shape (16, 16) - rows [i_t*64+32:i_t*64+48], cols [32:48]
            T.copy(Ad_33_shared, Ai[i_b, i_t * 64 + 32:i_t * 64 + 48, ih, 32:48])
            
            # p_Ai_44: (i_t * 64 + 48, 48), shape (16, 16) - rows [i_t*64+48:i_t*64+64], cols [48:64]
            T.copy(Ad_44_shared, Ai[i_b, i_t * 64 + 48:i_t * 64 + 64, ih, 48:64])
            
            # p_Ai_21: (i_t * 64 + 16, 0), shape (16, 16) - rows [i_t*64+16:i_t*64+32], cols [0:16]
            T.copy(Ad_21_shared, Ai[i_b, i_t * 64 + 16:i_t * 64 + 32, ih, 0:16])
            
            # p_Ai_31: (i_t * 64 + 32, 0), shape (16, 16) - rows [i_t*64+32:i_t*64+48], cols [0:16]
            T.copy(Ad_31_shared, Ai[i_b, i_t * 64 + 32:i_t * 64 + 48, ih, 0:16])
            
            # p_Ai_32: (i_t * 64 + 32, 16), shape (16, 16) - rows [i_t*64+32:i_t*64+48], cols [16:32]
            T.copy(Ad_32_shared, Ai[i_b, i_t * 64 + 32:i_t * 64 + 48, ih, 16:32])
            
            # p_Ai_41: (i_t * 64 + 48, 0), shape (16, 16) - rows [i_t*64+48:i_t*64+64], cols [0:16]
            T.copy(Ad_41_shared, Ai[i_b, i_t * 64 + 48:i_t * 64 + 64, ih, 0:16])
            
            # p_Ai_42: (i_t * 64 + 48, 16), shape (16, 16) - rows [i_t*64+48:i_t*64+64], cols [16:32]
            T.copy(Ad_42_shared, Ai[i_b, i_t * 64 + 48:i_t * 64 + 64, ih, 16:32])
            
            # p_Ai_43: (i_t * 64 + 48, 32), shape (16, 16) - rows [i_t*64+48:i_t*64+64], cols [32:48]
            T.copy(Ad_43_shared, Ai[i_b, i_t * 64 + 48:i_t * 64 + 64, ih, 32:48])
    
    return main

def tl_solve_tril(
    A: torch.Tensor,
    output_dtype: torch.dtype = torch.float,
    NT1: int = 16,
    NT2: int = 64,
)-> torch.Tensor:
    assert A.shape[-1] in [16, 32, 64]
    B, T, H, BT = A.shape
    Ad = torch.empty(B, T, H, NT1, device=A.device, dtype= torch.bfloat16)
    kernel_solve_tril = solve_tril_16x16(B, T, H, BT, NT1)
    kernel_solve_tril(A, Ad)
    Ai = torch.zeros(B, T, H, NT2, device=A.device, dtype= torch.bfloat16)
    kernel_merge_16x16_to_64x64_inverse = tl_merge_16x16_to_64x64_inverse_kernel(B, T, H, BT, NT2)

    kernel_merge_16x16_to_64x64_inverse(A, Ad, Ai)
    return Ai
    

if __name__ == "__main__":
    # Set print options to display full tensors
    # torch.set_printoptions(
    #     threshold=float('inf'),  # 打印所有元素
    #     linewidth=200,           # 每行字符数
    #     precision=4,             # 小数点后位数
    #     sci_mode=False,          # 不使用科学计数法
    # )
    
    B, Token, Head, Dim = 1, 100, 1, 64
    A = torch.randn(B, Token, Head, Dim).to(torch.float32).cuda()
    Ad_triton = torch.randn(B, Token, Head, 16).to(torch.bfloat16).cuda()
    Ai_triton = torch.empty(B, Token, Head, Dim).to(torch.bfloat16).cuda()
    Ai_tilelang = torch.empty(B, Token, Head, Dim).to(torch.bfloat16).cuda()
    cu_seqlens = torch.tensor([0, 64], dtype=torch.int32).cuda()
    
    # # For 16x16 solve_tril kernel
    # chunk_indices_16 = prepare_chunk_indices(cu_seqlens, 16)
    # NT_16 = len(chunk_indices_16) if cu_seqlens is not None else cdiv(Token, 16)
    
    # # For 64x64 merge kernel
    # chunk_indices_64 = prepare_chunk_indices(cu_seqlens, 64)
    # NT_64 = len(chunk_indices_64) if cu_seqlens is not None else cdiv(Token, 64)

    # # 运行 Tilelang kernel
    # kernel = solve_tril_16x16(B, Token, Head, Dim, NT)
    # Ad_tilelang = kernel(A)
    
    # # 运行 Triton kernel (varlen=False，所以传None)
    # solve_tril_16x16_kernel[NT, B * Head](A, Ad_triton, None, None, Token, Head, Dim)
    
    # # 转换为float32进行精确计算
    # Ad_tilelang_fp32 = Ad_tilelang.float()
    # Ad_triton_fp32 = Ad_triton.float()
    
    # # 计算绝对误差
    # abs_error = torch.abs(Ad_tilelang_fp32 - Ad_triton_fp32)
    
    # # 计算相对误差（避免除零）
    # epsilon = 1e-8
    # relative_error = abs_error / (torch.abs(Ad_triton_fp32) + epsilon)
    
    # # 统计信息
    # mean_relative_error = relative_error.mean().item()
    # max_relative_error = relative_error.max().item()
    # mean_abs_error = abs_error.mean().item()
    # max_abs_error = abs_error.max().item()
    
    # print("=" * 80)
    # print("误差分析 (Tilelang vs Triton):")
    # print("=" * 80)
    # print(f"平均相对误差: {mean_relative_error:.6e}")
    # print(f"最大相对误差: {max_relative_error:.6e}")
    # print(f"平均绝对误差: {mean_abs_error:.6e}")
    # print(f"最大绝对误差: {max_abs_error:.6e}")
    # print("=" * 80)
    
    # # 打印输出形状信息
    # print(f"\nTilelang输出形状: {Ad_tilelang.shape}")
    # print(f"Triton输出形状: {Ad_triton.shape}")
    
    # # 可选：打印一小部分数据对比
    # print("\n前3个元素对比 (Tilelang vs Triton):")
    # print("Tilelang:", Ad_tilelang.flatten()[:3])
    # print("Triton  :", Ad_triton.flatten()[:3])
    # Run TileLang kernel
    # kernel = tl_merge_16x16_to_64x64_inverse_kernel(
    #     B, Token, Head, Dim, NT_64,
    # )
    # kernel(A, Ad_triton, Ai_tilelang)
    
    # # Run Triton kernel
    # merge_16x16_to_64x64_inverse_kernel[NT_64, B * Head](A, Ad_triton, Ai_triton, None, None, Token, Head, Dim)
 

    # # Convert to float32 for accurate error calculation
    # Ai_tilelang_fp32 = Ai_tilelang.float()
    # Ai_triton_fp32 = Ai_triton.float()
    # print(Ai_tilelang_fp32)
    # print(Ai_triton_fp32)
    # # Calculate errors
    # abs_error = torch.abs(Ai_tilelang_fp32 - Ai_triton_fp32)
    
    # # Calculate relative error (avoid division by zero)
    # epsilon = 1e-8
    # relative_error = abs_error / (torch.abs(Ai_triton_fp32) + epsilon)
    
    # # Statistics
    # mean_relative_error = relative_error.mean().item()
    # max_relative_error = relative_error.max().item()
    # mean_abs_error = abs_error.mean().item()
    # max_abs_error = abs_error.max().item()
    
    # print("=" * 80)
    # print("误差分析 (Tilelang vs Triton):")
    # print("=" * 80)
    # print(f"平均相对误差: {mean_relative_error:.6e}")
    # print(f"最大相对误差: {max_relative_error:.6e}")
    # print(f"平均绝对误差: {mean_abs_error:.6e}")
    # print(f"最大绝对误差: {max_abs_error:.6e}")
    # print("=" * 80)
    
    # # Print shape info
    # print(f"\nTilelang输出形状: {Ai_tilelang.shape}")
    # print(f"Triton输出形状: {Ai_triton.shape}")
    
    # # Optional: print sample data
    # print("\n前5个元素对比 (Tilelang vs Triton):")
    # print("Tilelang:", Ai_tilelang.flatten()[:5])
    # print("Triton  :", Ai_triton.flatten()[:5])

    Ai_tilelang = tl_solve_tril(A)
    print(Ai_tilelang)
    Ai_triton = solve_tril(A)
    print(Ai_triton)

    
