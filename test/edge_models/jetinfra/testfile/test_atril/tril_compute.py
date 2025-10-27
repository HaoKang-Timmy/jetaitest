import torch

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import warnings
from typing import Optional
import itertools
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
    input_dtype = "float32",
    accum_dtype = "float32",
    output_dtype = "float32",
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
            A_21_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            A_32_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            A_31_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            A_43_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            A_42_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            A_41_shared = T.alloc_shared((16, 16), dtype=input_dtype)

            Ad_11_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_22_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_33_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_44_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_21_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_32_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_43_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_31_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_42_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            Ad_41_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            
            # Temporary shared buffers for intermediate results
            gemm_inv_tmp_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            multigemm1_tmp_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            multigemm2_tmp_shared = T.alloc_shared((16, 16), dtype=input_dtype)
            
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

def tl_merge_16x16_to_64x64_inverse_pytorch(A, Ad):
    """
    PyTorch implementation of the block matrix inverse kernel
    
    Args:
        A: [B, Token, H, BT] - off-diagonal blocks
        Ad: [B, Token, H, 16] - diagonal blocks
    
    Returns:
        Ai: [B, Token, H, 64] - inverse matrix blocks
    """
    B, Token, H, BT = A.shape
    NT = Token // 64  # number of 64-token chunks
    
    # Initialize output
    Ai = torch.zeros(B, Token, H, 64, dtype=A.dtype, device=A.device)
    
    for i_t in range(NT):
        # Extract 16x16 blocks from A tensor
        A_21 = A[:, i_t*64+16:i_t*64+32, :, 0:16]    # (B, 16, H, 16)
        A_32 = A[:, i_t*64+32:i_t*64+48, :, 16:32]
        A_31 = A[:, i_t*64+32:i_t*64+48, :, 0:16]
        A_43 = A[:, i_t*64+48:i_t*64+64, :, 32:48]
        A_42 = A[:, i_t*64+48:i_t*64+64, :, 16:32]
        A_41 = A[:, i_t*64+48:i_t*64+64, :, 0:16]
        
        # Extract diagonal blocks from Ad tensor
        Ad_11 = Ad[:, i_t*64+0:i_t*64+16, :, 0:16]
        Ad_22 = Ad[:, i_t*64+16:i_t*64+32, :, 0:16]
        Ad_33 = Ad[:, i_t*64+32:i_t*64+48, :, 0:16]
        Ad_44 = Ad[:, i_t*64+48:i_t*64+64, :, 0:16]
        
        # Reshape for batch matrix multiplication: (B, H, 16, 16)
        # PyTorch bmm expects (batch, m, k) @ (batch, k, n)
        def reshape_for_bmm(x):
            return x.transpose(1, 2)  # (B, H, 16, 16)
        
        # Compute Ai_21 = -(Ad_22 @ A_21 @ Ad_11)
        Ai_21 = -torch.matmul(torch.matmul(
            reshape_for_bmm(Ad_22), 
            reshape_for_bmm(A_21)
        ), reshape_for_bmm(Ad_11))
        
        # Compute Ai_32 = -(Ad_33 @ A_32 @ Ad_22)
        Ai_32 = -torch.matmul(torch.matmul(
            reshape_for_bmm(Ad_33),
            reshape_for_bmm(A_32)
        ), reshape_for_bmm(Ad_22))
        
        # Compute Ai_43 = -(Ad_44 @ A_43 @ Ad_33)
        Ai_43 = -torch.matmul(torch.matmul(
            reshape_for_bmm(Ad_44),
            reshape_for_bmm(A_43)
        ), reshape_for_bmm(Ad_33))
        
        # Compute Ai_31 = -Ad_33 @ (A_31 @ Ad_11 + A_32 @ Ai_21)
        temp1 = torch.matmul(reshape_for_bmm(A_31), reshape_for_bmm(Ad_11))
        temp2 = torch.matmul(reshape_for_bmm(A_32), Ai_21)
        Ai_31 = -torch.matmul(reshape_for_bmm(Ad_33), temp1 + temp2)
        
        # Compute Ai_42 = -Ad_44 @ (A_42 @ Ad_22 + A_43 @ Ai_32)
        temp1 = torch.matmul(reshape_for_bmm(A_42), reshape_for_bmm(Ad_22))
        temp2 = torch.matmul(reshape_for_bmm(A_43), Ai_32)
        Ai_42 = -torch.matmul(reshape_for_bmm(Ad_44), temp1 + temp2)
        
        # Compute Ai_41 = -Ad_44 @ (A_41 @ Ad_11 + A_42 @ Ai_21 + A_43 @ Ai_31)
        temp1 = torch.matmul(reshape_for_bmm(A_41), reshape_for_bmm(Ad_11))
        temp2 = torch.matmul(reshape_for_bmm(A_42), Ai_21)
        temp3 = torch.matmul(reshape_for_bmm(A_43), Ai_31)
        Ai_41 = -torch.matmul(reshape_for_bmm(Ad_44), temp1 + temp2 + temp3)
        
        # Reshape back and store results
        def reshape_back(x):
            return x.transpose(1, 2)  # (B, 16, H, 16)
        
        # Store diagonal blocks
        Ai[:, i_t*64+0:i_t*64+16, :, 0:16] = reshape_back(reshape_for_bmm(Ad_11))
        Ai[:, i_t*64+16:i_t*64+32, :, 16:32] = reshape_back(reshape_for_bmm(Ad_22))
        Ai[:, i_t*64+32:i_t*64+48, :, 32:48] = reshape_back(reshape_for_bmm(Ad_33))
        Ai[:, i_t*64+48:i_t*64+64, :, 48:64] = reshape_back(reshape_for_bmm(Ad_44))
        
        # Store off-diagonal blocks
        Ai[:, i_t*64+16:i_t*64+32, :, 0:16] = reshape_back(Ai_21)
        Ai[:, i_t*64+32:i_t*64+48, :, 0:16] = reshape_back(Ai_31)
        Ai[:, i_t*64+32:i_t*64+48, :, 16:32] = reshape_back(Ai_32)
        Ai[:, i_t*64+48:i_t*64+64, :, 0:16] = reshape_back(Ai_41)
        Ai[:, i_t*64+48:i_t*64+64, :, 16:32] = reshape_back(Ai_42)
        Ai[:, i_t*64+48:i_t*64+64, :, 32:48] = reshape_back(Ai_43)
    
    return Ai