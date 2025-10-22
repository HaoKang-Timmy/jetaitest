import tilelang
import tilelang.language as T
import sys  # noqa: F401
import itertools
from tilelang.autotuner import autotune
import time
import torch
from typing import Optional
from fla.ops.utils import chunk_local_cumsum, solve_tril
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



@tilelang.jit(out_idx=[-1])
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
            

            A_shared = T.alloc_shared((16, 16), dtype=output_dtype, scope="shared")
            A_fragment = T.alloc_fragment((16, 16), dtype=accum_dtype)
            reduce_fragment1 = T.alloc_fragment((16, 16), dtype=accum_dtype)
            reduce_fragment2 = T.alloc_fragment((16), dtype=accum_dtype)
            a_shared = T.alloc_shared((16), dtype=output_dtype)
            a_fragment = T.alloc_fragment((16), dtype=accum_dtype)

            T.copy(A[i_b, i_t * 16:(i_t + 1) * 16, ih, offset:offset + 16], A_shared)
            T.copy(A_shared, A_fragment)
            for i, j in T.Parallel(16, 16):
                A_fragment[i, j] = T.if_then_else(i >j, -A_fragment[i, j], 0)
                
            for i in T.serial(1, T.min(16, Token - i_t * 16)):
                T.copy(A[i_b, i_t * 16 + i, ih, offset:offset + 16], a_shared)
                T.copy(a_shared, a_fragment)
                for j in T.Parallel(16):
                    a_fragment[j] = -a_fragment[j]
                for ii, j in T.Parallel(16, 16):
                    reduce_fragment1[ii, j] = A_fragment[ii, j] * a_fragment[ii]
                T.reduce_sum(reduce_fragment1, reduce_fragment2, dim = 0)
                for j in T.Parallel(16):
                    a_fragment[j] = a_fragment[j] + reduce_fragment2[j]
                for j in T.Parallel(16):
                    A_fragment[i, j] = a_fragment[j]

            T.copy(A_fragment, A_shared)
            for i in T.Parallel(16):
                A_shared[i, i] = A_shared[i, i] + 1.0
            
            T.copy(A_shared, Ad[i_b, i_t * 16:(i_t + 1) * 16, ih, :])
    return main

@tilelang.jit(out_idx=[-1])
def tl_merge_16x16_to_64x64_inverse_kernel(
    B: int,
    Token: int,
    H: int,
    BT: int,
    NT: int,
    input_dtype = "bfloat16",
    accum_dtype = "bfloat16",
):
    @T.prim_func
    def main(
        A: T.Tensor(shape=(B, Token, H, BT), dtype=input_dtype),
        Ad: T.Tensor(shape=(B, Token, H, 16), dtype=input_dtype),
        Ai: T.Tensor(shape=(B, Token, H, 64), dtype=input_dtype),
    ):
        with T.Kernel(NT, B * H) as (i_t, i_bh):
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
    
    return main


if __name__ == "__main__":
    B, Token, Head, Dim = 1, 100, 12, 64
    A = torch.randn(B, Token, Head, Dim).to(torch.bfloat16).cuda()
    Ad = torch.randn(B, Token, Head, 16).to(torch.bfloat16).cuda()
    cu_seqlens = torch.tensor([0, 100], dtype=torch.int32).cuda()
    chunk_indices = prepare_chunk_indices(cu_seqlens, 16)
    NT = len(chunk_indices) if cu_seqlens is not None else cdiv(Token, 16)

    kernel = solve_tril_16x16(B, Token, Head, Dim, NT)
    Ad = kernel(A)
    # print(Ad)