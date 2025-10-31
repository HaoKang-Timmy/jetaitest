import argparse
import itertools
import tilelang as tl
import tilelang
import tilelang.language as T
from tvm import DataType
from tilelang.autotuner import autotune
from tilelang import jit
import torch
import time
import sys
sys.path.insert(0, '/home/haokang/jetlmrelated/jetaitest/flash-linear-attention')
from fla.modules.fused_norm_gate import FusedRMSNormGated


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def RMSNorm(
    B,
    S,
    Head,
    Dim,
    dtype = "bfloat16",
    accum_dtype = "float32",
    block_D = 256,
    threads = 128,
    scale = None,
):
    if scale is None:
        scale = 1.0 / Dim  # 修复：应该是 1/Dim，用于计算均值


    @T.prim_func
    def main(
        X: T.Tensor([B, S, Head, Dim], dtype),
        G: T.Tensor([B, S, Head, Dim], dtype),
        Weight: T.Tensor([Dim], dtype),
        # Y: T.Tensor([B, S, Head, Dim], dtype),
        
    ):
        with T.Kernel(B * Head, S, T.ceildiv(Dim, block_D), threads=threads) as (id_bh, id_s, id_d):
            id_b = id_bh // Head
            id_h = id_bh % Head
            X_shared = T.alloc_shared([block_D], dtype)
            G_shared = T.alloc_shared([block_D], dtype)
            W_shared = T.alloc_shared([block_D], dtype)
            Y_shared = T.alloc_shared([block_D], dtype)

            X_fragment = T.alloc_fragment([block_D], accum_dtype)
            X_rms_fragment = T.alloc_fragment([block_D], accum_dtype)
            X_reduce_square_fragment = T.alloc_fragment([1], accum_dtype)
            G_fragment = T.alloc_fragment([block_D], accum_dtype)
            G_sigmoid_fragment = T.alloc_fragment([block_D], accum_dtype)
            W_fragment = T.alloc_fragment([block_D], accum_dtype)
            Y_fragment = T.alloc_fragment([block_D], accum_dtype)

            T.copy(X[id_b, id_s, id_h, id_d * block_D], X_shared)
            T.copy(G[id_b, id_s, id_h, id_d * block_D], G_shared)
            T.copy(Weight[id_d * block_D], W_shared)

            T.copy(X_shared, X_fragment)
            T.copy(G_shared, G_fragment)
            T.copy(W_shared, W_fragment)

            #RMS(x) = sqrt( (1/D) * sum(x_i^2) )
            for i in T.Parallel(block_D):
                X_rms_fragment[i] = X_fragment[i] * X_fragment[i]
            T.reduce_sum(X_rms_fragment, X_reduce_square_fragment, dim = -1)
            X_reduce_square_fragment[0] = X_reduce_square_fragment[0] * scale
            
            for i in T.Parallel(block_D):
                X_rms_fragment[i] = X_fragment[i] * W_fragment[i] / (T.sqrt(X_reduce_square_fragment[0] + 1e-6))

            # gated activation, swish supported only
            for i in T.Parallel(block_D):
                G_sigmoid_fragment[i] = T.sigmoid(G_fragment[i])
            for i in T.Parallel(block_D):
                X_fragment[i] = X_rms_fragment[i] * G_fragment[i] * G_sigmoid_fragment[i]
            T.copy(X_fragment, X_shared)
            # inplace store to X
            T.copy(X_shared, X[id_b, id_s, id_h, id_d * block_D])
    return main
def tl_fused_rmsnorm(
    X: torch.Tensor,
    G: torch.Tensor,
    Weight: torch.Tensor,
):
    B, S, H, Dim = X.shape
    scale = 1.0 / Dim
    kernel = RMSNorm(B, S, H, Dim, dtype="bfloat16", accum_dtype="float32", 
                     block_D=256, threads=128, scale=scale)
    kernel(X, G, Weight)
    return X


if __name__ == "__main__":
    # 参数设置（匹配用户示例：[1, 1, 12, 256]）
    B, S, H, Dim = 1, 1, 12, 256
    datatype = torch.bfloat16
    acc_dtype = torch.float32
    eps = 1e-6
    autotune_interval = 524288
    
    # 准备输入数据
    X = torch.randn(B, S, H, Dim, device='cuda', dtype=datatype)
    G = torch.randn(B, S, H, Dim, device='cuda', dtype=datatype)
    Weight = torch.ones(Dim, device='cuda', dtype=datatype)  # 初始化为1
    
    # TileLang kernel 输出
    Y_tl = torch.empty(B, S, H, Dim, device='cuda', dtype=datatype)
    # scale = 1.0 / Dim  # 修复：应该是 1/Dim，不是 1/sqrt(Dim)
    # kernel = RMSNorm(B, S, H, Dim, dtype="bfloat16", accum_dtype="float32", 
    #                  block_D=256, threads=128, scale=scale)
    # kernel(X, G, Weight, Y_tl)
    Y_tl = tl_fused_rmsnorm(X, G, Weight)
    torch.cuda.synchronize()
    
    # FLA FusedRMSNormGated
    fla_norm = FusedRMSNormGated(
        hidden_size=Dim,
        elementwise_affine=True,
        eps=eps,
        activation='swish',
        device='cuda',
        dtype=datatype,
        autotune_interval=autotune_interval
    )
    # 设置相同的权重
    fla_norm.weight.data = Weight.clone()
    
    # 重塑输入以匹配 FLA 的期望：[B*S*H, Dim]
    X_fla = X.reshape(B * S * H, Dim)
    G_fla = G.reshape(B * S * H, Dim)
    
    # FLA kernel 输出
    Y_fla = fla_norm(X_fla, G_fla, residual=None, prenorm=False, residual_in_fp32=False)
    torch.cuda.synchronize()
    
    # 重塑回原始形状
    Y_fla = Y_fla.reshape(B, S, H, Dim)
    
    # 计算误差
    eps_compare = 1e-8
    Y_diff = torch.abs(Y_tl.float() - Y_fla.float())
    Y_relative_error = Y_diff / (torch.abs(Y_fla.float()) + eps_compare)
    
    mean_relative_error = Y_relative_error.mean().item()
    max_relative_error = Y_relative_error.max().item()
    mean_absolute_error = Y_diff.mean().item()
    max_absolute_error = Y_diff.max().item()
    
    # 打印结果
    print("\n" + "="*60)
    print("TileLang vs FLA FusedRMSNormGated 误差对比")
    print("="*60)
    print(f"\n输入形状: X={X.shape}, G={G.shape}, Weight={Weight.shape}")
    print(f"输出形状: Y={Y_tl.shape}")
    print(f"\n误差统计:")
    print(f"  平均相对误差: {mean_relative_error:.6e}")
    print(f"  最大相对误差: {max_relative_error:.6e}")
    print(f"  平均绝对误差: {mean_absolute_error:.6e}")
    print(f"  最大绝对误差: {max_absolute_error:.6e}")
    print("\n" + "="*60)
    
    # 检查是否匹配（使用合理的阈值）
    if max_relative_error < 1e-2:  # bfloat16 精度下 1% 的相对误差是可接受的
        print("✅ 测试通过：TileLang kernel 与 FLA kernel 结果匹配！")
    else:
        print("⚠️  警告：误差较大，可能存在实现差异")
    print("="*60 + "\n")

            
            
