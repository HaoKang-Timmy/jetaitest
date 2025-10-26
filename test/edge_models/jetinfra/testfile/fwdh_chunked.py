import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import warnings
from typing import Optional
import itertools
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h

import torch
import sys
import os

# 导入原版的kernel用于对比
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fwdh import tilelang_chunk_gated_delta_rule as tilelang_chunk_gated_delta_rule_original

def check_tensors_and_compute_errors(t1, names=None):
    has_nan_1 = torch.isnan(t1).any().item()
    has_inf_1 = torch.isinf(t1).any().item()
    print(f"{names} has_nan_1: {has_nan_1}")


def compute_relative_errors(tensor1, tensor2, tensor_name="Tensor"):
    """
    计算两个tensor之间的相对误差
    """
    # 转换为float32进行精确计算
    t1 = tensor1.float()
    t2 = tensor2.float()
    
    # 绝对误差
    abs_error = torch.abs(t1 - t2)
    max_abs_error = torch.max(abs_error).item()
    mean_abs_error = torch.mean(abs_error).item()
    
    # 相对误差 (避免除零)
    epsilon = 1e-8
    rel_error = abs_error / (torch.abs(t2) + epsilon)
    
    # 平均绝对相对误差
    abs_rel_error = torch.mean(rel_error).item()
    
    # 平均相对误差 (带符号)
    signed_rel_error = (t1 - t2) / (torch.abs(t2) + epsilon)
    mean_rel_error = torch.mean(signed_rel_error).item()
    
    # 最大相对误差
    max_rel_error = torch.max(rel_error).item()
    
    # 检查NaN和Inf
    has_nan_1 = torch.isnan(t1).any().item()
    has_nan_2 = torch.isnan(t2).any().item()
    has_inf_1 = torch.isinf(t1).any().item()
    has_inf_2 = torch.isinf(t2).any().item()
    
    print(f"\n{'='*70}")
    print(f"误差分析: {tensor_name}")
    print(f"{'='*70}")
    print(f"Tensor1 (测试) - NaN: {has_nan_1}, Inf: {has_inf_1}")
    print(f"Tensor2 (参考) - NaN: {has_nan_2}, Inf: {has_inf_2}")
    print(f"-"*70)
    print(f"平均绝对误差:           {mean_abs_error:.6e}")
    print(f"最大绝对误差:           {max_abs_error:.6e}")
    print(f"-"*70)
    print(f"平均绝对相对误差:       {abs_rel_error:.6e}")
    print(f"平均相对误差 (带符号):  {mean_rel_error:.6e}")
    print(f"最大相对误差:           {max_rel_error:.6e}")
    print(f"{'='*70}\n")
    
    return abs_rel_error, mean_rel_error, max_abs_error, max_rel_error


# ============================================
# Kernel 1: 单个chunk的V_new计算
# ============================================
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    }
)
def tilelang_compute_v_new_single_chunk(
    # task config
    B,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    use_g=True,
    # kernel config
    block_DK=128,
    block_DV=32,
    threads=128,
):
    """
    Kernel 1: 计算单个chunk的V_new
    
    输入:
    - K_chunk: (B, chunk_size, H, DK) - 当前chunk的K
    - W_chunk: (B, chunk_size, H, DK) - 当前chunk的W
    - U_chunk: (B, chunk_size, H, DV) - 当前chunk的U
    - G_chunk: (B, chunk_size, H) - 当前chunk的G
    - h_in: (B, H, DK, DV) - 输入的h状态
    
    输出:
    - V_new_chunk: (B, chunk_size, H, DV) - 计算得到的V_new
    - h_out: (B, H, DK, DV) - 更新后的h状态
    
    计算逻辑:
    1. V_new = U - W @ h_in
    2. 应用gate到V_new
    3. h_out = h_in * exp(G_last) + K^T @ V_new
    """
    block_S = chunk_size
    assert block_DK >= DK
    
    K_chunk_shape = (B, chunk_size, H, DK)
    W_chunk_shape = (B, chunk_size, H, DK)
    U_chunk_shape = (B, chunk_size, H, DV)
    G_chunk_shape = (B, chunk_size, H)
    h_shape = (B, H, DK, DV)
    V_chunk_shape = (B, chunk_size, H, DV)
    
    @T.prim_func
    def kernel(
            K_chunk: T.Tensor(K_chunk_shape, dtype=input_dtype),
            W_chunk: T.Tensor(W_chunk_shape, dtype=input_dtype),
            U_chunk: T.Tensor(U_chunk_shape, dtype=input_dtype),
            G_chunk: T.Tensor(G_chunk_shape, dtype=gate_dtype),
            h_in: T.Tensor(h_shape, dtype=input_dtype),  # 改为input_dtype避免往返精度损失
            V_new_chunk: T.Tensor(V_chunk_shape, dtype=output_dtype),
            h_out: T.Tensor(h_shape, dtype=input_dtype),  # 改为input_dtype避免往返精度损失
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            # 分配共享内存和fragment
            # 注意: b_h_shared必须使用input_dtype，避免与W_shared的混合精度GEMM问题
            b_h_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            b_h_fragment = T.alloc_fragment((block_DK, block_DV), dtype=accum_dtype)

            U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            # V_new_shared需要使用input_dtype以匹配K_shared的GEMM
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            
            G_last_local = T.alloc_local((1), dtype=gate_dtype)
            G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype)
            G_fragment = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)

            T.annotate_layout({
                b_h_shared: tilelang.layout.make_swizzled_layout(b_h_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                V_new_shared: tilelang.layout.make_swizzled_layout(V_new_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                G_shared: tilelang.layout.make_swizzled_layout(G_shared),
            })

            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()
            
            # 从输入加载h状态
            T.copy(h_in[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], b_h_shared)
            T.copy(b_h_shared, b_h_fragment)
            
            # 加载 W
            for i, j in T.Parallel(block_S, block_DK):
                W_shared[i, j] = W_chunk[bb, i, bh, j]
            
            # 第一个GEMM: 计算 W @ h
            T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)
            
            # 加载 U
            T.copy(
                U_chunk[bb, 0:block_S, bh, bv * block_DV:(bv + 1) * block_DV],
                U_shared)
            T.copy(U_shared, U_fragment)
            
            # V_new = U - W @ h
            for i_s2, i_v in T.Parallel(block_S, block_DV):
                V_new_fragment[i_s2, i_v] = -V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

            # 应用 gate 到 V_new
            if use_g:
                G_last_local[0] = G_chunk[bb, block_S - 1, bh]
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    G_shared[i_s2, i_v] = G_chunk[bb, i_s2, bh]
                T.copy(G_shared, G_fragment)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    with T.If(G_last_local[0] - G_fragment[i_s2, i_v] <= 0):
                        with T.Then():
                            V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(
                                G_last_local[0] - G_fragment[i_s2, i_v])
                        with T.Else():
                            V_new_fragment[i_s2, i_v] = 0
                G_last_local[0] = T.exp(G_last_local[0])
            
            # 保存 V_new 到输出
            T.copy(V_new_fragment, dst=V_new_shared)
            T.copy(
                V_new_shared, V_new_chunk[bb, 0:block_S, bh, bv * block_DV:(bv + 1) * block_DV])
            
            # 更新 h: h = h * exp(G_last) + K^T @ V_new
            # 加载 K
            for i, j in T.Parallel(block_S, DK):
                K_shared[i, j] = K_chunk[bb, i, bh, j]
            
            # 应用gate到h
            if use_g:
                for i_k, i_v in T.Parallel(DK, block_DV):
                    b_h_fragment[i_k, i_v] *= G_last_local[0]
            
            # 第二个GEMM: h = h + K^T @ V_new
            T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)
            
            # 保存更新后的h
            T.copy(b_h_fragment, b_h_shared)
            T.copy(b_h_shared[0:DK, 0:block_DV], h_out[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

    return kernel


# ============================================
# Kernel 2: 单个chunk的h更新（使用已有的V_new）
# ============================================
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    }
)
def tilelang_update_h_single_chunk(
    # task config
    B,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    use_g=True,
    # kernel config
    block_DK=128,
    block_DV=32,
    threads=128,
):
    """
    Kernel 2: 使用已有的V_new更新单个chunk的h状态
    
    输入:
    - K_chunk: (B, chunk_size, H, DK) - 当前chunk的K
    - V_new_chunk: (B, chunk_size, H, DV) - 已计算的V_new
    - G_chunk: (B, chunk_size, H) - 当前chunk的G
    - h_in: (B, H, DK, DV) - 输入的h状态
    
    输出:
    - h_out: (B, H, DK, DV) - 更新后的h状态
    
    计算逻辑:
    1. h_out = h_in * exp(G_last) + K^T @ V_new
    """
    block_S = chunk_size
    assert block_DK >= DK
    
    K_chunk_shape = (B, chunk_size, H, DK)
    V_chunk_shape = (B, chunk_size, H, DV)
    G_chunk_shape = (B, chunk_size, H)
    h_shape = (B, H, DK, DV)
    
    @T.prim_func
    def kernel(
            K_chunk: T.Tensor(K_chunk_shape, dtype=input_dtype),
            V_new_chunk: T.Tensor(V_chunk_shape, dtype=output_dtype),
            G_chunk: T.Tensor(G_chunk_shape, dtype=gate_dtype),
            h_in: T.Tensor(h_shape, dtype=input_dtype),  # 改为input_dtype避免往返精度损失
            h_out: T.Tensor(h_shape, dtype=input_dtype),  # 改为input_dtype避免往返精度损失
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            # 分配共享内存和fragment  
            # 注意: b_h_shared必须使用input_dtype，避免与K_shared的混合精度GEMM问题
            b_h_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            b_h_fragment = T.alloc_fragment((block_DK, block_DV), dtype=accum_dtype)

            # V_new_shared需要使用input_dtype以匹配K_shared的GEMM
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            
            G_last_local = T.alloc_local((1), dtype=gate_dtype)

            T.annotate_layout({
                b_h_shared: tilelang.layout.make_swizzled_layout(b_h_shared),
                V_new_shared: tilelang.layout.make_swizzled_layout(V_new_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
            })

            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()
            
            # 从输入加载h状态
            T.copy(h_in[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], b_h_shared)
            T.copy(b_h_shared, b_h_fragment)
            
            # 从HBM加载V_new
            T.copy(
                V_new_chunk[bb, 0:block_S, bh, bv * block_DV:(bv + 1) * block_DV],
                V_new_shared)
            
            # 加载 K
            for i, j in T.Parallel(block_S, DK):
                K_shared[i, j] = K_chunk[bb, i, bh, j]
            
            # 应用 gate 到 h
            if use_g:
                G_last_local[0] = G_chunk[bb, block_S - 1, bh]
                G_last_local[0] = T.exp(G_last_local[0])
                for i_k, i_v in T.Parallel(DK, block_DV):
                    b_h_fragment[i_k, i_v] *= G_last_local[0]
            
            # GEMM: h = h + K^T @ V_new
            T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)
            
            # 保存更新后的h
            T.copy(b_h_fragment, b_h_shared)
            T.copy(b_h_shared[0:DK, 0:block_DV], h_out[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

    return kernel


# ============================================
# 主函数: 循环调用单chunk kernel
# ============================================
def tilelang_chunk_gated_delta_rule_chunked(
    batch_size: int,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    chunk_size: int = 64,
):
    """
    分chunk调用版本: 每次kernel调用只处理一个chunk
    
    优势:
    1. 避免TileLang的T.serial/T.Pipelined同步问题
    2. 每个kernel调用简单，易于调试
    3. 状态管理在Python层面，更清晰
    4. 可以灵活控制每个chunk的处理
    
    逻辑:
    - 阶段1: 计算所有V_new
      for each chunk:
          V_new[i], h = kernel1(K[i], W[i], U[i], G[i], h)
    
    - 阶段2: 重新计算h和final_state
      for each chunk:
          h = kernel2(K[i], V_new[i], G[i], h)
    """
    B, Token, H, K_dim, V_dim = *k.shape, u.shape[-1]
    BS = (Token + chunk_size - 1) // chunk_size
    
    # 分配输出tensor
    h_states = k.new_empty(B, BS, H, K_dim, V_dim)
    v_new = torch.empty_like(u)
    final_state = k.new_empty(B, H, K_dim, V_dim, dtype=torch.float32) if output_final_state else None
    
    # 创建kernel
    kernel1 = tilelang_compute_v_new_single_chunk(
        B, H, K_dim, V_dim,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=chunk_size,
        use_g=g is not None,
    )
    
    kernel2 = tilelang_update_h_single_chunk(
        B, H, K_dim, V_dim,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=chunk_size,
        use_g=g is not None,
    )
    
    # 初始化h状态 (在Python层面用float32保持精度)
    if initial_state is not None:
        h_current_fp32 = initial_state.clone()
    else:
        h_current_fp32 = torch.zeros(B, H, K_dim, V_dim, dtype=torch.float32, device=k.device)
    
    # 阶段1: 计算所有V_new，逐chunk处理
    print(f"阶段1: 逐chunk计算V_new (共 {BS} 个chunks)...")
    for i_chunk in range(BS):
        start_idx = i_chunk * chunk_size
        end_idx = min((i_chunk + 1) * chunk_size, Token)
        actual_size = end_idx - start_idx
        
        # 准备当前chunk的输入
        if actual_size == chunk_size:
            k_chunk = k[:, start_idx:end_idx, :, :]
            w_chunk = w[:, start_idx:end_idx, :, :]
            u_chunk = u[:, start_idx:end_idx, :, :]
            g_chunk = g[:, start_idx:end_idx, :] if g is not None else None
            v_new_chunk = v_new[:, start_idx:end_idx, :, :]
        else:
            # 最后一个chunk可能不满，需要padding
            k_chunk = torch.zeros(B, chunk_size, H, K_dim, dtype=k.dtype, device=k.device)
            w_chunk = torch.zeros(B, chunk_size, H, K_dim, dtype=w.dtype, device=w.device)
            u_chunk = torch.zeros(B, chunk_size, H, V_dim, dtype=u.dtype, device=u.device)
            g_chunk = torch.zeros(B, chunk_size, H, dtype=g.dtype, device=g.device) if g is not None else None
            
            k_chunk[:, :actual_size, :, :] = k[:, start_idx:end_idx, :, :]
            w_chunk[:, :actual_size, :, :] = w[:, start_idx:end_idx, :, :]
            u_chunk[:, :actual_size, :, :] = u[:, start_idx:end_idx, :, :]
            if g is not None:
                g_chunk[:, :actual_size, :] = g[:, start_idx:end_idx, :]
            
            v_new_chunk = torch.zeros(B, chunk_size, H, V_dim, dtype=u.dtype, device=u.device)
        
        # 分配输出 (kernel输出bfloat16)
        h_next_bf16 = torch.empty(B, H, K_dim, V_dim, dtype=torch.bfloat16, device=k.device)
        
        # 调用kernel1 (输入输出都是bfloat16)
        h_current_bf16 = h_current_fp32.to(torch.bfloat16)
        kernel1(k_chunk, w_chunk, u_chunk, g_chunk, h_current_bf16, v_new_chunk, h_next_bf16)
        
        # 保存结果
        if actual_size == chunk_size:
            v_new[:, start_idx:end_idx, :, :] = v_new_chunk
        else:
            v_new[:, start_idx:end_idx, :, :] = v_new_chunk[:, :actual_size, :, :]
        
        # 更新h状态 (转回float32保持精度)
        h_current_fp32 = h_next_bf16.to(torch.float32)
        
        torch.cuda.synchronize()
    
    # 阶段2: 使用V_new重新计算h和final_state
    print(f"阶段2: 使用V_new重新计算h (共 {BS} 个chunks)...")
    # 重新初始化h (在Python层面用float32保持精度)
    if initial_state is not None:
        h_current_fp32 = initial_state.clone()
    else:
        h_current_fp32 = torch.zeros(B, H, K_dim, V_dim, dtype=torch.float32, device=k.device)
    
    for i_chunk in range(BS):
        start_idx = i_chunk * chunk_size
        end_idx = min((i_chunk + 1) * chunk_size, Token)
        actual_size = end_idx - start_idx
        
        # 保存当前chunk开始时的h (转为bfloat16保存)
        h_states[:, i_chunk, :, :, :] = h_current_fp32.to(k.dtype)
        
        # 准备当前chunk的输入
        if actual_size == chunk_size:
            k_chunk = k[:, start_idx:end_idx, :, :]
            v_new_chunk = v_new[:, start_idx:end_idx, :, :]
            g_chunk = g[:, start_idx:end_idx, :] if g is not None else None
        else:
            # 最后一个chunk可能不满，需要padding
            k_chunk = torch.zeros(B, chunk_size, H, K_dim, dtype=k.dtype, device=k.device)
            v_new_chunk = torch.zeros(B, chunk_size, H, V_dim, dtype=v_new.dtype, device=v_new.device)
            g_chunk = torch.zeros(B, chunk_size, H, dtype=g.dtype, device=g.device) if g is not None else None
            
            k_chunk[:, :actual_size, :, :] = k[:, start_idx:end_idx, :, :]
            v_new_chunk[:, :actual_size, :, :] = v_new[:, start_idx:end_idx, :, :]
            if g is not None:
                g_chunk[:, :actual_size, :] = g[:, start_idx:end_idx, :]
        
        # 分配输出 (kernel输出bfloat16)
        h_next_bf16 = torch.empty(B, H, K_dim, V_dim, dtype=torch.bfloat16, device=k.device)
        
        # 调用kernel2 (输入输出都是bfloat16)
        h_current_bf16 = h_current_fp32.to(torch.bfloat16)
        kernel2(k_chunk, v_new_chunk, g_chunk, h_current_bf16, h_next_bf16)
        
        # 更新h状态 (转回float32保持精度)
        h_current_fp32 = h_next_bf16.to(torch.float32)
        
        torch.cuda.synchronize()
    
    # 保存final_state (float32精度)
    if output_final_state:
        final_state[:] = h_current_fp32
    
    return h_states, v_new, final_state


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*10 + "对比测试: 分chunk调用版本 vs 原版")
    print("="*70 + "\n")
    
    B, Token, H, K, V = 1, 1024, 128, 128, 128
    input_dtype = torch.bfloat16
    element_dtype = torch.float32

    # 加载测试数据
    print("加载测试数据...")
    k = torch.load("./k.pt", map_location="cuda")
    w = torch.load("./w.pt", map_location="cuda")
    u = torch.load("./u.pt", map_location="cuda")
    g = torch.load("./g.pt", map_location="cuda")
    
    print(f"  k shape: {k.shape}")
    print(f"  w shape: {w.shape}")
    print(f"  u shape: {u.shape}")
    print(f"  g shape: {g.shape}")
    
    # 创建输入的副本
    k_copy = k.clone()
    w_copy = w.clone()
    u_copy = u.clone()
    g_copy = g.clone()
    
    print("\n" + "-"*70)
    print("运行原版 kernel...")
    print("-"*70)
    h_original, v_new_original, final_state_original = tilelang_chunk_gated_delta_rule_original(
        1, k, w, u, g
    )
    h_original = h_original.to(torch.bfloat16)
    v_new_original = v_new_original.to(torch.bfloat16)
    final_state_original = final_state_original.to(torch.bfloat16)
    
    print("原版完成！")
    check_tensors_and_compute_errors(h_original, "h_original")
    check_tensors_and_compute_errors(v_new_original, "v_new_original")
    check_tensors_and_compute_errors(final_state_original, "final_state_original")
    
    print("\n" + "-"*70)
    print("运行分chunk调用版本 kernel...")
    print("-"*70)
    h_chunked, v_new_chunked, final_state_chunked = tilelang_chunk_gated_delta_rule_chunked(
        1, k_copy, w_copy, u_copy, g_copy
    )
    h_chunked = h_chunked.to(torch.bfloat16)
    v_new_chunked = v_new_chunked.to(torch.bfloat16)
    final_state_chunked = final_state_chunked.to(torch.bfloat16)
    
    print("分chunk版本完成！")
    check_tensors_and_compute_errors(h_chunked, "h_chunked")
    check_tensors_and_compute_errors(v_new_chunked, "v_new_chunked")
    check_tensors_and_compute_errors(final_state_chunked, "final_state_chunked")
    
    # 比较两个版本的结果
    print("\n" + "="*70)
    print(" "*20 + "误差对比分析")
    print("="*70)
    
    # 比较 h
    compute_relative_errors(h_chunked, h_original, "h (分chunk版 vs 原版)")
    
    # 比较 v_new
    compute_relative_errors(v_new_chunked, v_new_original, "v_new (分chunk版 vs 原版)")
    
    # 比较 final_state
    compute_relative_errors(final_state_chunked, final_state_original, "final_state (分chunk版 vs 原版)")
    
    print("\n" + "="*70)
    print(" "*25 + "测试完成！")
    print("="*70 + "\n")

