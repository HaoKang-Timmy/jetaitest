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
    
    Args:
        tensor1: 第一个tensor (通常是测试结果)
        tensor2: 第二个tensor (通常是参考结果)
        tensor_name: tensor的名称，用于打印
    
    Returns:
        abs_rel_error: 平均绝对相对误差
        mean_rel_error: 平均相对误差
        max_abs_error: 最大绝对误差
        max_rel_error: 最大相对误差
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
# Kernel 1: 计算 V_new (实际包含两个GEMM)
# ============================================
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    }
)
def tilelang_compute_v_new(
    # task config
    B,
    S,
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
    save_h_intermediate=True,
    # kernel config
    block_DK=128,
    block_DV=32,
    threads=128,
    num_stages=2,
):
    """
    Kernel 1: 计算所有 chunk 的 V_new
    
    重要说明:
    - 输入: K, W, U, G
    - 输出: V_new (保存到HBM), h_intermediate (每个chunk开始时的h状态)
    - 逻辑: 
        对于每个chunk i:
            1. 保存 h[i] (chunk开始时的状态)
            2. 第一个GEMM: 计算 V_new[i] = U[i] - W[i] @ h[i]
            3. 应用gate到V_new[i]
            4. 保存V_new[i]到HBM
            5. 临时更新h: h = h * exp(G) + K[i]^T @ V_new[i] (第二个GEMM)
               这一步是必须的，否则下一个chunk的V_new[i+1]会计算错误
    
    注意: 虽然这个kernel包含了两个GEMM，但可以通过TileLang的分离编译来优化
    """
    block_S = chunk_size
    BS = (S + block_S - 1) // block_S

    W_shape = (B, S, H, DK)
    U_shape = (B, S, H, DV)
    G_shape = (B, S, H)
    V_shape = (B, S, H, DV)
    h_shape = (B, BS, H, DK, DV)
    
    assert block_DK >= DK
    
    @T.prim_func
    def kernel(
            K: T.Tensor((B, S, H, DK), dtype=input_dtype),  # 需要K来临时更新h
            W: T.Tensor(W_shape, dtype=input_dtype),
            U: T.Tensor(U_shape, dtype=input_dtype),
            G: T.Tensor(G_shape, dtype=gate_dtype),
            V_new: T.Tensor(V_shape, dtype=output_dtype),
            h_intermediate: T.Tensor(h_shape, dtype=output_dtype),  # 保存每个chunk开始时的h
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            # 分配共享内存和fragment
            b_h_shared = T.alloc_shared((block_DK, block_DV), dtype=output_dtype)
            b_h_fragment = T.alloc_fragment((block_DK, block_DV), dtype=accum_dtype)

            U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            
            G_last_local = T.alloc_local((1), dtype=gate_dtype)
            G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype)
            G_fragment = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)
            
            # K用于临时更新h (为了下一个chunk的V_new计算)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)

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
            
            # 初始化 h = 0
            T.clear(b_h_fragment)
            T.copy(b_h_fragment, b_h_shared)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # 保存当前chunk开始时的h状态 (用于验证)
                if save_h_intermediate:
                    T.copy(b_h_shared[0:DK, 0:block_DV], 
                           h_intermediate[bb, i_s, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])
                
                # 加载 W
                for i, j in T.Parallel(block_S, block_DK):
                    W_shared[i, j] = W[bb, i_s * block_S + i, bh, j]
                
                # 第一个GEMM: 计算 W @ h
                T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)
                
                # 加载 U
                T.copy(
                    U[bb, i_s * block_S:(i_s + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV],
                    U_shared)
                T.copy(U_shared, U_fragment)
                
                # V_new = U - W @ h
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    V_new_fragment[i_s2, i_v] = -V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

                # 应用 gate 到 V_new
                if use_g:
                    last_idx = T.min((i_s + 1) * block_S, S) - 1
                    G_last_local[0] = G[bb, last_idx, bh]
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        G_shared[i_s2, i_v] = G[bb, i_s * block_S + i_s2, bh]
                    T.copy(G_shared, G_fragment)
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        with T.If(G_last_local[0] - G_fragment[i_s2, i_v] <= 0):
                            with T.Then():
                                V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(
                                    G_last_local[0] - G_fragment[i_s2, i_v])
                            with T.Else():
                                V_new_fragment[i_s2, i_v] = 0
                    G_last_local[0] = T.exp(G_last_local[0])
                
                # 保存 V_new 到 HBM
                T.copy(V_new_fragment, dst=V_new_shared)
                T.copy(
                    V_new_shared, V_new[bb, i_s * block_S:(i_s + 1) * block_S, bh,
                                        bv * block_DV:(bv + 1) * block_DV])
                
                # 临时更新 h (为了下一个chunk能计算正确的V_new)
                # 必须做完整的状态更新: h = h * exp(G_last) + K^T @ V_new
                
                # 加载 K
                for i, j in T.Parallel(block_S, DK):
                    K_shared[i, j] = K[bb, i_s * block_S + i, bh, j]
                
                # 应用gate到h
                if use_g:
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_h_fragment[i_k, i_v] *= G_last_local[0]
                
                # 第二个GEMM: h = h + K^T @ V_new (临时更新，不保存)
                T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)
                
                T.copy(b_h_fragment, b_h_shared)
                
    return kernel


# ============================================
# Kernel 2: 用V_new计算h和final_state (包含第二个GEMM)
# ============================================
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    }
)
def tilelang_compute_h_from_v_new(
    # task config
    B,
    S,
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
    store_final_state=True,
    # kernel config
    block_DK=128,
    block_DV=32,
    threads=128,
    num_stages=2,
):
    """
    Kernel 2: 从V_new重新计算h和final_state
    
    - 输入: K, V_new (从HBM加载，由Kernel 1生成), G
    - 输出: h (每个chunk结束后的状态), final_state
    - 逻辑:
        初始化 h = 0
        对于每个chunk i:
            1. 保存 h[i] (实际上是前一个chunk结束的状态，也是当前chunk开始的状态)
            2. 从HBM加载V_new[i]
            3. 应用gate: h = h * exp(G)
            4. 第二个GEMM: h = h + K[i]^T @ V_new[i]
        最后保存final_state
    
    注意: 这个kernel的h输出应该与原版kernel完全一致
    """
    block_S = chunk_size
    BS = (S + block_S - 1) // block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    G_shape = (B, S, H)
    h_shape = (B, BS, H, DK, DV)
    final_state_shape = (B, H, DK, DV)
    
    assert block_DK >= DK
    
    @T.prim_func
    def kernel(
            K: T.Tensor(K_shape, dtype=input_dtype),
            V_new: T.Tensor(V_shape, dtype=output_dtype),  # 从kernel 1的输出加载
            G: T.Tensor(G_shape, dtype=gate_dtype),
            h: T.Tensor(h_shape, dtype=output_dtype),
            final_state: T.Tensor(final_state_shape, dtype=state_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            # 分配共享内存和fragment
            b_h_shared = T.alloc_shared((block_DK, block_DV), dtype=output_dtype)
            b_h_fragment = T.alloc_fragment((block_DK, block_DV), dtype=accum_dtype)

            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            
            G_last_local = T.alloc_local((1), dtype=gate_dtype)

            T.annotate_layout({
                b_h_shared: tilelang.layout.make_swizzled_layout(b_h_shared),
                V_new_shared: tilelang.layout.make_swizzled_layout(V_new_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
            })

            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()
            
            # 初始化 h = 0
            T.clear(b_h_fragment)
            T.copy(b_h_fragment, b_h_shared)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # 保存当前chunk结束后的h (实际上是下一个chunk开始的h)
                T.copy(b_h_shared[0:DK, 0:block_DV], 
                       h[bb, i_s, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])
                
                # 从HBM加载V_new (已经包含gate)
                T.copy(
                    V_new[bb, i_s * block_S:(i_s + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV],
                    V_new_shared)
                
                # 加载 K
                for i, j in T.Parallel(block_S, DK):
                    K_shared[i, j] = K[bb, i_s * block_S + i, bh, j]
                
                # 应用 gate 到 h
                if use_g:
                    last_idx = T.min((i_s + 1) * block_S, S) - 1
                    G_last_local[0] = G[bb, last_idx, bh]
                    G_last_local[0] = T.exp(G_last_local[0])
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_h_fragment[i_k, i_v] *= G_last_local[0]
                
                # 第二个GEMM: h = h + K^T @ V_new
                T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)
                
                T.copy(b_h_fragment, b_h_shared)

            # 保存 final_state
            if store_final_state:
                T.copy(b_h_fragment[0:DK, 0:block_DV], 
                       final_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

    return kernel


# ============================================
# 主函数: 调用两个kernel
# ============================================
def tilelang_chunk_gated_delta_rule_split(
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
    拆分版本: 调用两个kernel来完成计算
    
    拆分策略说明:
    1. Kernel 1 计算并保存 V_new 到 HBM
       - 必须包含完整的状态更新逻辑（两个GEMM）
       - 原因: V_new[i] 依赖于 h[i]，而 h[i] 依赖于 V_new[0:i-1]
       
    2. Kernel 2 从 HBM 读取 V_new，重新计算 h 和 final_state
       - 只包含第二个GEMM (K^T @ V_new)
       - 用于生成最终需要的 h 和 final_state
    
    优势:
    - V_new 被显式保存，可以用于其他用途（如反向传播）
    - Kernel 2 可以独立运行，灵活性更高
    
    劣势:
    - 增加了 HBM 访问（V_new 写入和读取）
    - Kernel 1 实际上还是包含两个GEMM（为了保证迭代正确性）
    
    数学等价性: 与原版kernel数学上完全等价
    """
    B, Token, H, K, V = *k.shape, u.shape[-1]
    BS = (Token + chunk_size - 1) // chunk_size
    
    # 分配输出tensor
    h = k.new_empty(B, BS, H, K, V)
    final_state = k.new_empty(B, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)
    h_intermediate = k.new_empty(B, BS, H, K, V)  # 用于验证
    
    # Kernel 1: 计算 V_new
    print("Running Kernel 1: Computing V_new...")
    kernel1 = tilelang_compute_v_new(
        B, Token, H, K, V,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=64,
        use_g=g is not None,
        save_h_intermediate=True,
    )
    kernel1(k, w, u, g, v_new, h_intermediate)
    torch.cuda.synchronize()
    # Kernel 2: 用 V_new 计算 h 和 final_state
    print("Running Kernel 2: Computing h and final_state from V_new...")
    kernel2 = tilelang_compute_h_from_v_new(
        B, Token, H, K, V,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=64,
        use_g=g is not None,
        store_final_state=output_final_state,
    )
    kernel2(k, v_new, g, h, final_state)
    
    return h, v_new, final_state


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "对比测试: 拆分版本 vs 原版")
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
    
    # 创建输入的副本（确保两个版本使用相同的输入）
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
    print("运行拆分版本 kernel...")
    print("-"*70)
    h_split, v_new_split, final_state_split = tilelang_chunk_gated_delta_rule_split(
        1, k_copy, w_copy, u_copy, g_copy
    )
    h_split = h_split.to(torch.bfloat16)
    v_new_split = v_new_split.to(torch.bfloat16)
    final_state_split = final_state_split.to(torch.bfloat16)
    
    print("拆分版本完成！")
    check_tensors_and_compute_errors(h_split, "h_split")
    check_tensors_and_compute_errors(v_new_split, "v_new_split")
    check_tensors_and_compute_errors(final_state_split, "final_state_split")
    
    # 比较两个版本的结果
    print("\n" + "="*70)
    print(" "*20 + "误差对比分析")
    print("="*70)
    
    # 比较 h
    compute_relative_errors(h_split, h_original, "h (拆分版 vs 原版)")
    
    # 比较 v_new
    compute_relative_errors(v_new_split, v_new_original, "v_new (拆分版 vs 原版)")
    
    # 比较 final_state
    compute_relative_errors(final_state_split, final_state_original, "final_state (拆分版 vs 原版)")
    
    print("\n" + "="*70)
    print(" "*25 + "测试完成！")
    print("="*70 + "\n")

