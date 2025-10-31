import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
from typing import Optional, Tuple
import time
import tilelang as tl

print("正在导入kernels...")

# ============================================================================
# Kernel 1: 从 tl_fusedrecurrent.py (edge版本)
# ============================================================================
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_recurrent_edge(
    Batch, 
    Token, 
    Head, 
    Head_V, 
    K_Dim, 
    V_Dim, 
    USE_QK_L2NORM_IN_KERNEL=True, 
    STORE_FINAL_STATE=True,
    dtype = "bfloat16",
    accum_dtype = "float32",
    block_K=128, 
    block_V=128, 
    num_stages=2, 
    threads=128
):
    @T.macro
    def L2Norm_QK(
        QK: T.SharedBuffer([block_K],accum_dtype),
    ):
        shared_reg = T.alloc_fragment([block_K], accum_dtype)
        squared_reg = T.alloc_fragment([block_K], accum_dtype)
        sum_reg = T.alloc_fragment([1], accum_dtype)
        T.copy(QK, shared_reg)
        
        # 计算元素的平方用于求norm
        for i in T.Parallel(block_K):
            squared_reg[i] = shared_reg[i] * shared_reg[i]
        T.reduce_sum(squared_reg, sum_reg, dim=0)
        sum_reg[0] = T.sqrt(sum_reg[0]) + 1e-6
        
        # 用原始元素除以norm
        for i in T.Parallel(block_K):
            shared_reg[i] = shared_reg[i] / sum_reg[0]
        T.copy(shared_reg, QK)

    @T.prim_func
    def main(
        Q: T.Tensor([Batch, Token, Head, K_Dim], dtype),
        K: T.Tensor([Batch, Token, Head, K_Dim], dtype),
        V: T.Tensor([Batch, Token, Head_V, V_Dim], dtype),
        g: T.Tensor([Batch, Token, Head_V], accum_dtype),
        Beta: T.Tensor([Batch, Token, Head_V],dtype),
        O: T.Tensor([Batch, Token, Head_V, V_Dim], dtype),
        h0: T.Tensor([Batch, Head_V, K_Dim, V_Dim], accum_dtype),
        scale: T.Tensor([1], dtype),
    ):
        with T.Kernel(T.ceildiv(K_Dim, block_K), T.ceildiv(V_Dim, block_V), T.ceildiv(Batch*Head_V, 1), threads=threads) as (bx, by, bz):
            id_b = bz // Head_V
            id_hv = bz % Head_V
            id_h = id_hv // (Head_V // Head)
            Q_shared = T.alloc_shared([block_K], dtype)
            Q_fragment = T.alloc_fragment([block_K], dtype)
            K_shared = T.alloc_shared([block_K], dtype)
            V_shared = T.alloc_shared([block_V], dtype)
            V_fragment = T.alloc_fragment([block_V], dtype)
            h0_shared = T.alloc_shared([block_K, block_V], accum_dtype)
            o_shared = T.alloc_shared([block_V], dtype)
            
            h0_fragment = T.alloc_fragment([block_K, block_V], accum_dtype)
            K_fragment = T.alloc_fragment([block_K], accum_dtype)
            v_min_reg = T.alloc_fragment([block_V], accum_dtype)

            
            T.copy(h0[id_b, id_hv, bx * block_K, by * block_V], h0_shared)
  
            for t in T.serial(Token):
                T.copy(Q[id_b, t, id_h, bx * block_K], Q_shared)
                T.copy(K[id_b, t, id_h, bx * block_K], K_shared)
                T.copy(V[id_b, t, id_hv, by * block_V], V_shared)

                T.copy(Q_shared, Q_fragment)
                T.copy(K_shared, K_fragment)
                T.copy(V_shared, V_fragment)
                T.copy(h0_shared, h0_fragment)

                for i in T.Parallel(block_K):
                    Q_fragment[i] = Q_fragment[i] * scale[0]

                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * T.exp(g[id_b, t, id_hv])
                
                T.copy(h0_fragment, h0_shared)
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * K_fragment[i]
                T.reduce_sum(h0_fragment, v_min_reg, dim=0)
                for i in T.Parallel(block_V):
                    V_fragment[i] = V_fragment[i] - v_min_reg[i]
                
                for i in T.Parallel(block_V):
                    V_fragment[i] = V_fragment[i] * Beta[id_b, t, id_hv]
                
                T.copy(h0_shared, h0_fragment)
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] + K_fragment[i] * V_fragment[j]
                
                T.copy(h0_fragment, h0_shared)
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * Q_fragment[i]
                
                T.reduce_sum(h0_fragment, v_min_reg, dim=0)
                T.copy(v_min_reg, o_shared)
                
                T.copy(o_shared, O[id_b, t, id_hv, by * block_V])
                

            if STORE_FINAL_STATE:
                T.copy(h0_shared, h0[id_b, id_hv, bx * block_K, by * block_V])
        
    return main

# ============================================================================
# Kernel 2: 从 tl_fusedrecurrent_bf16.py (bf16版本)
# ============================================================================
@tilelang.jit(
    out_idx = [-1],
)
def fused_recurrent_bf16(
    Batch, 
    Token, 
    Head, 
    Head_V, 
    K_Dim, 
    V_Dim, 
    USE_QK_L2NORM_IN_KERNEL=True, 
    STORE_FINAL_STATE=True,
    dtype = "bfloat16",
    accum_dtype = "float32",
    block_K=128, 
    block_V=128, 
    num_stages=3, 
    threads=128
):
    @T.macro
    def L2Norm_QK(
        QK: T.FragmentBuffer([block_K],dtype),
    ):
        squared_reg = T.alloc_fragment([block_K], dtype)
        sum_reg = T.alloc_fragment([1], dtype)
        
        for i in T.Parallel(block_K):
            squared_reg[i] = QK[i] * QK[i]
        T.reduce_sum(squared_reg, sum_reg, dim=0)
        sum_reg[0] = T.sqrt(sum_reg[0]) + 1e-6
        
        for i in T.Parallel(block_K):
            QK[i] = QK[i] / sum_reg[0]


    @T.prim_func
    def main(
        Q: T.Tensor([Batch, Token, Head, K_Dim], dtype),
        K: T.Tensor([Batch, Token, Head, K_Dim], dtype),
        V: T.Tensor([Batch, Token, Head_V, V_Dim], dtype),
        g: T.Tensor([Batch, Token, Head_V], accum_dtype),
        Beta: T.Tensor([Batch, Token, Head_V],dtype),
        h0: T.Tensor([Batch, Head_V, K_Dim, V_Dim], accum_dtype),
        scale: T.Tensor([1], dtype),
        O: T.Tensor([Batch, Token, Head_V, V_Dim], dtype),
    ):
        with T.Kernel(T.ceildiv(K_Dim, block_K), T.ceildiv(V_Dim, block_V), T.ceildiv(Batch*Head_V, 1), threads=threads) as (bx, by, bz):
            id_b = bz // Head_V
            id_hv = bz % Head_V
            id_h = id_hv // (Head_V // Head)
            Q_shared = T.alloc_shared([block_K], dtype)
            Q_fragment = T.alloc_fragment([block_K], dtype)
            K_shared = T.alloc_shared([block_K], dtype)
            V_shared = T.alloc_shared([block_V], dtype)
            V_fragment = T.alloc_fragment([block_V], dtype)
            h0_shared = T.alloc_shared([block_K, block_V], accum_dtype)
            o_shared = T.alloc_shared([block_V], dtype)
            
            h0_fragment = T.alloc_fragment([block_K, block_V], accum_dtype)
            K_fragment = T.alloc_fragment([block_K], accum_dtype)
            v_min_reg = T.alloc_fragment([block_V], accum_dtype)
            
            T.copy(h0[id_b, id_hv, bx * block_K, by * block_V], h0_shared)
            for t in T.serial(Token):
                T.copy(Q[id_b, t, id_h, bx * block_K], Q_shared)
                T.copy(K[id_b, t, id_h, bx * block_K], K_shared)
                T.copy(V[id_b, t, id_hv, by * block_V], V_shared)
 
                T.copy(Q_shared, Q_fragment)
                T.copy(K_shared, K_fragment)
                T.copy(V_shared, V_fragment)
                T.copy(h0_shared, h0_fragment)
                
                for i in T.Parallel(block_K):
                    Q_fragment[i] = Q_fragment[i] * scale[0]
                
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * T.exp(g[id_b, t, id_hv])
                T.copy(h0_fragment, h0_shared)

                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * K_fragment[i]
                T.reduce_sum(h0_fragment, v_min_reg, dim=0)
                for i in T.Parallel(block_V):
                    V_fragment[i] = V_fragment[i] - v_min_reg[i]
                
                for i in T.Parallel(block_V):
                    V_fragment[i] = V_fragment[i] * Beta[id_b, t, id_hv]
                
                T.copy(h0_shared, h0_fragment)
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] + K_fragment[i] * V_fragment[j]
                
                T.copy(h0_fragment, h0_shared)
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * Q_fragment[i]
                
                T.reduce_sum(h0_fragment, v_min_reg, dim=0)
                T.copy(v_min_reg, o_shared)
                
                T.copy(o_shared, O[id_b, t, id_hv, by * block_V])
                

            if STORE_FINAL_STATE:
                T.copy(h0_shared, h0[id_b, id_hv, bx * block_K, by * block_V])
        
    return main


# ============================================================================
# 辅助函数
# ============================================================================
def calculate_relative_errors(actual, expected, eps=1e-10):
    """
    计算平均相对误差和最大相对误差
    """
    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    
    abs_diff = torch.abs(actual_flat - expected_flat)
    expected_abs = torch.abs(expected_flat)
    
    non_zero_mask = expected_abs > eps
    relative_errors = torch.zeros_like(abs_diff)
    relative_errors[non_zero_mask] = abs_diff[non_zero_mask] / expected_abs[non_zero_mask]
    
    zero_mask = ~non_zero_mask
    actual_abs = torch.abs(actual_flat)
    relative_errors[zero_mask] = torch.where(
        actual_abs[zero_mask] <= eps, 
        torch.zeros_like(actual_abs[zero_mask]), 
        actual_abs[zero_mask]
    )
    
    mean_rel_error = torch.mean(relative_errors).item()
    max_rel_error = torch.max(relative_errors).item()
    
    return mean_rel_error, max_rel_error


# ============================================================================
# 主测试函数
# ============================================================================
def run_comparison():
    print("\n" + "="*80)
    print("开始对比两个Kernel实现")
    print("="*80)
    
    print("\n📋 Kernel配置对比:")
    print("  相同点:")
    print("    - 核心计算逻辑完全相同")
    print("    - accum_dtype: float32")
    print("    - h0类型: float32 (accum_dtype)")
    print("    - h0_shared类型: float32 (accum_dtype)")
    print("  差异点:")
    print("    - 装饰器: Edge使用pass_configs, BF16使用out_idx=[-1]")
    print("    - 参数顺序: Edge是(Q,K,V,g,Beta,O,h0,scale), BF16是(Q,K,V,g,Beta,h0,scale,O)")
    print("    - 默认num_stages: Edge=2, BF16=3")
    
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 测试参数
    B, Token, H, HV, K, V = 1, 10, 1, 1, 128, 128
    scale = K ** -0.5
    datatype = torch.bfloat16
    gdtype = torch.float32
    
    print(f"\n测试配置:")
    print(f"  Batch: {B}, Token: {Token}, Head: {H}, Head_V: {HV}")
    print(f"  K_Dim: {K}, V_Dim: {V}")
    print(f"  Scale: {scale:.6f}")
    print(f"  数据类型: {datatype}")
    print(f"  h0类型: {gdtype} (两个kernel统一使用)")
    
    # 生成测试数据
    q = torch.randn(B, Token, H, K, device='cuda', dtype=datatype)
    k = torch.randn(B, Token, H, K, device='cuda', dtype=datatype)
    v = torch.randn(B, Token, HV, V, device='cuda', dtype=datatype)
    g = torch.randn(B, Token, HV, device='cuda', dtype=gdtype).sigmoid()
    beta = torch.randn(B, Token, HV, device='cuda', dtype=datatype).sigmoid()
    
    # 为两个kernel准备相同配置的初始状态（都使用float32）
    h0_edge = torch.zeros(B, HV, K, V, device='cuda', dtype=gdtype)
    h0_bf16 = torch.zeros(B, HV, K, V, device='cuda', dtype=gdtype)
    
    # 预分配输出张量
    o_edge = torch.empty(B, Token, HV, V, device='cuda', dtype=datatype)
    o_bf16 = torch.empty(B, Token, HV, V, device='cuda', dtype=datatype)
    scale_tensor = torch.tensor([scale], device='cuda', dtype=datatype)
    
    # 运行Edge版本 kernel
    print("\n" + "-"*80)
    print("1. 运行Edge版本 Kernel (tl_fusedrecurrent.py)")
    print("-"*80)
    kernel_edge = fused_recurrent_edge(
        B, Token, H, HV, K, V, 
        USE_QK_L2NORM_IN_KERNEL=True, 
        STORE_FINAL_STATE=True,
        block_K=128, 
        block_V=128, 
        num_stages=2, 
        threads=128
    )
    
    torch.cuda.synchronize()
    start_edge = time.time()
    kernel_edge(q, k, v, g, beta, o_edge, h0_edge, scale_tensor)
    torch.cuda.synchronize()
    end_edge = time.time()
    
    print(f"  ✓ 执行完成")
    print(f"  执行时间: {(end_edge - start_edge) * 1000:.3f} ms")
    print(f"  输出形状: {o_edge.shape}")
    print(f"  Final state形状: {h0_edge.shape}")
    
    # 运行BF16版本 kernel
    print("\n" + "-"*80)
    print("2. 运行BF16版本 Kernel (tl_fusedrecurrent_bf16.py)")
    print("-"*80)
    kernel_bf16 = fused_recurrent_bf16(
        B, Token, H, HV, K, V, 
        USE_QK_L2NORM_IN_KERNEL=True, 
        STORE_FINAL_STATE=True,
        block_K=128, 
        block_V=128, 
        num_stages=3, 
        threads=128
    )
    
    torch.cuda.synchronize()
    start_bf16 = time.time()
    # 注意：BF16版本参数顺序是 (q, k, v, g, beta, h0, scale) 且O通过返回值
    o_bf16_result = kernel_bf16(q, k, v, g, beta, h0_bf16, scale_tensor)
    # 如果返回的是单个张量，直接使用；如果out_idx有效，会返回O
    o_bf16 = o_bf16_result if isinstance(o_bf16_result, torch.Tensor) else o_bf16_result
    torch.cuda.synchronize()
    end_bf16 = time.time()
    
    print(f"  ✓ 执行完成")
    print(f"  执行时间: {(end_bf16 - start_bf16) * 1000:.3f} ms")
    print(f"  输出形状: {o_bf16.shape}")
    print(f"  Final state形状: {h0_bf16.shape}")
    
    # 对比结果
    print("\n" + "="*80)
    print("结果对比")
    print("="*80)
    
    # 输出O的对比
    print("\n📊 输出张量 'O' 的误差分析:")
    o_mean_err, o_max_err = calculate_relative_errors(o_edge, o_bf16)
    print(f"  平均相对误差: {o_mean_err:.6e}")
    print(f"  最大相对误差: {o_max_err:.6e}")
    print(f"  绝对误差统计:")
    abs_diff_o = torch.abs(o_edge - o_bf16)
    print(f"    - 平均绝对误差: {abs_diff_o.mean().item():.6e}")
    print(f"    - 最大绝对误差: {abs_diff_o.max().item():.6e}")
    print(f"    - 最小绝对误差: {abs_diff_o.min().item():.6e}")
    
    # Final state的对比 (两者都是float32，无需转换)
    print("\n📊 Final State 'h0' 的误差分析:")
    h0_mean_err, h0_max_err = calculate_relative_errors(
        h0_edge, h0_bf16
    )
    print(f"  平均相对误差: {h0_mean_err:.6e}")
    print(f"  最大相对误差: {h0_max_err:.6e}")
    print(f"  绝对误差统计:")
    abs_diff_h0 = torch.abs(h0_edge - h0_bf16)
    print(f"    - 平均绝对误差: {abs_diff_h0.mean().item():.6e}")
    print(f"    - 最大绝对误差: {abs_diff_h0.max().item():.6e}")
    print(f"    - 最小绝对误差: {abs_diff_h0.min().item():.6e}")
    
    # 性能对比
    print("\n⚡ 性能对比:")
    time_edge_ms = (end_edge - start_edge) * 1000
    time_bf16_ms = (end_bf16 - start_bf16) * 1000
    speedup = time_edge_ms / time_bf16_ms
    print(f"  Edge版本: {time_edge_ms:.3f} ms")
    print(f"  BF16版本: {time_bf16_ms:.3f} ms")
    print(f"  速度比: {speedup:.2f}x {'(BF16更快)' if speedup > 1 else '(Edge更快)'}")
    
    # 判断是否通过
    print("\n" + "="*80)
    print("测试结论")
    print("="*80)
    
    tolerance = 1e-3
    o_passed = o_mean_err < tolerance
    h0_passed = h0_mean_err < tolerance
    
    print(f"\n输出 'O' 误差测试: {'✓ 通过' if o_passed else '✗ 未通过'} (阈值: {tolerance:.1e})")
    print(f"Final State 'h0' 误差测试: {'✓ 通过' if h0_passed else '✗ 未通过'} (阈值: {tolerance:.1e})")
    
    if o_passed and h0_passed:
        print("\n🎉 两个Kernel实现结果一致!")
    else:
        print("\n⚠️  两个Kernel实现存在差异，需要进一步检查")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    run_comparison()

