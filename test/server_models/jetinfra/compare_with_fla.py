import torch
import sys

# 添加 flash-linear-attention 库路径
sys.path.insert(0, "/home/haokang/jetlmrelated/jetaitest/flash-linear-attention")

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from tl_chunked_deltarule import tilelang_chunk_scaled_matmul_fwd, tilelang_chunk_scaled_dot_kkt_fwd

def compare_fla_vs_tilelang():
    # 设置测试参数
    B, S, H, DK = 1, 2000, 12, 96
    chunk_size = 64
    
    # 创建相同的输入数据
    torch.manual_seed(42)
    k = torch.randn(B, S, H, DK, dtype=torch.bfloat16).cuda()
    beta = torch.randn(B, S, H, dtype=torch.bfloat16).cuda()
    g_cumsum = torch.randn(B, S, H, dtype=torch.float32).cuda()
    
    print("=" * 70)
    print("对比 Flash-Linear-Attention 和 TileLang 实现")
    print("=" * 70)
    print(f"\n测试参数: B={B}, S={S}, H={H}, DK={DK}, chunk_size={chunk_size}")
    print(f"\n输入形状:")
    print(f"  K: {k.shape}, dtype: {k.dtype}")
    print(f"  Beta: {beta.shape}, dtype: {beta.dtype}")
    print(f"  G_cumsum: {g_cumsum.shape}, dtype: {g_cumsum.dtype}")
    
    # 测试 Flash-Linear-Attention 版本
    print("\n" + "-" * 70)
    print("运行 Flash-Linear-Attention: chunk_scaled_dot_kkt_fwd")
    print("-" * 70)
    output_fla = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=None,
        chunk_size=chunk_size,
        output_dtype=torch.float32
    )
    print(f"输出形状: {output_fla.shape}, dtype: {output_fla.dtype}")
    print(f"输出统计:")
    print(f"  min={output_fla.min().item():.6f}")
    print(f"  max={output_fla.max().item():.6f}")
    print(f"  mean={output_fla.mean().item():.6f}")
    print(f"  std={output_fla.std().item():.6f}")
    
    # 测试 TileLang 版本
    print("\n" + "-" * 70)
    print("运行 TileLang: tilelang_chunk_scaled_matmul_fwd")
    print("-" * 70)
    kernel_tl = tilelang_chunk_scaled_dot_kkt_fwd(
        B, S, H, DK, 
        chunk_size=chunk_size,
        use_g=True,
        output_dtype="float32"
    )
    output_tl = kernel_tl(k, beta, g_cumsum)
    print(f"输出形状: {output_tl.shape}, dtype: {output_tl.dtype}")
    print(f"输出统计:")
    print(f"  min={output_tl.min().item():.6f}")
    print(f"  max={output_tl.max().item():.6f}")
    print(f"  mean={output_tl.mean().item():.6f}")
    print(f"  std={output_tl.std().item():.6f}")
    
    # 比较结果
    print("\n" + "=" * 70)
    print("结果比较")
    print("=" * 70)
    
    # 计算差异
    diff = torch.abs(output_fla - output_tl)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 计算相对误差
    max_val = max(output_fla.abs().max().item(), output_tl.abs().max().item())
    relative_error = max_diff / (max_val + 1e-8)
    
    # 尝试不同的容差
    is_close_strict = torch.allclose(output_fla, output_tl, rtol=1e-5, atol=1e-7)
    is_close_normal = torch.allclose(output_fla, output_tl, rtol=1e-3, atol=1e-5)
    is_close_relaxed = torch.allclose(output_fla, output_tl, rtol=1e-2, atol=1e-4)
    
    print(f"\n差异统计:")
    print(f"  最大绝对差异: {max_diff:.6e}")
    print(f"  平均绝对差异: {mean_diff:.6e}")
    print(f"  最大相对误差: {relative_error:.6e}")
    
    print(f"\n是否近似相等:")
    print(f"  严格容差 (rtol=1e-5, atol=1e-7): {is_close_strict}")
    print(f"  正常容差 (rtol=1e-3, atol=1e-5): {is_close_normal}")
    print(f"  宽松容差 (rtol=1e-2, atol=1e-4): {is_close_relaxed}")
    
    # 显示样本值对比
    print(f"\n样本值对比 (第一个chunk，前8个元素):")
    print(f"  batch=0, seq=0, head=0, chunk_pos=0:7")
    print(f"  FLA 输出:      {output_fla[0, 0, 0, :8]}")
    print(f"  TileLang 输出: {output_tl[0, 0, 0, :8]}")
    print(f"  差异:          {diff[0, 0, 0, :8]}")
    
    # 检查一些不同位置的值
    print(f"\n  batch=0, seq=64, head=0, chunk_pos=0:8")
    print(f"  FLA 输出:      {output_fla[0, 64, 0, :8]}")
    print(f"  TileLang 输出: {output_tl[0, 64, 0, :8]}")
    print(f"  差异:          {diff[0, 64, 0, :8]}")
    
    # 结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if is_close_normal:
        print("✓ Flash-Linear-Attention 和 TileLang 的输出结果相同")
        print("  （在正常数值容差范围内）")
    elif is_close_relaxed:
        print("⚠ Flash-Linear-Attention 和 TileLang 的输出结果基本相同")
        print("  （在宽松容差范围内，存在小的数值误差）")
    else:
        print("✗ Flash-Linear-Attention 和 TileLang 的输出结果存在显著差异")
        print(f"\n可能的原因:")
        print(f"  1. 实现细节差异（Triton vs TileLang）")
        print(f"  2. 浮点运算顺序不同导致的累积误差")
        print(f"  3. 内存布局和访问模式差异")
        print(f"  4. 编译器优化策略不同")
    
    print("\n" + "=" * 70)
    
    return output_fla, output_tl, diff

if __name__ == "__main__":
    compare_fla_vs_tilelang()
