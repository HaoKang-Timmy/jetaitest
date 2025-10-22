import torch
import sys
sys.path.append('/home/haokang/jetlmrelated/jetaitest/flash-linear-attention')
from fla.ops.utils import solve_tril
from tl_trilcompute import solve_tril_16x16

def test_solve_tril_comparison():
    """比较Triton和TileLang实现的solve_tril_16x16是否输出相同"""
    
    # 测试参数
    B = 2
    T = 64
    H = 4
    BT = 16  # 测试16x16的情况
    
    # 创建测试输入 - 严格下三角矩阵
    torch.manual_seed(42)
    A = torch.randn(B, T, H, BT, dtype=torch.float32, device='cuda')
    # 确保是严格下三角矩阵
    tril_mask = torch.tril(torch.ones(BT, BT, device='cuda'), diagonal=-1)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                A[b, t, h] = A[b, t, h] * tril_mask[t % BT]
    
    # 创建cu_seqlens (变长序列)
    cu_seqlens = torch.tensor([0, 32, 64], dtype=torch.int32, device='cuda')
    
    print("=" * 60)
    print("测试参数:")
    print(f"  Batch: {B}, Token: {T}, Head: {H}, Block: {BT}")
    print(f"  输入形状: {A.shape}")
    print(f"  cu_seqlens: {cu_seqlens.tolist()}")
    print("=" * 60)
    
    # 运行Triton版本
    print("\n运行 Triton solve_tril...")
    A_triton = A.clone()
    output_triton = solve_tril(A_triton, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
    print(f"  输出形状: {output_triton.shape}")
    
    # 运行TileLang版本
    print("\n运行 TileLang solve_tril_16x16...")
    A_tilelang = A.clone()
    output_tilelang = solve_tril_16x16(A_tilelang, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
    print(f"  输出形状: {output_tilelang.shape}")
    
    # 计算误差
    print("\n" + "=" * 60)
    print("误差分析:")
    print("=" * 60)
    
    # 绝对误差
    abs_diff = torch.abs(output_triton - output_tilelang)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    # 相对误差
    eps = 1e-8
    rel_diff = abs_diff / (torch.abs(output_triton) + eps)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"绝对误差:")
    print(f"  最大值: {max_abs_diff:.6e}")
    print(f"  平均值: {mean_abs_diff:.6e}")
    print(f"\n相对误差:")
    print(f"  最大值: {max_rel_diff:.6e}")
    print(f"  平均值: {mean_rel_diff:.6e}")
    
    # 检查是否通过
    threshold = 1e-4
    passed = max_rel_diff < threshold
    
    print("\n" + "=" * 60)
    if passed:
        print(f"✅ 测试通过! (最大相对误差 < {threshold})")
    else:
        print(f"❌ 测试失败! (最大相对误差 = {max_rel_diff:.6e} >= {threshold})")
    print("=" * 60)
    
    # 输出一些样本值对比
    print("\n样本值对比 (前5个元素):")
    print(f"  Triton:   {output_triton[0, 0, 0, :5].tolist()}")
    print(f"  TileLang: {output_tilelang[0, 0, 0, :5].tolist()}")
    
    return passed

if __name__ == "__main__":
    test_solve_tril_comparison()

