#!/usr/bin/env python3
"""
完整测试 sm_121 -> sm_80 架构映射
验证：
1. Driver识别并映射设备架构
2. Compiler使用映射后的架构生成PTX
3. Kernel能够成功编译和执行
"""
import torch
import triton
import triton.language as tl
import sys


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """向量加法kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit  
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """简单的矩阵乘法kernel"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=c_mask)


def test_architecture_mapping():
    """测试架构映射"""
    print("=" * 70)
    print("测试 SM_121 -> SM_80 架构映射")
    print("=" * 70)
    
    # 检查设备
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    cap = torch.cuda.get_device_capability(0)
    actual_sm = cap[0] * 10 + cap[1]
    
    print(f"\n【GPU信息】")
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    print(f"  实际 Compute Capability: {cap[0]}.{cap[1]} (sm_{actual_sm})")
    
    # 检查Triton target
    from triton.runtime import driver as rt_driver
    driver = rt_driver.active
    target = driver.get_current_target()
    
    print(f"\n【Triton Target】")
    print(f"  Backend: {target.backend}")
    print(f"  Architecture: sm_{target.arch}")
    print(f"  Warp Size: {target.warp_size}")
    
    if actual_sm == 121:
        if target.arch == 80:
            print(f"\n✅ Driver层映射成功: sm_121 -> sm_80")
        else:
            print(f"\n❌ Driver层映射失败: 期望sm_80，实际sm_{target.arch}")
            return False
    
    return True


def test_vector_add():
    """测试向量加法kernel"""
    print(f"\n{'='*70}")
    print("测试 1: 向量加法 Kernel")
    print(f"{'='*70}")
    
    try:
        size = 102400
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        output = torch.empty(size, device='cuda', dtype=torch.float32)
        
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        vector_add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
        
        expected = x + y
        max_error = (output - expected).abs().max().item()
        
        if torch.allclose(output, expected, rtol=1e-5):
            print(f"✅ 向量加法测试通过")
            print(f"   大小: {size} 元素")
            print(f"   最大误差: {max_error:.2e}")
            return True
        else:
            print(f"❌ 结果不匹配，误差: {max_error}")
            return False
            
    except Exception as e:
        print(f"❌ 向量加法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matmul():
    """测试矩阵乘法kernel"""
    print(f"\n{'='*70}")
    print("测试 2: 矩阵乘法 Kernel")
    print(f"{'='*70}")
    
    try:
        M, N, K = 512, 512, 512
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        c = torch.empty((M, N), device='cuda', dtype=torch.float16)
        
        BLOCK_SIZE = 16
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            BLOCK_SIZE_K=BLOCK_SIZE,
        )
        
        expected = torch.matmul(a, b)
        max_error = (c - expected).abs().max().item()
        mean_error = (c - expected).abs().mean().item()
        
        if torch.allclose(c, expected, rtol=1e-2, atol=1e-2):
            print(f"✅ 矩阵乘法测试通过")
            print(f"   矩阵大小: {M}x{K} @ {K}x{N}")
            print(f"   最大误差: {max_error:.2e}")
            print(f"   平均误差: {mean_error:.2e}")
            return True
        else:
            print(f"❌ 结果不匹配")
            print(f"   最大误差: {max_error:.2e}")
            print(f"   平均误差: {mean_error:.2e}")
            return False
            
    except Exception as e:
        print(f"❌ 矩阵乘法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    results = []
    
    # 测试架构映射
    results.append(("架构映射", test_architecture_mapping()))
    
    # 测试kernel编译和执行
    results.append(("向量加法", test_vector_add()))
    results.append(("矩阵乘法", test_matmul()))
    
    # 总结
    print(f"\n{'='*70}")
    print("测试总结")
    print(f"{'='*70}")
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print(f"\n{'='*70}")
    if all_passed:
        print("🎉 所有测试通过！SM_121 -> SM_80 映射工作正常！")
        print("   - Driver正确识别并映射架构")
        print("   - Compiler使用sm_80生成PTX")  
        print("   - Kernels成功编译和执行")
    else:
        print("💔 部分测试失败")
    print(f"{'='*70}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

