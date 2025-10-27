#!/usr/bin/env python3
"""
测试triton在sm_121映射到sm_80后是否能正常工作
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """简单的向量加法kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_simple_kernel():
    """测试简单的triton kernel"""
    print("=" * 60)
    print("测试 Triton Kernel: sm_121 -> sm_80 架构映射")
    print("=" * 60)
    
    # 检查设备信息
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    cap = torch.cuda.get_device_capability(0)
    print(f"\nGPU信息:")
    print(f"  设备名称: {torch.cuda.get_device_name(0)}")
    print(f"  原始 Compute Capability: {cap[0]}.{cap[1]} (sm_{cap[0]*10+cap[1]})")
    
    # 检查triton target
    from triton.runtime import driver as rt_driver
    driver = rt_driver.active
    target = driver.get_current_target()
    print(f"\nTriton Target:")
    print(f"  Backend: {target.backend}")
    print(f"  Architecture: sm_{target.arch} (映射后)")
    print(f"  Warp Size: {target.warp_size}")
    
    if cap[0]*10 + cap[1] == 121 and target.arch == 80:
        print(f"\n✅ 架构映射成功: sm_121 -> sm_80")
    
    # 测试kernel
    print(f"\n{'='*60}")
    print("运行 Triton Kernel 测试...")
    print(f"{'='*60}")
    
    try:
        size = 10240
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        output = torch.empty(size, device='cuda', dtype=torch.float32)
        
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
        
        # 验证结果
        expected = x + y
        max_error = (output - expected).abs().max().item()
        
        if torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
            print(f"\n✅ Kernel运行成功！")
            print(f"   测试大小: {size} 个元素")
            print(f"   最大误差: {max_error:.2e}")
            print(f"   Grid大小: {triton.cdiv(size, 256)} blocks")
            print(f"   Block大小: 256 threads")
            return True
        else:
            print(f"\n❌ 结果不匹配，最大误差: {max_error}")
            return False
            
    except Exception as e:
        print(f"\n❌ Kernel运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_kernel()
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！sm_121映射工作正常！")
    else:
        print("💔 测试失败")
    print("=" * 60)

