import torch
import matplotlib.pyplot as plt
import numpy as np

def check_nan(tensor, name):
    """检查tensor是否包含nan值"""
    has_nan = torch.isnan(tensor).any().item()
    if has_nan:
        nan_count = torch.isnan(tensor).sum().item()
        total_elements = tensor.numel()
        print(f"{name}: 包含 NaN! ({nan_count}/{total_elements} 个元素是NaN)")
    else:
        print(f"{name}: 无 NaN ✓")
    return has_nan

def check_nan_blocks(tensor, name, block_size=256):
    """检查nan是否集中在某些block中"""
    has_nan = torch.isnan(tensor).any().item()
    if not has_nan:
        print(f"{name}: 无 NaN，无需检查block分布")
        return
    
    # 将tensor展平并重塑为以block_size为单位的块
    flat_tensor = tensor.flatten()
    total_elements = flat_tensor.numel()
    
    # 计算完整的block数量
    num_full_blocks = total_elements // block_size
    remainder = total_elements % block_size
    
    print(f"\n{name} - Block分析 (block_size={block_size}):")
    print(f"  总元素数: {total_elements}")
    print(f"  完整block数: {num_full_blocks}")
    print(f"  剩余元素数: {remainder}")
    
    # 检查每个完整block
    blocks_with_nan = []
    blocks_all_nan = []
    nan_distribution = []
    
    for i in range(num_full_blocks):
        block = flat_tensor[i * block_size:(i + 1) * block_size]
        block_nan_count = torch.isnan(block).sum().item()
        
        if block_nan_count > 0:
            blocks_with_nan.append(i)
            nan_distribution.append(block_nan_count)
            if block_nan_count == block_size:
                blocks_all_nan.append(i)
    
    # 检查剩余元素
    remainder_nan_count = 0
    if remainder > 0:
        remainder_block = flat_tensor[num_full_blocks * block_size:]
        remainder_nan_count = torch.isnan(remainder_block).sum().item()
    
    # 输出结果
    print(f"  包含NaN的block数: {len(blocks_with_nan)}/{num_full_blocks}")
    print(f"  完全是NaN的block数: {len(blocks_all_nan)}")
    
    if len(blocks_with_nan) > 0:
        print(f"  包含NaN的block索引: {blocks_with_nan[:20]}" + 
              (f"... (共{len(blocks_with_nan)}个)" if len(blocks_with_nan) > 20 else ""))
        print(f"  这些block中NaN数量分布: min={min(nan_distribution)}, max={max(nan_distribution)}, avg={sum(nan_distribution)/len(nan_distribution):.1f}")
    
    if blocks_all_nan:
        print(f"  完全是NaN的block索引: {blocks_all_nan[:20]}" + 
              (f"... (共{len(blocks_all_nan)}个)" if len(blocks_all_nan) > 20 else ""))
    
    if remainder_nan_count > 0:
        print(f"  剩余元素中的NaN数: {remainder_nan_count}/{remainder}")
    
    # 计算nan的集中程度
    total_nan = torch.isnan(flat_tensor).sum().item()
    print(f"  总NaN数: {total_nan}")
    print(f"  NaN集中度: {len(blocks_with_nan) * block_size / total_elements * 100:.2f}% 的空间包含了所有NaN")

def analyze_nan_per_head(tensor, name):
    """Per head分析nan的分布"""
    has_nan = torch.isnan(tensor).any().item()
    if not has_nan:
        print(f"\n{name}: 无 NaN，无需per head分析")
        return None, None
    
    shape = tensor.shape
    print(f"\n{name} - Per Head 分析:")
    print(f"  Tensor shape: {shape}")
    
    # 假设shape为 (B, S, H, D) 或 (B, H, K, V)
    # 根据维度数判断格式
    if len(shape) == 4:
        B = shape[0]
        if shape[1] > shape[2]:  # 可能是 (B, S, H, D)
            S, H, D = shape[1], shape[2], shape[3]
            format_type = "v_new"
            print(f"  格式: (Batch={B}, Seq={S}, Heads={H}, Dim={D})")
        else:  # 可能是 (B, H, K, V)
            H, K, V = shape[1], shape[2], shape[3]
            format_type = "final_state"
            print(f"  格式: (Batch={B}, Heads={H}, K={K}, V={V})")
        
        # Per head统计
        nan_per_head = []
        total_per_head = []
        
        if format_type == "v_new":
            for b in range(B):
                for h in range(H):
                    head_tensor = tensor[b, :, h, :]
                    nan_count = torch.isnan(head_tensor).sum().item()
                    total_count = head_tensor.numel()
                    nan_per_head.append(nan_count)
                    total_per_head.append(total_count)
        else:  # final_state
            for b in range(B):
                for h in range(H):
                    head_tensor = tensor[b, h, :, :]
                    nan_count = torch.isnan(head_tensor).sum().item()
                    total_count = head_tensor.numel()
                    nan_per_head.append(nan_count)
                    total_per_head.append(total_count)
        
        nan_per_head = np.array(nan_per_head)
        heads_with_nan = np.where(nan_per_head > 0)[0]
        
        print(f"  包含NaN的head数: {len(heads_with_nan)}/{len(nan_per_head)}")
        if len(heads_with_nan) > 0:
            print(f"  包含NaN的head索引: {heads_with_nan.tolist()}")
            print(f"  每个head的NaN数量:")
            for idx in heads_with_nan[:20]:  # 只显示前20个
                print(f"    Head {idx}: {nan_per_head[idx]}/{total_per_head[idx]} 元素")
            if len(heads_with_nan) > 20:
                print(f"    ... (共{len(heads_with_nan)}个head包含NaN)")
        
        return nan_per_head, format_type
    else:
        print(f"  未知格式，维度数={len(shape)}")
        return None, None

def visualize_nan_distribution(tensor, name, nan_per_head=None):
    """可视化nan的分布"""
    has_nan = torch.isnan(tensor).any().item()
    if not has_nan:
        print(f"\n{name}: 无 NaN，无需可视化")
        return
    
    shape = tensor.shape
    
    if len(shape) == 4:
        B = shape[0]
        if shape[1] > shape[2]:  # v_new: (B, S, H, D)
            S, H, D = shape[1], shape[2], shape[3]
            
            # 创建热图显示nan的分布
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'{name} - NaN Distribution Analysis', fontsize=14)
            
            # 图1: Per head的nan数量
            if nan_per_head is not None:
                axes[0].bar(range(len(nan_per_head)), nan_per_head)
                axes[0].set_xlabel('Head Index')
                axes[0].set_ylabel('NaN Count')
                axes[0].set_title('NaN Count per Head')
                axes[0].grid(True, alpha=0.3)
            
            # 图2: 沿着序列维度的nan分布 (B=0)
            nan_mask = torch.isnan(tensor[0]).float()  # (S, H, D)
            nan_along_seq = nan_mask.sum(dim=2).cpu().numpy()  # (S, H)
            im2 = axes[1].imshow(nan_along_seq.T, aspect='auto', cmap='hot')
            axes[1].set_xlabel('Sequence Position')
            axes[1].set_ylabel('Head Index')
            axes[1].set_title('NaN Distribution (Seq vs Head)')
            plt.colorbar(im2, ax=axes[1], label='NaN count')
            
            # 图3: 每个位置的nan总数
            nan_per_position = nan_mask.sum(dim=(1, 2)).cpu().numpy()  # (S,)
            axes[2].plot(nan_per_position)
            axes[2].set_xlabel('Sequence Position')
            axes[2].set_ylabel('Total NaN Count')
            axes[2].set_title('NaN Count per Sequence Position')
            axes[2].grid(True, alpha=0.3)
            
        else:  # final_state: (B, H, K, V)
            H, K, V = shape[1], shape[2], shape[3]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'{name} - NaN Distribution Analysis', fontsize=14)
            
            # 图1: Per head的nan数量
            if nan_per_head is not None:
                axes[0].bar(range(len(nan_per_head)), nan_per_head)
                axes[0].set_xlabel('Head Index')
                axes[0].set_ylabel('NaN Count')
                axes[0].set_title('NaN Count per Head')
                axes[0].grid(True, alpha=0.3)
            
            # 图2: K-V维度的nan分布 (B=0, 所有head求和)
            nan_mask = torch.isnan(tensor[0]).float()  # (H, K, V)
            nan_kv = nan_mask.sum(dim=0).cpu().numpy()  # (K, V)
            im2 = axes[1].imshow(nan_kv, aspect='auto', cmap='hot')
            axes[1].set_xlabel('V dimension')
            axes[1].set_ylabel('K dimension')
            axes[1].set_title('NaN Distribution (K vs V, summed over heads)')
            plt.colorbar(im2, ax=axes[1], label='NaN count')
            
            # 图3: 每个head的nan分布热图
            nan_per_head_2d = []
            for h in range(min(H, 32)):  # 最多显示32个head
                head_nan = torch.isnan(tensor[0, h]).float().sum(dim=1).cpu().numpy()  # (K,)
                nan_per_head_2d.append(head_nan)
            
            if nan_per_head_2d:
                nan_per_head_2d = np.array(nan_per_head_2d)
                im3 = axes[2].imshow(nan_per_head_2d, aspect='auto', cmap='hot')
                axes[2].set_xlabel('K dimension')
                axes[2].set_ylabel('Head Index')
                axes[2].set_title(f'NaN Distribution (Head vs K, first {min(H, 32)} heads)')
                plt.colorbar(im3, ax=axes[2], label='NaN count along V')
        
        plt.tight_layout()
        save_path = f'{name}_nan_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化已保存到: {save_path}")
        plt.close()

# 直接加载三个输出
h = torch.load("./h.pt", map_location="cuda")
v_new = torch.load("./v_new.pt", map_location="cuda")
final_state = torch.load("./final_state.pt", map_location="cuda")

# 显示输出 shapes
print("输出数据 shapes:")
print(f"h shape: {h.shape}")
print(f"v_new shape: {v_new.shape}")
print(f"final_state shape: {final_state.shape}")

# 显示数据类型
print("\n" + "="*50 + "\n")
print("输出数据 dtypes:")
print(f"h dtype: {h.dtype}")
print(f"v_new dtype: {v_new.dtype}")
print(f"final_state dtype: {final_state.dtype}")

# 检查是否有nan
print("\n" + "="*50 + "\n")
print("NaN 检查:")
check_nan(h, "h")
check_nan(v_new, "v_new")
check_nan(final_state, "final_state")

# 检查nan在block中的分布
print("\n" + "="*50 + "\n")
print("NaN Block 分布分析:")
check_nan_blocks(h, "h", block_size=256)
check_nan_blocks(v_new, "v_new", block_size=256)
check_nan_blocks(final_state, "final_state", block_size=256)

# Per head分析
print("\n" + "="*50 + "\n")
print("Per Head NaN 分析:")
h_nan_per_head, h_format = analyze_nan_per_head(h, "h")
v_new_nan_per_head, v_new_format = analyze_nan_per_head(v_new, "v_new")
final_state_nan_per_head, final_state_format = analyze_nan_per_head(final_state, "final_state")

# 可视化nan分布
print("\n" + "="*50 + "\n")
print("生成可视化图表...")
visualize_nan_distribution(h, "h", h_nan_per_head)
visualize_nan_distribution(v_new, "v_new", v_new_nan_per_head)
visualize_nan_distribution(final_state, "final_state", final_state_nan_per_head)
print("\n所有分析完成!")
