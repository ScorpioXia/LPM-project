import os
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import Counter


def analyze_labels(labels_dir, max_files=10):
    """
    分析标注文件，查看标签值
    """
    labels_path = Path(labels_dir)
    label_files = sorted(list(labels_path.glob("*.nii.gz")) + list(labels_path.glob("*.nii")))
    
    if not label_files:
        print("未找到标注文件!")
        return None
    
    print(f"找到 {len(label_files)} 个标注文件")
    print(f"将分析前 {min(max_files, len(label_files))} 个文件\n")
    
    all_unique_labels = set()
    sample_info = []
    
    for i, label_file in enumerate(label_files[:max_files]):
        try:
            lbl = nib.load(str(label_file))
            lbl_data = lbl.get_fdata()
            
            # 转换为整数标签
            lbl_data_int = lbl_data.astype(np.int32)
            unique_labels = np.unique(lbl_data_int)
            
            all_unique_labels.update(unique_labels)
            
            if i < 5:
                sample_info.append({
                    "file": label_file.name,
                    "unique_labels": sorted(unique_labels),
                    "shape": lbl_data.shape
                })
                print(f"{label_file.name}:")
                print(f"  标签值: {sorted(unique_labels)}")
                print(f"  形状: {lbl_data.shape}")
                
        except Exception as e:
            print(f"警告: 无法分析 {label_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"所有文件中发现的标签值: {sorted(all_unique_labels)}")
    print(f"{'='*60}")
    
    return {
        "all_unique_labels": sorted(all_unique_labels),
        "sample_info": sample_info
    }


def create_label_mapping(current_labels, target_labels=[0, 1, 2, 3, 4, 5, 6]):
    """
    创建标签映射配置
    """
    print(f"\n当前标签: {sorted(current_labels)}")
    print(f"目标标签: {target_labels}")
    
    if set(current_labels) == set(target_labels):
        print("\n✓ 标签已经正确，无需转换!")
        return None
    
    print("\n请根据需要创建标签映射")
    print("示例:")
    print("  如果当前标签是 [0, 2, 3, 4, 5, 6, 7]，想要映射到 [0, 1, 2, 3, 4, 5, 6]")
    print("  映射字典: {0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}")
    
    return None


def convert_labels(labels_dir, output_dir, label_mapping=None):
    """
    转换标签到目标空间
    
    Args:
        labels_dir: 输入标签目录
        output_dir: 输出目录
        label_mapping: 标签映射字典 {旧标签: 新标签}
    """
    if label_mapping is None:
        print("未提供标签映射，将直接复制文件")
        label_mapping = {x: x for x in range(10)}
    
    input_path = Path(labels_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    label_files = sorted(list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii")))
    
    print(f"\n找到 {len(label_files)} 个标注文件")
    print(f"标签映射: {label_mapping}")
    
    success_count = 0
    
    for label_file in label_files:
        print(f"\n处理: {label_file.name}")
        
        try:
            img = nib.load(str(label_file))
            data = img.get_fdata()
            affine = img.affine
            header = img.header
            
            new_data = np.zeros_like(data)
            
            for old_label, new_label in label_mapping.items():
                mask = data == old_label
                new_data[mask] = new_label
            
            output_file = output_path / label_file.name
            new_img = nib.Nifti1Image(new_data.astype(np.int32), affine, header)
            nib.save(new_img, str(output_file))
            
            print(f"  ✓ 已保存: {output_file.name}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print(f"\n{'='*60}")
    print(f"转换完成! 成功: {success_count}/{len(label_files)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("="*60)
    print("步骤2: 标签检查和转换")
    print("="*60)
    
    print("\n使用方法:")
    print("1. 先运行分析模式查看标签")
    print("2. 根据需要创建标签映射")
    print("3. 运行转换模式")
    
    print("\n请修改以下配置后运行:")
    print("""
    # 模式1: 只分析标签
    LABELS_DIR = "path/to/your/labels"
    analyze_labels(LABELS_DIR, max_files=10)
    
    # 模式2: 转换标签（如果需要）
    # LABEL_MAPPING = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
    # OUTPUT_DIR = "path/to/converted/labels"
    # convert_labels(LABELS_DIR, OUTPUT_DIR, LABEL_MAPPING)
    """)
