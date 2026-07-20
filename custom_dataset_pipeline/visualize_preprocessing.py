"""
可视化预处理结果的工具
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.patches as patches


def load_nifti(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载NIfTI文件
    
    Args:
        file_path: NIfTI文件路径
    
    Returns:
        (数据数组, affine矩阵)
    """
    img = nib.load(file_path)
    return img.get_fdata(), img.affine


def visualize_slice_comparison(
    original_slice: np.ndarray,
    corrected_slice: np.ndarray,
    normalized_slice: np.ndarray,
    label_slice: Optional[np.ndarray] = None,
    fat_mask: Optional[np.ndarray] = None,
    slice_index: int = 0,
    patient_id: str = 'unknown',
    output_path: Optional[str] = None
):
    """
    可视化单个切片的处理前后对比
    
    Args:
        original_slice: 原始图像切片
        corrected_slice: N4校正后的切片
        normalized_slice: 标准化后的切片
        label_slice: 分割标签切片
        fat_mask: 皮下脂肪掩码
        slice_index: 切片索引
        patient_id: 病人ID
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Patient: {patient_id}, Slice: {slice_index}', fontsize=16, y=1.02)
    
    vmin_orig = np.percentile(original_slice, 1)
    vmax_orig = np.percentile(original_slice, 99)
    
    im0 = axes[0, 0].imshow(original_slice, cmap='gray', vmin=vmin_orig, vmax=vmax_orig)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(corrected_slice, cmap='gray', vmin=vmin_orig, vmax=vmax_orig)
    axes[0, 1].set_title('N4 Bias Field Corrected')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(normalized_slice, cmap='gray', vmin=0, vmax=2)
    axes[0, 2].set_title('Normalized (Subcutaneous Fat)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    diff = corrected_slice - original_slice
    im3 = axes[1, 0].imshow(diff, cmap='coolwarm', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[1, 0].set_title('Correction Difference')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    if label_slice is not None:
        im4 = axes[1, 1].imshow(label_slice, cmap='tab20', vmin=0, vmax=6)
        axes[1, 1].set_title('Segmentation Label')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
    else:
        axes[1, 1].axis('off')
        axes[1, 1].set_title('No Label Available')
    
    if fat_mask is not None:
        y_size = fat_mask.shape[1]
        back_start = int(y_size * 0.8)
        rect = patches.Rectangle((0, back_start), fat_mask.shape[0], y_size - back_start,
                                  linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        
        im5 = axes[1, 2].imshow(original_slice, cmap='gray', vmin=vmin_orig, vmax=vmax_orig)
        axes[1, 2].imshow(fat_mask, cmap='Reds', alpha=0.5)
        axes[1, 2].add_patch(rect)
        axes[1, 2].set_title('Subcutaneous Fat (Overlay)')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')
        axes[1, 2].set_title('No Fat Mask')
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to: {output_path}')
    
    plt.show()


def visualize_single_patient(
    patient_dir: str,
    original_image_path: str,
    label_path: Optional[str] = None,
    slice_index: Optional[int] = None,
    output_dir: Optional[str] = None
):
    """
    可视化单个病人的预处理结果
    
    Args:
        patient_dir: 病人预处理结果目录
        original_image_path: 原始图像路径
        label_path: 分割标签路径
        slice_index: 要查看的切片索引，如果为None则查看中间切片
        output_dir: 输出目录
    """
    patient_path = Path(patient_dir)
    patient_id = patient_path.name
    
    corrected_path = patient_path / f'{patient_id}_corrected.nii.gz'
    normalized_path = patient_path / f'{patient_id}_normalized.nii.gz'
    
    if not corrected_path.exists() or not normalized_path.exists():
        print(f'预处理结果文件不存在于: {patient_dir}')
        return
    
    original_data, _ = load_nifti(original_image_path)
    corrected_data, _ = load_nifti(str(corrected_path))
    normalized_data, _ = load_nifti(str(normalized_path))
    
    label_data = None
    if label_path is not None and Path(label_path).exists():
        label_data, _ = load_nifti(label_path)
    
    if slice_index is None:
        slice_index = original_data.shape[2] // 2
    
    if slice_index < 0 or slice_index >= original_data.shape[2]:
        print(f'切片索引 {slice_index} 超出范围 [0, {original_data.shape[2] - 1}]')
        return
    
    original_slice = original_data[:, :, slice_index]
    corrected_slice = corrected_data[:, :, slice_index]
    normalized_slice = normalized_data[:, :, slice_index]
    
    label_slice = None
    if label_data is not None:
        label_slice = label_data[:, :, slice_index]
    
    from image_preprocessing import extract_subcutaneous_fat
    fat_mean, fat_mask, _ = extract_subcutaneous_fat(
        corrected_slice,
        percentile_threshold=95.0,
        back_ratio=0.2,
        min_area_ratio=0.01,
        max_area_ratio=0.3
    )
    
    output_path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = str(Path(output_dir) / f'{patient_id}_slice{slice_index:04d}.png')
    
    visualize_slice_comparison(
        original_slice,
        corrected_slice,
        normalized_slice,
        label_slice,
        fat_mask,
        slice_index,
        patient_id,
        output_path
    )


def plot_fat_signal_statistics(slice_csv_path: str, output_dir: Optional[str] = None):
    """
    绘制脂肪信号统计图表
    
    Args:
        slice_csv_path: 切片统计CSV文件路径
        output_dir: 输出目录
    """
    import pandas as pd
    
    df = pd.read_csv(slice_csv_path)
    
    valid_df = df[df['status'] == 'valid']
    
    if len(valid_df) == 0:
        print('没有有效的切片数据')
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].hist(valid_df['mean_fat_signal'], bins=30, edgecolor='black')
    axes[0, 0].set_title('Distribution of Mean Fat Signal')
    axes[0, 0].set_xlabel('Mean Fat Signal')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(valid_df['fat_area_ratio'], bins=30, edgecolor='black')
    axes[0, 1].set_title('Distribution of Fat Area Ratio')
    axes[0, 1].set_xlabel('Fat Area Ratio')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    status_counts = df['status'].value_counts()
    axes[1, 0].bar(status_counts.index, status_counts.values)
    axes[1, 0].set_title('Slice Status Distribution')
    axes[1, 0].set_xlabel('Status')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    if 'muscle_csa' in valid_df.columns and 'fat_ratio' in valid_df.columns:
        axes[1, 1].scatter(valid_df['muscle_csa'], valid_df['fat_ratio'], alpha=0.6)
        axes[1, 1].set_xlabel('Muscle CSA (mm²)')
        axes[1, 1].set_ylabel('Fat Ratio')
        axes[1, 1].set_title('Muscle CSA vs Fat Ratio')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = str(Path(output_dir) / 'fat_signal_statistics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved statistics plot to: {output_path}')
    
    plt.show()


if __name__ == '__main__':
    print('=' * 80)
    print('预处理结果可视化工具')
    print('=' * 80)
    
    print('\n使用说明:')
    print('''
    1. visualize_single_patient: 可视化单个病人的预处理结果
    2. plot_fat_signal_statistics: 绘制脂肪信号统计图表
    
    示例:
    ''')
    
    print('''
    # 示例1: 可视化单个病人
    visualize_single_patient(
        patient_dir="path/to/output/patient_id",
        original_image_path="path/to/original/image.nii.gz",
        label_path="path/to/label.nii.gz",
        slice_index=10,
        output_dir="path/to/visualizations"
    )
    ''')
    
    print('''
    # 示例2: 绘制统计图表
    plot_fat_signal_statistics(
        slice_csv_path="path/to/slice_statistics.csv",
        output_dir="path/to/visualizations"
    )
    ''')
