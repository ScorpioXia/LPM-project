"""
图像预处理脚本
根据研究路径文档1.2节实现：
1. N4偏置场校正
2. 同层动态标准化（提取背部皮下脂肪信号）
3. 异常值筛查
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List
from sklearn.mixture import GaussianMixture
from scipy import ndimage


def get_global_fat_signal(image_3d):
    """
    使用高斯混合模型从整个3D图像中提取脂肪信号
    
    Args:
        image_3d: 3D图像数组，形状为(x, y, z)
    
    Returns:
        脂肪信号平均值
    """
    # 将三维图像展平，采样（可随机采样以减少计算量）
    all_pixels = image_3d.flatten()
    # 只取一定范围的像素，剔除背景（例如取>1%分位数）
    low_cut = np.percentile(all_pixels, 5)
    high_cut = np.percentile(all_pixels, 99)
    filtered = all_pixels[(all_pixels > low_cut) & (all_pixels < high_cut)]
    
    # 拟合3个高斯分量
    data = filtered.reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(data)
    means = gmm.means_.flatten()
    # 取均值最大的分量作为脂肪
    fat_mean = np.max(means)
    return fat_mean


def n4_bias_field_correction(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    shrink_factor: int = 4,
    convergence_threshold: float = 1e-6,
    max_iterations: Tuple[int, int, int, int] = (50, 50, 50, 50)
) -> np.ndarray:
    """
    使用SimpleITK进行N4偏置场校正（安全版本）
    
    Args:
        image: 输入图像数组，形状为(x, y, z)
        mask: 可选的掩码数组
        shrink_factor: 缩小因子，加速计算（使用内置加速）
        convergence_threshold: 收敛阈值
        max_iterations: 每一层的最大迭代次数
    
    Returns:
        校正后的图像数组
    """
    try:
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing([1.0, 1.0, 1.0])
        
        if mask is None:
            mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        else:
            mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_image.SetSpacing([1.0, 1.0, 1.0])
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetConvergenceThreshold(convergence_threshold)
        corrector.SetMaximumNumberOfIterations(max_iterations)
        
        corrected_image = corrector.Execute(sitk_image, mask_image)
        
        return sitk.GetArrayFromImage(corrected_image)
    except Exception as e:
        print(f'  N4校正失败，跳过校正: {e}')
        return image


def extract_subcutaneous_fat(
    image_slice: np.ndarray,
    percentile_threshold: float = 80.0,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.8,
    debug: bool = False,
    debug_save_path: Optional[str] = None,
    slice_index: Optional[int] = None
) -> Tuple[Optional[float], Optional[np.ndarray], Dict]:
    """
    从图像切片中提取全图脂肪区域并计算其平均信号值
    
    Args:
        image_slice: 单张图像切片 (x, y)
        percentile_threshold: 用于筛选脂肪的百分位数阈值
        min_area_ratio: 脂肪区域最小面积比例
        max_area_ratio: 脂肪区域最大面积比例
        debug: 是否启用调试模式
        debug_save_path: 调试图像保存路径
        slice_index: 切片索引（用于调试文件名）
    
    Returns:
        tuple: (脂肪平均信号值, 脂肪掩码, 统计信息字典)
    """
    stats = {}
    stats['total_pixels'] = image_slice.size
    
    # 使用全图策略：直接基于信号强度筛选脂肪
    threshold = np.percentile(image_slice, percentile_threshold)
    stats['percentile_threshold'] = percentile_threshold
    stats['signal_threshold'] = threshold
    
    # 全图筛选高信号区域作为脂肪
    subcutaneous_fat = image_slice > threshold
    stats['subcutaneous_fat_pixels'] = np.sum(subcutaneous_fat)
    
    fat_area_ratio = np.sum(subcutaneous_fat) / image_slice.size
    stats['fat_area_ratio'] = fat_area_ratio
    stats['back_region'] = 'whole_image'  # 全图策略
    
    # 调试模式：保存可视化图像（移除转置，修复方向问题）
    if debug and debug_save_path is not None:
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path
            
            save_dir = Path(debug_save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(np.rot90(image_slice, k=1), cmap='gray')  # 逆时针旋转90度并使用灰度显示
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(np.rot90(subcutaneous_fat, k=1), cmap='hot')  # 逆时针旋转90度并使用热图显示
            plt.title(f'Fat Candidates (>{percentile_threshold}%)')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(np.rot90(subcutaneous_fat, k=1), cmap='hot')  # 逆时针旋转90度并使用热图显示
            plt.title(f'Extracted Fat (area={fat_area_ratio:.3f})')
            plt.axis('off')
            
            plt.tight_layout()
            
            if slice_index is not None:
                plt.savefig(save_dir / f'debug_fat_slice_{slice_index:03d}.png', dpi=150, bbox_inches='tight')
            else:
                plt.savefig(save_dir / f'debug_fat_slice.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'  调试图像保存失败: {e}')
    
    if min_area_ratio <= fat_area_ratio <= max_area_ratio:
        fat_signal_values = image_slice[subcutaneous_fat]
        mean_signal = np.mean(fat_signal_values)
        median_signal = np.median(fat_signal_values)
        
        stats['mean_fat_signal'] = mean_signal
        stats['median_fat_signal'] = median_signal
        stats['quality_check'] = 'pass'
        
        return mean_signal, subcutaneous_fat, stats
    else:
        stats['quality_check'] = 'fail'
        stats['reason'] = f'Fat area ratio {fat_area_ratio:.4f} outside range [{min_area_ratio}, {max_area_ratio}]'
        return None, None, stats


def extract_pure_dorsal_subcutaneous_fat(
    image_slice: np.ndarray,
    high_percentile: float = 99.5,
    min_area: int = 100,
    dorsal_ratio: float = 0.6,
    edge_margin: int = 5,
    debug: bool = False,
    debug_save_path: Optional[str] = None,
    slice_index: Optional[int] = None
) -> Tuple[Optional[float], Optional[np.ndarray], Dict]:
    """
    全自动提取纯净背部皮下脂肪区域，用于后续标准化。
    算法：极高阈值→选背部区域→最大连通域→再腐蚀净化→取均值。

    Args:
        image_slice: 单张切片 (H, W)
        high_percentile: 用于抓取绝对高信号区域的百分位数（默认99.5）
        min_area: 连通域最小面积（像素）
        dorsal_ratio: 背部区域在Y轴方向的比例，如0.6表示只考虑后60%的部位
        edge_margin: 边缘剔除宽度（像素），避免紧贴边界的伪影
        debug: 是否启用调试模式
        debug_save_path: 调试图像保存路径
        slice_index: 切片索引（用于调试文件名）

    Returns:
        (脂肪平均信号, 纯净脂肪二值掩膜, 统计信息字典)
    """
    stats = {'method': 'pure_dorsal_fat'}
    H, W = image_slice.shape

    threshold = np.percentile(image_slice, high_percentile)
    stats['high_signal_threshold'] = threshold

    high_signal_mask = image_slice > threshold
    if not np.any(high_signal_mask):
        stats['error'] = 'No high signal pixels'
        return None, None, stats

    dorsal_start = int(H * (1 - dorsal_ratio))
    dorsal_mask = np.zeros_like(high_signal_mask)
    dorsal_mask[dorsal_start:, :] = True
    dorsal_mask[:edge_margin, :] = False
    dorsal_mask[-edge_margin:, :] = False
    dorsal_mask[:, :edge_margin] = False
    dorsal_mask[:, -edge_margin:] = False

    candidate_mask = high_signal_mask & dorsal_mask
    if not np.any(candidate_mask):
        stats['error'] = 'No high signal in dorsal region'
        return None, None, stats

    labeled, num_features = ndimage.label(candidate_mask)
    if num_features == 0:
        stats['error'] = 'No connected components'
        return None, None, stats

    areas = ndimage.sum(np.ones_like(candidate_mask), labeled, index=range(1, num_features+1))
    if len(areas) == 0:
        stats['error'] = 'No regions with sufficient area'
        return None, None, stats

    largest_label = np.argmax(areas) + 1
    largest_area = areas[largest_label - 1]
    if largest_area < min_area:
        stats['error'] = f'Largest region area {largest_area} < {min_area}'
        return None, None, stats

    dorsal_fat_mask = labeled == largest_label

    structure = np.ones((3, 3))
    pure_fat_mask = ndimage.binary_erosion(dorsal_fat_mask, structure, iterations=2).astype(bool)
    if not np.any(pure_fat_mask):
        pure_fat_mask = dorsal_fat_mask

    fat_signal_values = image_slice[pure_fat_mask]
    mean_signal = np.mean(fat_signal_values)
    stats['fat_region_pixels'] = np.sum(pure_fat_mask)
    stats['fat_region_area_ratio'] = np.sum(pure_fat_mask) / image_slice.size
    stats['quality_check'] = 'pass'
    stats['mean_fat_signal'] = mean_signal

    if debug and debug_save_path is not None:
        try:
            import matplotlib.pyplot as plt
            save_dir = Path(debug_save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(np.rot90(image_slice, k=1), cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(np.rot90(candidate_mask, k=1), cmap='hot')
            plt.title('Dorsal Fat Candidates')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(np.rot90(pure_fat_mask, k=1), cmap='hot')
            plt.title('Pure Fat Mask')
            plt.axis('off')

            plt.tight_layout()

            if slice_index is not None:
                plt.savefig(save_dir / f'debug_fat_slice_{slice_index:03d}.png', dpi=150, bbox_inches='tight')
            else:
                plt.savefig(save_dir / f'debug_fat_slice.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'  调试图像保存失败: {e}')

    return mean_signal, pure_fat_mask, stats


def normalize_image_slice(
    image_slice: np.ndarray,
    fat_mean_signal: float
) -> np.ndarray:
    """
    使用皮下脂肪平均信号值标准化图像切片
    
    Args:
        image_slice: 原始图像切片
        fat_mean_signal: 皮下脂肪平均信号值
    
    Returns:
        标准化后的图像切片
    """
    if fat_mean_signal <= 0:
        return image_slice
    
    normalized = image_slice / fat_mean_signal
    return normalized


def calculate_muscle_csa(
    label_slice: np.ndarray,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
) -> float:
    """
    计算肌肉横截面积 (CSA)
    
    Args:
        label_slice: 分割标签切片 (0为背景，非0为肌肉)
        pixel_spacing: 像素物理间距 (x, y) mm
    
    Returns:
        肌肉横截面积 (mm²)
    """
    muscle_mask = label_slice > 0
    pixel_area = pixel_spacing[0] * pixel_spacing[1]
    csa = np.sum(muscle_mask) * pixel_area
    return csa


def calculate_fat_ratio(
    normalized_slice: np.ndarray,
    label_slice: np.ndarray,
    muscle_threshold: float = 0.5
) -> float:
    """
    计算肌肉区域内的脂肪比例
    
    Args:
        normalized_slice: 标准化后的图像切片
        label_slice: 分割标签切片
        muscle_threshold: 肌肉阈值 (标准化信号值 < threshold视为肌肉)
    
    Returns:
        脂肪比例
    """
    muscle_mask = label_slice > 0
    muscle_pixels = normalized_slice[muscle_mask]
    
    if len(muscle_pixels) == 0:
        return 0.0
    
    fat_pixels = muscle_pixels > muscle_threshold
    fat_ratio = np.sum(fat_pixels) / len(muscle_pixels)
    
    return fat_ratio


def check_abnormal_slice(
    label_slice: np.ndarray,
    normalized_slice: np.ndarray,
    min_csa: float = 10.0,
    max_fat_ratio: float = 0.95,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
) -> Tuple[bool, str]:
    """
    检查切片是否异常
    
    Args:
        label_slice: 分割标签切片
        normalized_slice: 标准化后的图像切片
        min_csa: 最小肌肉CSA (mm²)
        max_fat_ratio: 最大脂肪比例
        pixel_spacing: 像素物理间距
    
    Returns:
        (是否异常, 异常信息)
    """
    csa = calculate_muscle_csa(label_slice, pixel_spacing)
    if csa < min_csa:
        return True, f'Muscle CSA {csa:.2f} mm² < {min_csa} mm²'
    
    fat_ratio = calculate_fat_ratio(normalized_slice, label_slice)
    if fat_ratio > max_fat_ratio:
        return True, f'Fat ratio {fat_ratio:.4f} > {max_fat_ratio}'
    
    return False, 'Normal'


def process_single_patient(
    image_path: str,
    label_path: str,
    patient_id: str,
    output_dir: str,
    pixel_spacing: Tuple[float, float] = None,
    n4_shrink_factor: int = 4,
    fat_percentile_threshold: float = 80.0,
    min_fat_area_ratio: float = 0.001,
    max_fat_area_ratio: float = 0.8,
    min_csa: float = 10.0,
    max_fat_ratio: float = 0.95,
    use_first_slice_fat: bool = True,
    first_slice_index: int = 0,
    use_gmm_fat: bool = False,
    use_pure_dorsal_fat: bool = True,
    pure_dorsal_params: Dict = None,
    debug: bool = False
) -> Tuple[Dict, List[Dict]]:
    """
    处理单个病人的图像和分割标签

    Args:
        image_path: 原始图像路径
        label_path: 分割标签路径
        patient_id: 病人ID
        output_dir: 输出目录
        pixel_spacing: 像素物理间距（如果为None，则从图像头文件自动读取）
        n4_shrink_factor: N4校正缩小因子
        fat_percentile_threshold: 脂肪百分位数阈值（默认80）
        min_fat_area_ratio: 最小脂肪面积比例（默认0.001）
        max_fat_area_ratio: 最大脂肪面积比例（默认0.8）
        min_csa: 最小肌肉CSA
        max_fat_ratio: 最大脂肪比例
        use_first_slice_fat: 是否使用指定切片的脂肪信号进行所有切片的标准化
        first_slice_index: 指定的切片索引（默认0，即第1张）
        use_gmm_fat: 是否使用高斯混合模型从全图提取脂肪信号
        use_pure_dorsal_fat: 是否使用纯净背部皮下脂肪提取方法（默认True）
        pure_dorsal_params: 纯净脂肪提取的参数字典
        debug: 是否启用调试模式

    Returns:
        (病人总体统计信息, 各切片统计信息列表)
    """
    if pure_dorsal_params is None:
        pure_dorsal_params = {
            'high_percentile': 99.5,
            'min_area': 100,
            'dorsal_ratio': 0.6,
            'edge_margin': 5
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    img = nib.load(image_path)
    image_data = img.get_fdata()
    affine = img.affine
    
    # 从nifti头文件读取真实像素间距
    if pixel_spacing is None:
        zooms = img.header.get_zooms()
        pixel_spacing = (zooms[0], zooms[1])
        print(f'  自动读取像素间距: {pixel_spacing} mm')
    
    lbl = nib.load(label_path)
    label_data = lbl.get_fdata().astype(np.int32)
    
    assert image_data.shape == label_data.shape, f'Image shape {image_data.shape} != label shape {label_data.shape}'
    
    print(f'Processing patient {patient_id}, shape: {image_data.shape}')
    
    corrected_image = n4_bias_field_correction(
        image_data,
        shrink_factor=n4_shrink_factor
    )
    
    normalized_image = np.zeros_like(corrected_image)
    
    patient_stats = {
        'patient_id': patient_id,
        'total_slices': image_data.shape[2],
        'valid_slices': 0,
        'invalid_slices': 0,
        'fat_signal_values': []
    }
    
    slice_stats_list = []
    
    # 调试模式：创建调试图像保存目录
    debug_save_path = None
    if debug:
        debug_save_path = output_path / 'debug_images' / patient_id
    
    # 首先提取脂肪信号
    global_fat_mean = None
    
    if use_pure_dorsal_fat:
        print(f'  使用纯净背部皮下脂肪提取方法...')
        if first_slice_index < image_data.shape[2]:
            first_slice = corrected_image[:, :, first_slice_index]
            fat_mean, _, _ = extract_pure_dorsal_subcutaneous_fat(
                first_slice,
                high_percentile=pure_dorsal_params.get('high_percentile', 99.5),
                min_area=pure_dorsal_params.get('min_area', 100),
                dorsal_ratio=pure_dorsal_params.get('dorsal_ratio', 0.6),
                edge_margin=pure_dorsal_params.get('edge_margin', 5),
                debug=debug,
                debug_save_path=debug_save_path,
                slice_index=first_slice_index
            )
            if fat_mean is not None:
                global_fat_mean = fat_mean
                print(f'  纯净背部脂肪信号: {global_fat_mean:.2f}')
            else:
                print(f'  警告: 纯净脂肪提取失败，将尝试备选方法')
        else:
            print(f'  警告: 指定切片索引 {first_slice_index} 超出范围')
    
    if use_gmm_fat and global_fat_mean is None:
        print(f'  使用高斯混合模型从全图提取脂肪信号...')
        try:
            global_fat_mean = get_global_fat_signal(corrected_image)
            print(f'  GMM全局脂肪信号: {global_fat_mean:.2f}')
        except Exception as e:
            print(f'  GMM脂肪提取失败: {e}')
            use_gmm_fat = False
    
    if global_fat_mean is None and use_first_slice_fat:
        if first_slice_index < image_data.shape[2]:
            print(f'  从第 {first_slice_index} 张切片提取脂肪信号（备选方法）...')
            first_slice = corrected_image[:, :, first_slice_index]
            fat_mean, _, _ = extract_subcutaneous_fat(
                first_slice,
                percentile_threshold=fat_percentile_threshold,
                min_area_ratio=min_fat_area_ratio,
                max_area_ratio=max_fat_area_ratio,
                debug=debug,
                debug_save_path=debug_save_path,
                slice_index=first_slice_index
            )
            if fat_mean is not None:
                global_fat_mean = fat_mean
                print(f'  指定切片脂肪信号: {global_fat_mean:.2f}')
            else:
                print(f'  警告: 指定切片脂肪提取失败，将使用逐切片提取')
        else:
            print(f'  警告: 指定切片索引 {first_slice_index} 超出范围，将使用逐切片提取')
    
    # 记录成功提取的脂肪信号（用于备用）
    success_fat_signals = []
    
    for z in tqdm(range(image_data.shape[2]), desc=f'Patient {patient_id}'):
        img_slice = corrected_image[:, :, z]
        lbl_slice = label_data[:, :, z]
        
        slice_stats = {
            'patient_id': patient_id,
            'slice_index': z,
            'status': 'unknown'
        }
        
        # 提取当前切片的脂肪信号
        if use_pure_dorsal_fat:
            fat_mean, fat_mask, fat_stats = extract_pure_dorsal_subcutaneous_fat(
                img_slice,
                high_percentile=pure_dorsal_params.get('high_percentile', 99.5),
                min_area=pure_dorsal_params.get('min_area', 100),
                dorsal_ratio=pure_dorsal_params.get('dorsal_ratio', 0.6),
                edge_margin=pure_dorsal_params.get('edge_margin', 5),
                debug=debug,
                debug_save_path=debug_save_path,
                slice_index=z
            )
        else:
            fat_mean, fat_mask, fat_stats = extract_subcutaneous_fat(
                img_slice,
                percentile_threshold=fat_percentile_threshold,
                min_area_ratio=min_fat_area_ratio,
                max_area_ratio=max_fat_area_ratio,
                debug=debug,
                debug_save_path=debug_save_path,
                slice_index=z
            )
        
        slice_stats.update(fat_stats)
        
        # 确定使用哪个脂肪信号值
        actual_fat_mean = None
        
        if use_first_slice_fat and global_fat_mean is not None:
            # 使用指定切片的脂肪信号
            actual_fat_mean = global_fat_mean
            slice_stats['fat_source'] = 'first_slice'
            slice_stats['used_fat_signal'] = actual_fat_mean
        elif fat_mean is not None:
            # 使用当前切片的脂肪信号
            actual_fat_mean = fat_mean
            slice_stats['fat_source'] = 'current_slice'
            slice_stats['used_fat_signal'] = actual_fat_mean
            success_fat_signals.append(fat_mean)
        elif success_fat_signals:
            # 使用备用：最后一个成功的脂肪信号
            actual_fat_mean = success_fat_signals[-1]
            slice_stats['fat_source'] = 'fallback'
            slice_stats['used_fat_signal'] = actual_fat_mean
            slice_stats['status'] = 'fat_fallback'
        else:
            # 完全失败
            slice_stats['status'] = 'fat_extraction_failed'
            slice_stats_list.append(slice_stats)
            continue
        
        # 记录到病人统计中（仅当从当前切片成功提取时）
        if fat_mean is not None:
            patient_stats['fat_signal_values'].append(fat_mean)
        
        # 标准化图像
        normalized_slice = normalize_image_slice(img_slice, actual_fat_mean)
        normalized_image[:, :, z] = normalized_slice
        
        # 检查切片是否异常
        is_abnormal, abnormal_reason = check_abnormal_slice(
            lbl_slice,
            normalized_slice,
            min_csa=min_csa,
            max_fat_ratio=max_fat_ratio,
            pixel_spacing=pixel_spacing
        )
        
        if is_abnormal:
            slice_stats['status'] = 'abnormal'
            slice_stats['abnormal_reason'] = abnormal_reason
            patient_stats['invalid_slices'] += 1
        else:
            slice_stats['status'] = 'valid'
            slice_stats['muscle_csa'] = calculate_muscle_csa(lbl_slice, pixel_spacing)
            slice_stats['fat_ratio'] = calculate_fat_ratio(normalized_slice, lbl_slice)
            patient_stats['valid_slices'] += 1
        
        slice_stats_list.append(slice_stats)
    
    if len(patient_stats['fat_signal_values']) > 0:
        patient_stats['mean_fat_signal_patient'] = np.mean(patient_stats['fat_signal_values'])
        patient_stats['median_fat_signal_patient'] = np.median(patient_stats['fat_signal_values'])
        patient_stats['std_fat_signal_patient'] = np.std(patient_stats['fat_signal_values'])
    
    corrected_img = nib.Nifti1Image(corrected_image, affine)
    nib.save(corrected_img, output_path / f'{patient_id}_corrected.nii.gz')
    
    normalized_img = nib.Nifti1Image(normalized_image, affine)
    nib.save(normalized_img, output_path / f'{patient_id}_normalized.nii.gz')
    
    return patient_stats, slice_stats_list


def batch_process(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    patient_id_pattern: str = '{name}',
    pixel_spacing: Tuple[float, float] = None,
    use_first_slice_fat: bool = True,
    first_slice_index: int = 0,
    use_gmm_fat: bool = False,
    use_pure_dorsal_fat: bool = True,
    pure_dorsal_params: Dict = None,
    debug: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    批量处理所有病人

    Args:
        images_dir: 原始图像目录
        labels_dir: 分割标签目录
        output_dir: 输出目录
        patient_id_pattern: 病人ID命名模式
        pixel_spacing: 像素物理间距（如果为None，则从每个图像头文件自动读取）
        use_first_slice_fat: 是否使用指定切片的脂肪信号进行所有切片的标准化
        first_slice_index: 指定的切片索引（默认0，即第1张）
        use_gmm_fat: 是否使用高斯混合模型从全图提取脂肪信号
        use_pure_dorsal_fat: 是否使用纯净背部皮下脂肪提取方法（默认True）
        pure_dorsal_params: 纯净脂肪提取的参数字典
        debug: 是否启用调试模式

    Returns:
        (病人级统计DataFrame, 切片级统计DataFrame)
    """
    if pure_dorsal_params is None:
        pure_dorsal_params = {
            'high_percentile': 99.5,
            'min_area': 100,
            'dorsal_ratio': 0.6,
            'edge_margin': 5
        }
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    image_files = sorted(list(images_path.glob('*.nii.gz')) + list(images_path.glob('*.nii')))
    label_files = sorted(list(labels_path.glob('*.nii.gz')) + list(labels_path.glob('*.nii')))
    
    print(f'Found {len(image_files)} image files')
    print(f'Found {len(label_files)} label files')
    
    patient_stats_all = []
    slice_stats_all = []
    
    for img_file in tqdm(image_files, desc='Batch processing'):
        patient_name = img_file.stem.replace('.nii', '').replace('_0000', '')
        
        label_file = labels_path / f'{patient_name}.nii.gz'
        if not label_file.exists():
            label_file = labels_path / f'{patient_name}.nii'
        
        if not label_file.exists():
            print(f'Skipping {patient_name}: label file not found')
            continue
        
        patient_output_dir = Path(output_dir) / patient_name
        
        try:
            patient_stats, slice_stats = process_single_patient(
                str(img_file),
                str(label_file),
                patient_name,
                str(patient_output_dir),
                pixel_spacing=pixel_spacing,
                use_first_slice_fat=use_first_slice_fat,
                first_slice_index=first_slice_index,
                use_gmm_fat=use_gmm_fat,
                use_pure_dorsal_fat=use_pure_dorsal_fat,
                pure_dorsal_params=pure_dorsal_params,
                debug=debug,
                **kwargs
            )
            
            patient_stats_all.append(patient_stats)
            slice_stats_all.extend(slice_stats)
            
        except Exception as e:
            print(f'Error processing {patient_name}: {e}')
            import traceback
            traceback.print_exc()
    
    patient_df = pd.DataFrame(patient_stats_all)
    slice_df = pd.DataFrame(slice_stats_all)
    
    output_path = Path(output_dir)
    patient_df.to_csv(output_path / 'patient_statistics.csv', index=False, encoding='utf-8-sig')
    slice_df.to_csv(output_path / 'slice_statistics.csv', index=False, encoding='utf-8-sig')
    
    return patient_df, slice_df


if __name__ == '__main__':
    print('=' * 80)
    print('图像预处理脚本')
    print('=' * 80)
    
    print('\n使用说明:')
    print('''
    1. 修改下方配置参数
    2. 运行脚本进行批量预处理
    3. 结果将保存在输出目录中，包括:
       - 每个病人的校正后图像和标准化图像
       - patient_statistics.csv: 病人级统计
       - slice_statistics.csv: 切片级统计
    ''')
    
    # ==========================================
    # 配置参数（请根据您的实际情况修改）
    # ==========================================
    
    # 输入路径
    IMAGES_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\images"  # 原始图像目录
    LABELS_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\results"  # 分割标签目录
    OUTPUT_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\preprocessed-ds-copy1test1"  # 输出目录
    
    # 像素物理间距（设为None，自动从每个图像头文件读取）
    PIXEL_SPACING = None
    
    # 预处理参数（更宽松的参数设置）
    N4_SHRINK_FACTOR = 4
    FAT_PERCENTILE_THRESHOLD = 55.0  # 降低阈值以识别更多脂肪
    MIN_FAT_AREA_RATIO = 0.001  # 非常宽松的最小面积限制
    MAX_FAT_AREA_RATIO = 0.9    # 非常宽松的最大面积限制
    MIN_CSA = 10.0
    MAX_FAT_RATIO = 0.95
    
    # 脂肪信号提取策略
    USE_FIRST_SLICE_FAT = True  # 是否使用指定切片的脂肪信号进行所有切片的标准化
    FIRST_SLICE_INDEX = 0  # 指定的切片索引（0表示第1张）
    USE_GMM_FAT = False  # 是否使用高斯混合模型从全图提取脂肪信号
    USE_PURE_DORSAL_FAT = True  # 是否使用纯净背部皮下脂肪提取方法（推荐开启）
    PURE_DORSAL_PARAMS = {
        'high_percentile': 99.5,  # 用于抓取绝对高信号区域的百分位数
        'min_area': 100,           # 连通域最小面积（像素）
        'dorsal_ratio': 0.6,      # 背部区域在Y轴方向的比例
        'edge_margin': 5           # 边缘剔除宽度（像素）
    }
    DEBUG = True  # 是否启用调试模式（会保存脂肪提取的可视化图像）
    
    # ==========================================
    # 检查路径是否存在
    # ==========================================
    if not Path(IMAGES_DIR).exists():
        print(f'\n错误: 图像目录不存在: {IMAGES_DIR}')
        print('请修改 IMAGES_DIR 配置后重新运行')
        exit(1)
    
    if not Path(LABELS_DIR).exists():
        print(f'\n错误: 标签目录不存在: {LABELS_DIR}')
        print('请修改 LABELS_DIR 配置后重新运行')
        exit(1)
    
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f'\n开始批量处理...')
    print(f'图像目录: {IMAGES_DIR}')
    print(f'标签目录: {LABELS_DIR}')
    print(f'输出目录: {OUTPUT_DIR}')
    print(f'像素间距: {PIXEL_SPACING}')
    
    # ==========================================
    # 执行批量处理
    # ==========================================
    patient_df, slice_df = batch_process(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        pixel_spacing=PIXEL_SPACING,
        n4_shrink_factor=N4_SHRINK_FACTOR,
        fat_percentile_threshold=FAT_PERCENTILE_THRESHOLD,
        min_fat_area_ratio=MIN_FAT_AREA_RATIO,
        max_fat_area_ratio=MAX_FAT_AREA_RATIO,
        min_csa=MIN_CSA,
        max_fat_ratio=MAX_FAT_RATIO,
        use_first_slice_fat=USE_FIRST_SLICE_FAT,
        first_slice_index=FIRST_SLICE_INDEX,
        use_gmm_fat=USE_GMM_FAT,
        use_pure_dorsal_fat=USE_PURE_DORSAL_FAT,
        pure_dorsal_params=PURE_DORSAL_PARAMS,
        debug=DEBUG
    )
    
    # ==========================================
    # 显示结果摘要
    # ==========================================
    print('\n' + '=' * 80)
    print('处理完成!')
    print('=' * 80)
    
    print(f'\n处理结果:')
    print(f'  - 成功处理病人数: {len(patient_df)}')
    print(f'  - 总切片数: {patient_df["total_slices"].sum()}')
    print(f'  - 有效切片数: {patient_df["valid_slices"].sum()}')
    print(f'  - 无效切片数: {patient_df["invalid_slices"].sum()}')
    
    print(f'\n输出文件:')
    print(f'  - 病人统计: {OUTPUT_DIR}/patient_statistics.csv')
    print(f'  - 切片统计: {OUTPUT_DIR}/slice_statistics.csv')
    
    if len(patient_df) > 0:
        print(f'\n脂肪信号统计:')
        if 'mean_fat_signal_patient' in patient_df.columns:
            print(f'  - 平均脂肪信号: {patient_df["mean_fat_signal_patient"].mean():.2f}')
        else:
            print(f'  - 平均脂肪信号: 无有效数据')
        if 'median_fat_signal_patient' in patient_df.columns:
            print(f'  - 中位数脂肪信号: {patient_df["median_fat_signal_patient"].median():.2f}')
        else:
            print(f'  - 中位数脂肪信号: 无有效数据')
    
    print('\n病人统计预览:')
    # 确保只选择存在的列
    columns_to_show = ["patient_id", "total_slices", "valid_slices", "invalid_slices"]
    if 'mean_fat_signal_patient' in patient_df.columns:
        columns_to_show.append("mean_fat_signal_patient")
    print(patient_df[columns_to_show].to_string())
    
    print('\n切片统计预览（前10行）:')
    # 确保只选择存在的列
    slice_columns_to_show = ["patient_id", "slice_index", "status"]
    if 'mean_fat_signal' in slice_df.columns:
        slice_columns_to_show.append("mean_fat_signal")
    print(slice_df[slice_columns_to_show].head(10).to_string())
