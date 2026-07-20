"""
图像预处理脚本
功能：
1. N4偏置场校正
2. 同层动态标准化（提取背部皮下脂肪信号）

脂肪识别方法：全图信号强度阈值法
- 默认使用55%分位数阈值
- 面积比例约束：0.001 ~ 0.9
- 可选GMM高斯混合模型脂肪提取（默认关闭）
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from sklearn.mixture import GaussianMixture


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
        shrink_factor: 缩小因子，加速计算
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
    percentile_threshold: float = 55.0,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.9,
    debug: bool = False,
    debug_save_path: Optional[str] = None,
    slice_index: Optional[int] = None
) -> Tuple[Optional[float], Optional[np.ndarray], Dict]:
    """
    从图像切片中提取全图脂肪区域并计算其平均信号值

    核心识别流程：
    1. 信号强度阈值计算：计算图像切片中所有像素的百分位数（默认55%分位数）
    2. 脂肪区域二值化：信号值 > 阈值 → 判定为脂肪区域
    3. 质量检查：面积比例约束（默认0.001 ~ 0.9）

    Args:
        image_slice: 单张图像切片 (x, y)
        percentile_threshold: 用于筛选脂肪的百分位数阈值（默认55%）
        min_area_ratio: 脂肪区域最小面积比例（默认0.001）
        max_area_ratio: 脂肪区域最大面积比例（默认0.9）
        debug: 是否启用调试模式
        debug_save_path: 调试图像保存路径
        slice_index: 切片索引

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
    stats['back_region'] = 'whole_image'

    # 调试模式可视化（避免批量处理卡顿，可按需启用）
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
            plt.imshow(np.rot90(subcutaneous_fat, k=1), cmap='hot')
            plt.title(f'Fat Candidates (>{percentile_threshold}%)')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(np.rot90(subcutaneous_fat, k=1), cmap='hot')
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


def get_global_fat_signal_gmm(
    image_3d: np.ndarray,
    n_components: int = 3,
    random_state: int = 0
) -> Optional[float]:
    """
    使用高斯混合模型从整个3D图像中提取脂肪信号（可选方法）

    Args:
        image_3d: 3D图像数组，形状为(x, y, z)
        n_components: 高斯分量数量（默认3个）
        random_state: 随机种子

    Returns:
        脂肪信号平均值，如果失败返回None
    """
    try:
        all_pixels = image_3d.flatten()
        low_cut = np.percentile(all_pixels, 5)
        high_cut = np.percentile(all_pixels, 99)
        filtered = all_pixels[(all_pixels > low_cut) & (all_pixels < high_cut)]

        if len(filtered) < 100:
            return None

        data = filtered.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=3)
        gmm.fit(data)
        means = gmm.means_.flatten()
        fat_mean = np.max(means)

        return float(fat_mean)
    except Exception as e:
        print(f'  GMM脂肪提取失败: {e}')
        return None


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


def process_single_patient(
    image_path: str,
    label_path: str,
    patient_id: str,
    output_dir: str,
    pixel_spacing: Tuple[float, float] = None,
    n4_shrink_factor: int = 4,
    fat_percentile_threshold: float = 55.0,
    min_fat_area_ratio: float = 0.001,
    max_fat_area_ratio: float = 0.9,
    use_first_slice_fat: bool = True,
    first_slice_index: int = 0,
    use_gmm_fat: bool = False
) -> bool:
    """
    处理单个病人的图像和分割标签

    Args:
        image_path: 原始图像路径
        label_path: 分割标签路径
        patient_id: 病人ID
        output_dir: 输出目录
        pixel_spacing: 像素物理间距（如果为None，则从图像头文件自动读取）
        n4_shrink_factor: N4校正缩小因子
        fat_percentile_threshold: 脂肪百分位数阈值（默认55%）
        min_fat_area_ratio: 最小脂肪面积比例（默认0.001）
        max_fat_area_ratio: 最大脂肪面积比例（默认0.9）
        use_first_slice_fat: 是否使用指定切片的脂肪信号进行所有切片的标准化
        first_slice_index: 指定的切片索引（默认0，即第1张）
        use_gmm_fat: 是否使用高斯混合模型从全图提取脂肪信号（默认关闭）

    Returns:
        是否处理成功
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        img = nib.load(image_path)
        image_data = img.get_fdata()
        affine = img.affine

        # 从nifti头文件读取真实像素间距
        if pixel_spacing is None:
            zooms = img.header.get_zooms()
            pixel_spacing = (zooms[0], zooms[1])

        # 加载标签（仅用于验证形状，不进行特征计算）
        lbl = nib.load(label_path)
        label_data = lbl.get_fdata().astype(np.int32)

        assert image_data.shape == label_data.shape, f'Image shape {image_data.shape} != label shape {label_data.shape}'

        print(f'Processing patient {patient_id}, shape: {image_data.shape}')

        # N4偏置场校正
        corrected_image = n4_bias_field_correction(
            image_data,
            shrink_factor=n4_shrink_factor
        )

        normalized_image = np.zeros_like(corrected_image)

        # 提取脂肪信号（优先使用指定切片）
        global_fat_mean = None

        # 策略1：GMM全局脂肪提取（可选）
        if use_gmm_fat:
            print(f'  使用高斯混合模型从全图提取脂肪信号...')
            gmm_fat = get_global_fat_signal_gmm(corrected_image)
            if gmm_fat is not None:
                global_fat_mean = gmm_fat
                print(f'  GMM全局脂肪信号: {global_fat_mean:.2f}')

        # 策略2：使用指定切片的脂肪信号（默认）
        if global_fat_mean is None and use_first_slice_fat and first_slice_index < image_data.shape[2]:
            first_slice = corrected_image[:, :, first_slice_index]
            fat_mean, _, _ = extract_subcutaneous_fat(
                first_slice,
                percentile_threshold=fat_percentile_threshold,
                min_area_ratio=min_fat_area_ratio,
                max_area_ratio=max_fat_area_ratio
            )
            if fat_mean is not None:
                global_fat_mean = fat_mean
                print(f'  使用第{first_slice_index}张切片的脂肪信号: {global_fat_mean:.2f}')
            else:
                print(f'  指定切片脂肪提取失败，将使用逐切片提取')

        # 备用脂肪信号列表
        success_fat_signals = []

        # 处理每个切片
        for z in tqdm(range(image_data.shape[2]), desc=f'Patient {patient_id}', leave=False):
            img_slice = corrected_image[:, :, z]

            # 提取当前切片的脂肪信号
            fat_mean, _, _ = extract_subcutaneous_fat(
                img_slice,
                percentile_threshold=fat_percentile_threshold,
                min_area_ratio=min_fat_area_ratio,
                max_area_ratio=max_fat_area_ratio
            )

            # 确定使用哪个脂肪信号值
            actual_fat_mean = None

            if use_first_slice_fat and global_fat_mean is not None:
                actual_fat_mean = global_fat_mean
            elif fat_mean is not None:
                actual_fat_mean = fat_mean
                success_fat_signals.append(fat_mean)
            elif success_fat_signals:
                actual_fat_mean = success_fat_signals[-1]
            else:
                # 完全失败，使用图像均值作为回退
                actual_fat_mean = np.mean(img_slice)

            # 标准化图像
            normalized_slice = normalize_image_slice(img_slice, actual_fat_mean)
            normalized_image[:, :, z] = normalized_slice

        # 保存校正后和标准化后的图像
        corrected_img = nib.Nifti1Image(corrected_image, affine)
        nib.save(corrected_img, output_path / f'{patient_id}_corrected.nii.gz')

        normalized_img = nib.Nifti1Image(normalized_image, affine)
        nib.save(normalized_img, output_path / f'{patient_id}_normalized.nii.gz')

        print(f'  处理完成')
        return True

    except Exception as e:
        print(f'Error processing {patient_id}: {e}')
        import traceback
        traceback.print_exc()
        return False


def batch_process(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    pixel_spacing: Tuple[float, float] = None,
    use_first_slice_fat: bool = True,
    first_slice_index: int = 0,
    use_gmm_fat: bool = False,
    **kwargs
) -> int:
    """
    批量处理所有病人

    Args:
        images_dir: 原始图像目录
        labels_dir: 分割标签目录
        output_dir: 输出目录
        pixel_spacing: 像素物理间距（如果为None，则从每个图像头文件自动读取）
        use_first_slice_fat: 是否使用指定切片的脂肪信号进行所有切片的标准化
        first_slice_index: 指定的切片索引（默认0，即第1张）
        use_gmm_fat: 是否使用高斯混合模型从全图提取脂肪信号（默认关闭）

    Returns:
        成功处理的病人数量
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    image_files = sorted(list(images_path.glob('*.nii.gz')) + list(images_path.glob('*.nii')))
    label_files = sorted(list(labels_path.glob('*.nii.gz')) + list(labels_path.glob('*.nii')))

    print(f'Found {len(image_files)} image files')
    print(f'Found {len(label_files)} label files')

    success_count = 0

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
            success = process_single_patient(
                str(img_file),
                str(label_file),
                patient_name,
                str(patient_output_dir),
                pixel_spacing=pixel_spacing,
                use_first_slice_fat=use_first_slice_fat,
                first_slice_index=first_slice_index,
                use_gmm_fat=use_gmm_fat,
                **kwargs
            )

            if success:
                success_count += 1

        except Exception as e:
            print(f'Error processing {patient_name}: {e}')
            import traceback
            traceback.print_exc()

    print(f'\n批量处理完成，成功处理 {success_count}/{len(image_files)} 个病人')
    return success_count


if __name__ == '__main__':
    print('=' * 80)
    print('图像预处理脚本')
    print('脂肪识别方法：全图信号强度阈值法（55%分位数）')
    print('=' * 80)

    # 配置参数
    IMAGES_DIR = r'E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\images'
    LABELS_DIR = r'E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\labels'
    OUTPUT_DIR = r'E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\output'

    # 处理参数
    USE_FIRST_SLICE_FAT = True  # 使用第一张切片的脂肪信号进行标准化
    FIRST_SLICE_INDEX = 0       # 使用第0张切片
    FAT_PERCENTILE_THRESHOLD = 55.0  # 脂肪百分位数阈值（55%）
    MIN_FAT_AREA_RATIO = 0.001   # 最小脂肪面积比例
    MAX_FAT_AREA_RATIO = 0.9     # 最大脂肪面积比例
    USE_GMM_FAT = False          # 是否使用高斯混合模型（默认关闭）

    print(f'\n配置参数:')
    print(f'  图像目录: {IMAGES_DIR}')
    print(f'  标签目录: {LABELS_DIR}')
    print(f'  输出目录: {OUTPUT_DIR}')
    print(f'  脂肪百分位数阈值: {FAT_PERCENTILE_THRESHOLD}%')
    print(f'  脂肪面积比例范围: [{MIN_FAT_AREA_RATIO}, {MAX_FAT_AREA_RATIO}]')
    print(f'  使用首切片脂肪: {USE_FIRST_SLICE_FAT}')
    print(f'  首切片索引: {FIRST_SLICE_INDEX}')
    print(f'  使用GMM脂肪提取: {USE_GMM_FAT}')

    # 执行批量处理
    batch_process(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        use_first_slice_fat=USE_FIRST_SLICE_FAT,
        first_slice_index=FIRST_SLICE_INDEX,
        use_gmm_fat=USE_GMM_FAT,
        fat_percentile_threshold=FAT_PERCENTILE_THRESHOLD,
        min_fat_area_ratio=MIN_FAT_AREA_RATIO,
        max_fat_area_ratio=MAX_FAT_AREA_RATIO
    )
