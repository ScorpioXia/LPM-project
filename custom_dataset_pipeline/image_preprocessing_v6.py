"""
图像预处理脚本 (版本 v6)
功能：
1. N4偏置场校正
2. 基于脑脊液（CSF）信号的图像标准化（替代原脂肪信号标准化）

标准化策略：
- 优先使用 CSF 中位数（抗部分容积效应，若CSV中提供）
- 若无中位数，则降级使用 CSF 均值
- N4校正 → CSF标准化（顺序不可颠倒）

新增特性：
- 从 CSV 文件读取每位病人的 CSF 参考信号值
- 兼容仅包含 csf_reference 列的旧 CSV
- 自动记录每位病人实际使用的参考类型（MEDIAN / MEAN）
- 输出标准化日志为 JSON 文件，便于后续回溯分析

输出文件（每个病人一个子目录）：
- {patient_id}_corrected.nii.gz: N4偏置场校正后的图像
- {patient_id}_normalized.nii.gz: CSF标准化后的图像
- {patient_id}_log.json: 预处理日志（CSF值、标准化类型、处理状态等）
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from datetime import datetime
import json


def n4_bias_field_correction(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    shrink_factor: int = 4,
    convergence_threshold: float = 1e-6,
    max_iterations: Tuple[int, int, int, int] = (50, 50, 50, 50)
) -> np.ndarray:
    """
    使用 SimpleITK 进行 N4 偏置场校正

    Args:
        image: 输入图像数组，形状为 (x, y, z)
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
        print(f'  N4 校正失败，跳过校正: {e}')
        return image


def normalize_image_slice(
    image_slice: np.ndarray,
    csf_reference: float
) -> np.ndarray:
    """
    使用脑脊液参考信号值（均值或中位数）标准化图像切片

    原理：将图像中所有像素值除以 CSF 参考信号值
    结果：CSF 区域信号 ≈ 1.0，其它组织按比例缩放

    Args:
        image_slice: 原始图像切片（建议先经过 N4 校正）
        csf_reference: 该患者 CSF 参考值（优先中位数，若无则用均值）

    Returns:
        标准化后的图像切片
    """
    if csf_reference <= 0:
        raise ValueError(f'csf_reference 必须为正数，当前值为 {csf_reference}')

    normalized = image_slice / csf_reference
    return normalized


def check_csf_normalization(
    image_norm: np.ndarray,
    tol: float = 0.2
) -> Dict:
    """
    粗略检查中心区域高信号是否接近 1.0

    由于没有脑脊液区域标注，用图像中心区域的亮区均值近似验证
    若偏离 1.0 过大（如 > ±0.2），返回警告信息

    Args:
        image_norm: 标准化后的图像切片
        tol: 容差（默认 0.2，即偏离 1.0 超过 ±20% 警告）

    Returns:
        包含检查结果的字典
    """
    result = {
        'check_passed': True,
        'center_bright_mean': None,
        'deviation': None,
        'warning': None
    }

    h, w = image_norm.shape[:2]
    c_h, c_w = h // 2, w // 2

    # 取中心区域（上下左右各 1/5 的区域）
    roi = image_norm[c_h - h // 5: c_h + h // 5,
                     c_w - w // 5: c_w + w // 5]

    # 假设 CSF 信号高于 0.8（标准化后）
    bright = roi[roi > 0.8]
    if len(bright) > 0:
        measured = float(np.mean(bright))
    else:
        measured = float(np.max(roi))

    deviation = abs(measured - 1.0)

    result['center_bright_mean'] = measured
    result['deviation'] = deviation

    if deviation > tol:
        result['check_passed'] = False
        result['warning'] = f'中心亮区均值 {measured:.3f} 偏离 1.0 超过 {tol:.1f}'

    return result


def decide_csf_reference(
    csf_mean: float,
    csf_median: Optional[float] = None
) -> Tuple[float, str]:
    """
    决策使用哪个 CSF 参考值

    策略：
    1. 优先使用中位数（抗部分容积效应）
    2. 若中位数不可用（None、NaN 或 ≤0），则使用均值
    3. 两者都无效时报错

    Args:
        csf_mean: CSF 均值（必填，保底值）
        csf_median: CSF 中位数（可选，优选值）

    Returns:
        (实际使用的参考值, 类型标签 'MEDIAN' / 'MEAN')
    """
    ref_type = 'UNKNOWN'
    csf_ref = None

    # 检查中位数是否可用
    median_valid = (
        csf_median is not None
        and not (isinstance(csf_median, float) and np.isnan(csf_median))
        and csf_median > 0
    )

    if median_valid:
        csf_ref = csf_median
        ref_type = 'MEDIAN'
    elif csf_mean is not None and not np.isnan(csf_mean) and csf_mean > 0:
        csf_ref = csf_mean
        ref_type = 'MEAN'
    else:
        raise ValueError(
            f'没有有效的 CSF 参考值: mean={csf_mean}, median={csf_median}'
        )

    return csf_ref, ref_type


def process_single_patient(
    image_path: str,
    label_path: str,
    patient_id: str,
    output_dir: str,
    csf_mean: float,
    csf_median: Optional[float] = None,
    n4_shrink_factor: int = 4,
    normalization_check: bool = True
) -> bool:
    """
    处理单个病人的图像和分割标签

    处理流程：
    1. 加载原始图像和标签
    2. N4 偏置场校正
    3. 基于 CSF 参考值进行逐切片标准化
    4. （可选）中心亮区质量检查
    5. 保存校正后图像、标准化后图像、日志

    Args:
        image_path: 原始图像路径
        label_path: 分割标签路径
        patient_id: 病人ID
        output_dir: 输出目录
        csf_mean: CSF 均值（必填）
        csf_median: CSF 中位数（可选，优先使用）
        n4_shrink_factor: N4 校正缩小因子
        normalization_check: 是否进行标准化质量检查

    Returns:
        是否处理成功
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 初始化日志字典
    log_data = {
        'patient_id': patient_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_image': str(image_path),
        'input_label': str(label_path),
        'csf_mean': csf_mean,
        'csf_median': csf_median,
        'csf_reference_used': None,
        'reference_type': None,
        'status': 'failed',
        'error': None,
        'normalization_check': None,
        'image_shape': None,
        'voxel_spacing': None
    }

    try:
        # ---- 1. 决策：选择 CSF 参考值 ----
        csf_ref, ref_type = decide_csf_reference(csf_mean, csf_median)
        log_data['csf_reference_used'] = float(csf_ref)
        log_data['reference_type'] = ref_type
        print(f'处理病人 {patient_id}')
        print(f'  使用 CSF {ref_type} = {csf_ref:.2f}')

        # ---- 2. 加载数据 ----
        img = nib.load(image_path)
        image_data = img.get_fdata()
        affine = img.affine
        zooms = img.header.get_zooms()
        pixel_spacing = (zooms[0], zooms[1], zooms[2] if len(zooms) > 2 else 1.0)
        log_data['image_shape'] = list(image_data.shape)
        log_data['voxel_spacing'] = [float(z) for z in pixel_spacing]

        lbl = nib.load(label_path)
        label_data = lbl.get_fdata().astype(np.int32)
        assert image_data.shape == label_data.shape, \
            f'图像形状 {image_data.shape} 与标签形状 {label_data.shape} 不匹配'

        # ---- 3. N4 偏置场校正 ----
        print(f'  进行 N4 偏置场校正...')
        corrected_image = n4_bias_field_correction(
            image_data,
            shrink_factor=n4_shrink_factor
        )

        # ---- 4. 逐层标准化（除以 CSF 参考值） ----
        print(f'  基于 CSF 信号进行图像标准化...')
        normalized_image = np.zeros_like(corrected_image)
        for z in tqdm(range(corrected_image.shape[2]),
                      desc=f'标准化 {patient_id}', leave=False):
            normalized_image[:, :, z] = normalize_image_slice(
                corrected_image[:, :, z], csf_ref
            )

        # ---- 5. 标准化质量检查（可选） ----
        if normalization_check and normalized_image.shape[2] > 0:
            mid_z = normalized_image.shape[2] // 2
            check_result = check_csf_normalization(normalized_image[:, :, mid_z])
            log_data['normalization_check'] = check_result
            if not check_result['check_passed']:
                print(f'  警告: {check_result["warning"]}')

        # ---- 6. 保存结果 ----
        corrected_img = nib.Nifti1Image(corrected_image, affine)
        nib.save(corrected_img, output_path / f'{patient_id}_corrected.nii.gz')
        normalized_img = nib.Nifti1Image(normalized_image, affine)
        nib.save(normalized_img, output_path / f'{patient_id}_normalized.nii.gz')

        log_data['status'] = 'success'
        print(f'  处理完成 ✓')

    except Exception as e:
        error_msg = f'处理 {patient_id} 时发生错误: {e}'
        log_data['error'] = str(e)
        print(f'  ✗ {error_msg}')
        import traceback
        traceback.print_exc()

    # ---- 无论成功失败，都保存日志 ----
    log_file = output_path / f'{patient_id}_log.json'
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'  警告: 日志文件保存失败: {e}')

    return log_data['status'] == 'success'


def load_csf_data(csv_path: str) -> Dict[str, Dict[str, float]]:
    """
    从 CSV 文件加载 CSF 数据

    兼容两种列名格式：
    - 新格式: 包含 csf_mean 和可选的 csf_median 列
    - 旧格式: 仅包含 csf_reference 列（自动映射为 csf_mean）

    支持当前实际数据: 姓名, patient_id, csf_reference

    Args:
        csv_path: CSV 文件路径

    Returns:
        { patient_id: {'mean': float, 'median': float or None} }
    """
    import pandas as pd

    csf_df = pd.read_csv(csv_path, encoding='gbk')
    csf_dict = {}

    # 检查可用的列
    columns = set(csf_df.columns)

    # 确定 mean 列
    mean_col = None
    if 'csf_mean' in columns:
        mean_col = 'csf_mean'
    elif 'csf_reference' in columns:
        mean_col = 'csf_reference'
    else:
        raise KeyError(
            f'CSV 必须包含 csf_mean 或 csf_reference 列。当前列: {list(csf_df.columns)}'
        )

    # 确定 median 列（可能不存在）
    median_col = 'csf_median' if 'csf_median' in columns else None

    # 加载数据
    for _, row in csf_df.iterrows():
        pid = str(row['patient_id']).strip()

        # 读取 mean 值
        mean_val = row[mean_col]
        try:
            mean_val = float(mean_val)
        except (ValueError, TypeError):
            mean_val = None

        # 读取 median 值（如果有列）
        median_val = None
        if median_col is not None:
            med_raw = row[median_col]
            try:
                med_val = float(med_raw)
                if not np.isnan(med_val) and med_val > 0:
                    median_val = med_val
            except (ValueError, TypeError):
                median_val = None

        csf_dict[pid] = {
            'mean': mean_val,
            'median': median_val
        }

    print(f'从 CSV 加载了 {len(csf_dict)} 个病人的 CSF 数据')
    if median_col is None:
        print('  注意: CSV 中未找到 csf_median 列，将全部使用 csf_mean')

    return csf_dict


def batch_process(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    csf_file: str,
    n4_shrink_factor: int = 4,
    normalization_check: bool = True
) -> int:
    """
    批量处理所有病人

    Args:
        images_dir: 原始图像目录
        labels_dir: 分割标签目录
        output_dir: 输出目录
        csf_file: 包含 CSF 信号数据的 CSV 文件路径
        n4_shrink_factor: N4 校正缩小因子
        normalization_check: 是否进行标准化质量检查

    Returns:
        成功处理的病人数量
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    image_files = sorted(list(images_path.glob('*.nii.gz')) + list(images_path.glob('*.nii')))
    label_files = sorted(list(labels_path.glob('*.nii.gz')) + list(labels_path.glob('*.nii')))

    print(f'找到 {len(image_files)} 个图像文件')
    print(f'找到 {len(label_files)} 个标签文件')

    # 加载 CSF 数据
    csf_dict = load_csf_data(csf_file)

    success_count = 0
    failed_patients = []
    skipped_patients = []

    for img_file in tqdm(image_files, desc='批量处理'):
        patient_name = img_file.stem.replace('.nii', '').replace('_0000', '')

        # 查找对应标签文件
        label_file = labels_path / f'{patient_name}.nii.gz'
        if not label_file.exists():
            label_file = labels_path / f'{patient_name}.nii'

        if not label_file.exists():
            print(f'跳过 {patient_name}: 未找到标签文件')
            skipped_patients.append({'patient_id': patient_name, 'reason': 'label_not_found'})
            continue

        # 检查是否有 CSF 数据
        if patient_name not in csf_dict:
            print(f'跳过 {patient_name}: 未找到 CSF 数据')
            skipped_patients.append({'patient_id': patient_name, 'reason': 'no_csf_data'})
            continue

        csf_info = csf_dict[patient_name]
        if csf_info['mean'] is None or csf_info['mean'] <= 0:
            print(f'跳过 {patient_name}: CSF 均值无效 ({csf_info["mean"]})')
            skipped_patients.append({'patient_id': patient_name, 'reason': 'invalid_csf_mean'})
            continue

        patient_output_dir = Path(output_dir) / patient_name

        try:
            success = process_single_patient(
                str(img_file),
                str(label_file),
                patient_name,
                str(patient_output_dir),
                csf_mean=csf_info['mean'],
                csf_median=csf_info['median'],
                n4_shrink_factor=n4_shrink_factor,
                normalization_check=normalization_check
            )

            if success:
                success_count += 1
            else:
                failed_patients.append({'patient_id': patient_name, 'reason': 'processing_error'})

        except Exception as e:
            print(f'处理 {patient_name} 时出错: {e}')
            failed_patients.append({'patient_id': patient_name, 'reason': str(e)})
            import traceback
            traceback.print_exc()

    # 保存批量处理汇总日志
    batch_log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'success_count': success_count,
        'failed_count': len(failed_patients),
        'skipped_count': len(skipped_patients),
        'csf_file': csf_file,
        'images_dir': images_dir,
        'labels_dir': labels_dir,
        'output_dir': output_dir,
        'failed_patients': failed_patients,
        'skipped_patients': skipped_patients
    }

    try:
        batch_log_path = Path(output_dir) / 'batch_processing_log_v6.json'
        with open(batch_log_path, 'w', encoding='utf-8') as f:
            json.dump(batch_log, f, ensure_ascii=False, indent=2)
        print(f'批量处理日志已保存: {batch_log_path}')
    except Exception as e:
        print(f'警告: 批量日志保存失败: {e}')

    print(f'\n批量处理完成，成功处理 {success_count}/{len(image_files)} 个病人')
    return success_count


if __name__ == '__main__':
    print('=' * 80)
    print('图像预处理脚本 (版本 v6)')
    print('标准化基准: 脑脊液（CSF）信号，优先中位数，降级为均值')
    print('=' * 80)

    # 配置参数
    IMAGES_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\images'
    LABELS_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\labels'
    OUTPUT_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\output_v6_20260623'
    CSF_FILE = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\csf_data.csv'

    N4_SHRINK_FACTOR = 4
    NORMALIZATION_CHECK = True

    print(f'\n配置参数:')
    print(f'  图像目录: {IMAGES_DIR}')
    print(f'  标签目录: {LABELS_DIR}')
    print(f'  输出目录: {OUTPUT_DIR}')
    print(f'  CSF 文件: {CSF_FILE}')
    print(f'  N4 缩小因子: {N4_SHRINK_FACTOR}')
    print(f'  标准化质量检查: {NORMALIZATION_CHECK}')

    # 执行批量处理
    batch_process(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        csf_file=CSF_FILE,
        n4_shrink_factor=N4_SHRINK_FACTOR,
        normalization_check=NORMALIZATION_CHECK
    )
