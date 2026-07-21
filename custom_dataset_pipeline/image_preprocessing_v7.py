"""MRI 图像预处理 v7：物理空间 N4 偏置场校正与 CSF 信号归一化。

以 patient_id 为唯一标识符：
  - images: "{patient_id}_{姓名拼音}_0000.nii.gz"
  - labels: "{patient_id}_{姓名拼音}.nii.gz"
  - 输出目录: "{output_dir}/{patient_id}/"
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from patient_selection_v7 import (
    canonical_patient_id,
    filter_paths_to_patients,
    load_patient_list_with_csf,
    patient_id_from_nifti,
)


# ==================== 固定路径配置 ====================
IMAGES_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\images'
LABELS_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\labels'
OUTPUT_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\output_v7_20260721'
PATIENT_LIST_FILE = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\PATIENT_LIST_FILE.xlsx'


def calculate_n4_shrink_factors(image: sitk.Image, shrink_factor: int):
    """根据真实体素间距计算各方向的自适应 N4 缩小倍数。"""
    spacing = image.GetSpacing()
    target_step = min(spacing) * int(shrink_factor)
    factors = [
        max(1, int(np.floor(target_step / value + 0.500001)))
        for value in spacing
    ]
    return [
        min(factor, max(1, size // 4))
        for factor, size in zip(factors, image.GetSize())
    ]


def n4_bias_field_correction(
    image: sitk.Image,
    mask: Optional[sitk.Image] = None,
    shrink_factor: int = 4,
    convergence_threshold: float = 1e-6,
    max_iterations: Tuple[int, ...] = (50, 50, 50, 50),
) -> sitk.Image:
    """在缩小的物理网格估计 N4 偏置场，再校正原始分辨率图像。"""
    if shrink_factor < 1:
        raise ValueError(f"shrink_factor must be >= 1, got {shrink_factor}")
    original = sitk.Cast(image, sitk.sitkFloat32)
    mask_image = (
        sitk.OtsuThreshold(original, 0, 1, 200)
        if mask is None else sitk.Cast(mask, sitk.sitkUInt8)
    )
    if mask_image.GetSize() != original.GetSize():
        raise ValueError("N4 mask size does not match image size")

    factors = calculate_n4_shrink_factors(original, shrink_factor)
    working_image = sitk.Shrink(original, factors) if shrink_factor > 1 else original
    working_mask = sitk.Shrink(mask_image, factors) if shrink_factor > 1 else mask_image
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetConvergenceThreshold(float(convergence_threshold))
    corrector.SetMaximumNumberOfIterations([int(x) for x in max_iterations])
    corrector.Execute(working_image, working_mask)

    log_bias = corrector.GetLogBiasFieldAsImage(original)
    corrected = original / sitk.Exp(log_bias)

    foreground = sitk.GetArrayViewFromImage(mask_image).astype(bool)
    before = sitk.GetArrayViewFromImage(original)[foreground]
    after = sitk.GetArrayViewFromImage(corrected)[foreground]
    if before.size and after.size:
        before_median = float(np.median(before))
        after_median = float(np.median(after))
        if np.isfinite(before_median) and np.isfinite(after_median) and after_median > 0:
            corrected = corrected * (before_median / after_median)
    return corrected


def decide_csf_reference(csf_mean: float, csf_median: Optional[float] = None):
    if csf_median is not None and np.isfinite(csf_median) and csf_median > 0:
        return float(csf_median), "MEDIAN"
    if csf_mean is not None and np.isfinite(csf_mean) and csf_mean > 0:
        return float(csf_mean), "MEAN"
    raise ValueError(f"No valid CSF reference: mean={csf_mean}, median={csf_median}")


def process_single_patient(
    image_path: str,
    label_path: str,
    patient_id: str,
    output_dir: str,
    csf_mean: float,
    csf_median: Optional[float] = None,
    n4_shrink_factor: int = 4,
) -> Dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference, reference_type = decide_csf_reference(csf_mean, csf_median)
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    label = sitk.ReadImage(str(label_path))
    if image.GetSize() != label.GetSize():
        raise ValueError(f"Image size {image.GetSize()} != label size {label.GetSize()}")

    corrected = n4_bias_field_correction(image, shrink_factor=n4_shrink_factor)
    normalized = corrected / reference
    corrected_path = output / f"{patient_id}_corrected.nii.gz"
    normalized_path = output / f"{patient_id}_normalized.nii.gz"
    sitk.WriteImage(corrected, str(corrected_path))
    sitk.WriteImage(normalized, str(normalized_path))

    values = sitk.GetArrayViewFromImage(normalized)
    log = {
        "version": "v7",
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_image": str(image_path),
        "input_label": str(label_path),
        "image_size": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "n4_shrink_factor": n4_shrink_factor,
        "n4_shrink_factors_xyz": calculate_n4_shrink_factors(image, n4_shrink_factor),
        "csf_reference_used": reference,
        "reference_type": reference_type,
        "normalized_finite_fraction": float(np.mean(np.isfinite(values))),
        "status": "success",
    }
    (output / f"{patient_id}_log_v7.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return log


def _find_label(labels_dir: Path, patient_id: str) -> Path:
    """根据 patient_id 在 labels 目录查找匹配的标签文件。

    支持文件名格式："{patient_id}_{姓名拼音}.nii.gz"
    """
    labels_root = Path(labels_dir)
    candidates = list(labels_root.glob("*.nii.gz")) + list(labels_root.glob("*.nii"))
    matches = [
        path for path in candidates
        if canonical_patient_id(patient_id_from_nifti(path)) == canonical_patient_id(patient_id)
    ]
    if len(matches) == 0:
        raise FileNotFoundError(f"未找到 patient_id={patient_id} 的标签文件")
    if len(matches) > 1:
        raise FileNotFoundError(f"patient_id={patient_id} 匹配到多个标签文件: {matches}")
    return matches[0]


def validate_input_consistency(
    patient_ids: Dict[str, str],
    csf_values: Dict[str, float],
    images_dir: Path,
    labels_dir: Path,
) -> Dict:
    """验证输入数据的一致性：PATIENT_LIST_FILE、图像文件、标签文件、CSF 值。

    Args:
        patient_ids: {canonical_id: original_id} 映射
        csf_values: {canonical_id: csf_value} 映射
        images_dir: 原始图像目录
        labels_dir: 标签文件目录

    Returns:
        验证结果字典
    """
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)

    image_candidates = list(images_root.glob("*.nii")) + list(images_root.glob("*.nii.gz"))
    label_candidates = list(labels_root.glob("*.nii.gz")) + list(labels_root.glob("*.nii"))

    image_ids = {
        canonical_patient_id(patient_id_from_nifti(p)) for p in image_candidates
    }
    label_ids = {
        canonical_patient_id(patient_id_from_nifti(p)) for p in label_candidates
    }

    result = {
        "total_in_patient_list": len(patient_ids),
        "total_images_found": len(image_ids),
        "total_labels_found": len(label_ids),
        "images_in_list": 0,
        "images_missing_from_list": [],
        "labels_in_list": 0,
        "labels_missing_from_list": [],
        "patients_missing_images": [],
        "patients_missing_labels": [],
        "patients_with_csf": 0,
        "patients_missing_csf": [],
        "csf_range": {"min": None, "max": None, "mean": None},
    }

    csf_list = []
    for key in sorted(patient_ids.keys()):
        patient_id = patient_ids[key]
        if key in image_ids:
            result["images_in_list"] += 1
        else:
            result["patients_missing_images"].append(patient_id)
        if key in label_ids:
            result["labels_in_list"] += 1
        else:
            result["patients_missing_labels"].append(patient_id)
        if key in csf_values and csf_values[key] is not None:
            result["patients_with_csf"] += 1
            csf_list.append(csf_values[key])
        else:
            result["patients_missing_csf"].append(patient_id)

    if csf_list:
        result["csf_range"]["min"] = float(min(csf_list))
        result["csf_range"]["max"] = float(max(csf_list))
        result["csf_range"]["mean"] = float(np.mean(csf_list))

    list_keys = set(patient_ids.keys())
    result["images_missing_from_list"] = sorted(
        p for p in image_ids if p not in list_keys
    )
    result["labels_missing_from_list"] = sorted(
        p for p in label_ids if p not in list_keys
    )

    return result


def batch_process(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    patient_list_file: str,
    n4_shrink_factor: int = 4,
    continue_on_error: bool = True,
) -> Dict:
    """批量预处理。

    Args:
        images_dir: 原始图像目录
        labels_dir: 分割标签目录
        output_dir: 输出目录
        patient_list_file: 病人列表 Excel（含 patient_id 和 CSF 值）
        n4_shrink_factor: N4 缩小因子
        continue_on_error: 单个病人失败时是否继续

    Returns:
        汇总字典
    """
    images_root, labels_root, output_root = map(
        Path, (images_dir, labels_dir, output_dir)
    )

    patient_ids, csf_values = load_patient_list_with_csf(patient_list_file)

    validation = validate_input_consistency(
        patient_ids, csf_values, images_root, labels_root
    )
    print(f"\n输入数据一致性验证:")
    print(f"  PATIENT_LIST_FILE 病人数: {validation['total_in_patient_list']}")
    print(f"  图像目录总文件数: {validation['total_images_found']}")
    print(f"  标签目录总文件数: {validation['total_labels_found']}")
    print(f"  列表中有图像的病人: {validation['images_in_list']}")
    print(f"  列表中有标签的病人: {validation['labels_in_list']}")
    print(f"  列表中有 CSF 值的病人: {validation['patients_with_csf']}")
    if validation["patients_missing_images"]:
        print(f"  列表中缺少图像的病人 ({len(validation['patients_missing_images'])}): {validation['patients_missing_images'][:5]}...")
    if validation["patients_missing_labels"]:
        print(f"  列表中缺少标签的病人 ({len(validation['patients_missing_labels'])}): {validation['patients_missing_labels'][:5]}...")
    if validation["images_missing_from_list"]:
        print(f"  图像中不在列表里的 ({len(validation['images_missing_from_list'])}): {validation['images_missing_from_list'][:5]}...")
    if validation["labels_missing_from_list"]:
        print(f"  标签中不在列表里的 ({len(validation['labels_missing_from_list'])}): {validation['labels_missing_from_list'][:5]}...")
    print()

    candidates = list(images_root.glob("*.nii")) + list(images_root.glob("*.nii.gz"))
    images, missing = filter_paths_to_patients(candidates, patient_ids)
    if missing:
        names = [patient_ids[key] for key in sorted(missing)]
        raise FileNotFoundError(f"缺少 MRI 图像的患者: {names}")

    successful = []
    failed = []
    for image_path in tqdm(images, desc="v7 图像预处理"):
        key = canonical_patient_id(patient_id_from_nifti(image_path))
        patient_id = patient_ids[key]
        try:
            label_path = _find_label(labels_root, patient_id)
            csf_mean = csf_values[key]
            log = process_single_patient(
                str(image_path),
                str(label_path),
                patient_id,
                str(output_root / patient_id),
                csf_mean=csf_mean,
                csf_median=None,
                n4_shrink_factor=n4_shrink_factor,
            )
            successful.append(log)
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            failed.append({
                "patient_id": patient_id,
                "reason": error_msg,
                "error_type": error_type,
            })
            tqdm.write(f"  ⚠️  {patient_id} 预处理失败: {error_type} - {error_msg}")
            if not continue_on_error:
                raise

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "v7",
        "total_requested": len(patient_ids),
        "success_count": len(successful),
        "failed_count": len(failed),
        "successful_patients": [x["patient_id"] for x in successful],
        "failed_patients": failed,
        "index_by": "patient_id",
        "continue_on_error": continue_on_error,
        "validation": validation,
    }
    (output_root / "batch_processing_log_v7.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nv7 预处理完成：成功 {len(successful)}/{len(patient_ids)} 人，失败 {len(failed)} 人")
    return summary


def main():
    """使用文件顶部的固定路径执行预处理。"""
    summary = batch_process(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        patient_list_file=PATIENT_LIST_FILE,
        n4_shrink_factor=4,
        continue_on_error=True,
    )
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
