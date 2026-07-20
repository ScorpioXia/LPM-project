"""MRI 图像预处理 v7：物理空间 N4 偏置场校正与 CSF 信号归一化。"""

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
    load_patient_ids,
    patient_id_from_nifti,
)


# ==================== 固定路径配置 ====================
IMAGES_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\images'
LABELS_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\labels'
OUTPUT_DIR = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\output_v7_20260719'
CSF_FILE = r'D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\csf_data.csv'
PATIENT_LIST_FILE = r'E:\code\Proj_0501\Proj_0501\patient_stable_311.xlsx'


def calculate_n4_shrink_factors(image: sitk.Image, shrink_factor: int):
    """根据真实体素间距计算各方向的自适应 N4 缩小倍数。"""
    spacing = image.GetSpacing()
    target_step = min(spacing) * int(shrink_factor)
    # 使用明确的四舍五入，避免浮点数 3.499999 被 Python 的 round 取为 3。
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

    # 按真实 spacing 自适应缩小：以最小体素间距方向的 shrink_factor 为基准，
    # 让各方向缩小后的物理采样间隔尽量接近。厚层方向通常保持为 1，避免过度降采样。
    factors = calculate_n4_shrink_factors(original, shrink_factor)
    working_image = sitk.Shrink(original, factors) if shrink_factor > 1 else original
    working_mask = sitk.Shrink(mask_image, factors) if shrink_factor > 1 else mask_image
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetConvergenceThreshold(float(convergence_threshold))
    corrector.SetMaximumNumberOfIterations([int(x) for x in max_iterations])
    corrector.Execute(working_image, working_mask)

    log_bias = corrector.GetLogBiasFieldAsImage(original)
    corrected = original / sitk.Exp(log_bias)

    # N4 的全局乘法尺度不是唯一的。这里恢复前景中位数，使 N4 前测量的
    # CSF 参考值仍与校正后图像处于相同强度尺度，N4 只负责消除低频空间偏置。
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


def load_csf_data(csv_path: str) -> Dict[str, Dict[str, Optional[float]]]:
    try:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(csv_path, encoding="gbk")
    if "patient_id" not in frame:
        raise KeyError("CSF file must contain patient_id")
    mean_column = "csf_mean" if "csf_mean" in frame else "csf_reference"
    if mean_column not in frame:
        raise KeyError("CSF file must contain csf_mean or csf_reference")

    result = {}
    for _, row in frame.iterrows():
        patient_id = canonical_patient_id(row["patient_id"])
        if not patient_id:
            raise ValueError("CSF file contains an empty patient_id")
        if patient_id in result:
            raise ValueError(f"Duplicate patient_id in CSF file: {row['patient_id']}")
        mean_value = pd.to_numeric(row[mean_column], errors="coerce")
        median_value = pd.to_numeric(row.get("csf_median", np.nan), errors="coerce")
        result[patient_id] = {
            "mean": None if pd.isna(mean_value) else float(mean_value),
            "median": None if pd.isna(median_value) else float(median_value),
        }
    return result


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
    for suffix in (".nii.gz", ".nii"):
        candidate = labels_dir / f"{patient_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = [
        path for path in labels_dir.iterdir()
        if path.is_file() and canonical_patient_id(patient_id_from_nifti(path)) == canonical_patient_id(patient_id)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one label for {patient_id}, found {len(matches)}")
    return matches[0]


def batch_process(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    csf_file: str,
    patient_list_file: str,
    patient_id_column: str = "patient_id",
    n4_shrink_factor: int = 4,
) -> int:
    images_root, labels_root = Path(images_dir), Path(labels_dir)
    requested = load_patient_ids(patient_list_file, patient_id_column)
    candidates = list(images_root.glob("*.nii")) + list(images_root.glob("*.nii.gz"))
    images, missing = filter_paths_to_patients(candidates, requested)
    if missing:
        names = [requested[key] for key in sorted(missing)]
        raise FileNotFoundError(f"Requested patients without MRI: {names}")
    csf_data = load_csf_data(csf_file)
    missing_csf = set(requested) - set(csf_data)
    if missing_csf:
        names = [requested[key] for key in sorted(missing_csf)]
        raise KeyError(f"Requested patients without CSF reference: {names}")

    logs = []
    for image_path in tqdm(images, desc="v7 图像预处理"):
        key = canonical_patient_id(patient_id_from_nifti(image_path))
        patient_id = requested[key]
        label_path = _find_label(labels_root, patient_id)
        info = csf_data[key]
        logs.append(process_single_patient(
            str(image_path), str(label_path), patient_id,
            str(Path(output_dir) / patient_id), info["mean"], info["median"],
            n4_shrink_factor,
        ))
    summary = {
        "version": "v7", "requested_count": len(requested),
        "success_count": len(logs), "patient_ids": [x["patient_id"] for x in logs],
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "batch_processing_log_v7.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(logs)


def main():
    """使用文件顶部的固定路径执行 311 人预处理。"""
    count = batch_process(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        csf_file=CSF_FILE,
        patient_list_file=PATIENT_LIST_FILE,
        patient_id_column="patient_id",
        n4_shrink_factor=4,
    )
    print(f"v7 预处理完成：成功处理 {count} 名目标患者。")


if __name__ == "__main__":
    main()
