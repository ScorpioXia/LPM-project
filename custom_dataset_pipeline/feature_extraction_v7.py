"""六块腰椎肌肉的特征提取脚本 v7。

以 patient_id 为唯一标识符：
  - 预处理输出目录: "{preprocessed_dir}/{patient_id}/"
  - 标签文件: "{patient_id}_{姓名拼音}.nii.gz"
  - 特征表格第一列为 patient_id
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from fat_threshold_v7 import get_fat_threshold_per_muscle
from muscle_feature_calculator_v7 import (
    calculate_3d_features,
    calculate_all_muscle_features,
    calculate_cross_layer_gradient_features,
    calculate_multi_muscle_features,
)
from patient_selection_v7 import (
    canonical_patient_id,
    load_patient_list_with_csf,
    patient_id_from_nifti,
)


# ==================== 固定路径配置 ====================
PREPROCESSED_DIR = r"D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\output_v7_20260721"
LABELS_DIR = r"D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\labels"
OUTPUT_DIR = r"D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\features_v7_20260721"
PATIENT_LIST_FILE = r"D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\PATIENT_LIST_FILE.xlsx"


MUSCLE_LABELS = {
    1: "psoas_left",
    2: "psoas_right",
    3: "multifidus_left",
    4: "multifidus_right",
    5: "erector_spinae_left",
    6: "erector_spinae_right",
}


def calculate_features(image_slice, label_slice, muscle_label, pixel_spacing):
    muscle_mask = label_slice == muscle_label
    muscle_pixels = image_slice[muscle_mask]
    threshold, decision = get_fat_threshold_per_muscle(muscle_pixels)
    features = calculate_all_muscle_features(
        image_slice, muscle_mask, pixel_spacing, threshold
    )
    features.update({
        "pixel_spacing_x": float(pixel_spacing[0]),
        "pixel_spacing_y": float(pixel_spacing[1]),
        "muscle_area_mm2": features["Area"],
        "Total_CSA": int(np.sum(muscle_mask)),
        "fat_threshold_used": threshold,
        "fat_threshold_decision": decision,
    })
    return features


def _load_image_and_label(image_path, label_path):
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    image_data = image.get_fdata(dtype=np.float32)
    label_data = label.get_fdata().astype(np.int16)
    if image_data.shape != label_data.shape:
        raise ValueError(f"Image shape {image_data.shape} != label shape {label_data.shape}")
    present_labels = set(np.unique(label_data).astype(int))
    missing_labels = set(MUSCLE_LABELS) - present_labels
    if missing_labels:
        missing_names = [MUSCLE_LABELS[value] for value in sorted(missing_labels)]
        raise ValueError(f"分割标签缺少肌肉类别：{missing_names}")
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
    return image_data, label_data, spacing


def extract_patient(image_path, label_path, patient_id, csf_value=None):
    image, label, spacing = _load_image_and_label(image_path, label_path)
    pixel_spacing, slice_thickness = spacing[:2], spacing[2]
    rows_2d = []
    for z in range(image.shape[2]):
        for muscle_label, muscle_name in MUSCLE_LABELS.items():
            if not np.any(label[:, :, z] == muscle_label):
                continue
            row = calculate_features(image[:, :, z], label[:, :, z], muscle_label, pixel_spacing)
            row.update({"patient_id": patient_id, "slice_index": z, "muscle_name": muscle_name})
            if csf_value is not None:
                row["csf_value"] = float(csf_value)
            rows_2d.append(row)

    rows_3d, rows_cross = [], []
    for muscle_label, muscle_name in MUSCLE_LABELS.items():
        row_3d = calculate_3d_features(
            image, label, muscle_label, pixel_spacing, slice_thickness
        )
        row_3d.update({
            "patient_id": patient_id, "muscle_name": muscle_name,
            "pixel_spacing_x": pixel_spacing[0], "pixel_spacing_y": pixel_spacing[1],
            "slice_thickness": slice_thickness,
        })
        if csf_value is not None:
            row_3d["csf_value"] = float(csf_value)
        rows_3d.append(row_3d)

        row_cross = calculate_cross_layer_gradient_features(
            image, label, muscle_label, pixel_spacing, slice_thickness
        )
        row_cross.update({"patient_id": patient_id, "muscle_name": muscle_name})
        if csf_value is not None:
            row_cross["csf_value"] = float(csf_value)
        rows_cross.append(row_cross)

    row_multi = calculate_multi_muscle_features(
        image, label, pixel_spacing, slice_thickness
    )
    row_multi["patient_id"] = patient_id
    if csf_value is not None:
        row_multi["csf_value"] = float(csf_value)
    return rows_2d, rows_3d, rows_cross, row_multi


def _find_normalized(patient_dir: Path, patient_id: str) -> Path:
    exact = patient_dir / f"{patient_id}_normalized.nii.gz"
    if exact.exists():
        return exact
    matches = list(patient_dir.glob("*_normalized.nii.gz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one normalized image in {patient_dir}, found {len(matches)}"
        )
    return matches[0]


def _find_label(labels_dir: Path, patient_id: str) -> Path:
    """根据 patient_id 在 labels 目录查找匹配的标签文件。"""
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


def check_csv_consistency(
    successful_patients: List[str],
    all_2d: List[Dict],
    all_3d: List[Dict],
    all_cross: List[Dict],
    all_multi: List[Dict],
) -> Dict:
    """检查四类特征表格的病人一致性。"""
    result = {
        "expected_patients": set(successful_patients),
        "csv_patients_2d": set(),
        "csv_patients_3d": set(),
        "csv_patients_level3_cross": set(),
        "csv_patients_level3_multi": set(),
        "missing_in_2d": set(),
        "missing_in_3d": set(),
        "missing_in_level3_cross": set(),
        "missing_in_level3_multi": set(),
        "inconsistent_between_csv": {},
    }

    if all_2d:
        result["csv_patients_2d"] = set(pd.DataFrame(all_2d)["patient_id"].unique())
    if all_3d:
        result["csv_patients_3d"] = set(pd.DataFrame(all_3d)["patient_id"].unique())
    if all_cross:
        result["csv_patients_level3_cross"] = set(
            pd.DataFrame(all_cross)["patient_id"].unique()
        )
    if all_multi:
        result["csv_patients_level3_multi"] = set(
            pd.DataFrame(all_multi)["patient_id"].unique()
        )

    result["missing_in_2d"] = result["expected_patients"] - result["csv_patients_2d"]
    result["missing_in_3d"] = result["expected_patients"] - result["csv_patients_3d"]
    result["missing_in_level3_cross"] = (
        result["expected_patients"] - result["csv_patients_level3_cross"]
    )
    result["missing_in_level3_multi"] = (
        result["expected_patients"] - result["csv_patients_level3_multi"]
    )

    csv_keys = [
        "csv_patients_2d",
        "csv_patients_3d",
        "csv_patients_level3_cross",
        "csv_patients_level3_multi",
    ]
    csv_labels = ["2D特征", "3D特征", "Level3.1特征", "Level3.2-3.5特征"]

    for i in range(len(csv_keys)):
        for j in range(i + 1, len(csv_keys)):
            set_i = result[csv_keys[i]]
            set_j = result[csv_keys[j]]
            if set_i != set_j:
                key = f"{csv_labels[i]} vs {csv_labels[j]}"
                result["inconsistent_between_csv"][key] = {
                    f"仅在{csv_labels[i]}": sorted(set_i - set_j),
                    f"仅在{csv_labels[j]}": sorted(set_j - set_i),
                }

    return result


def validate_patient_consistency(
    patient_ids: Dict[str, str],
    csf_values: Dict[str, float],
    preprocessed_dir: Path,
    labels_dir: Path,
) -> Dict:
    """验证 patient_id、CSF 值、预处理图像和标签文件的一致性。

    Args:
        patient_ids: {canonical_id: original_id} 映射
        csf_values: {canonical_id: csf_value} 映射
        preprocessed_dir: 预处理后图像根目录
        labels_dir: 标签文件目录

    Returns:
        验证结果字典
    """
    result = {
        "total_in_patient_list": len(patient_ids),
        "with_csf_value": 0,
        "missing_csf_value": [],
        "preprocessed_dirs_found": 0,
        "missing_preprocessed": [],
        "label_files_found": 0,
        "missing_labels": [],
        "csf_range": {"min": None, "max": None, "mean": None},
    }

    csf_list = []
    for key in sorted(patient_ids.keys()):
        patient_id = patient_ids[key]
        if key in csf_values and csf_values[key] is not None:
            result["with_csf_value"] += 1
            csf_list.append(csf_values[key])
        else:
            result["missing_csf_value"].append(patient_id)

    if csf_list:
        result["csf_range"]["min"] = float(min(csf_list))
        result["csf_range"]["max"] = float(max(csf_list))
        result["csf_range"]["mean"] = float(np.mean(csf_list))

    preprocessed_root = Path(preprocessed_dir)
    labels_root = Path(labels_dir)

    available_dirs = {
        canonical_patient_id(path.name): path
        for path in preprocessed_root.iterdir() if path.is_dir()
    }
    label_candidates = list(labels_root.glob("*.nii.gz")) + list(labels_root.glob("*.nii"))
    label_ids = {
        canonical_patient_id(patient_id_from_nifti(p)) for p in label_candidates
    }

    for key in sorted(patient_ids.keys()):
        patient_id = patient_ids[key]
        if key in available_dirs:
            result["preprocessed_dirs_found"] += 1
        else:
            result["missing_preprocessed"].append(patient_id)
        if key in label_ids:
            result["label_files_found"] += 1
        else:
            result["missing_labels"].append(patient_id)

    return result


def save_extraction_report(
    output_dir: str,
    successful_patients: List[str],
    failed_patients: List[Dict],
    total_patients: int,
    consistency_result: Optional[Dict] = None,
    validation_result: Optional[Dict] = None,
) -> Path:
    """保存特征提取检测报告（v7 版本）。"""
    report_path = Path(output_dir) / "extraction_report_v7.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("肌肉特征提取检测报告 (版本 v7)\n")
        f.write("说明：基于 CSF 信号标准化的图像，脂肪阈值采用自适应 Otsu + 固定回退 0.6\n")
        f.write("索引方式：以 patient_id 为唯一标识符\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if validation_result:
            f.write("-" * 80 + "\n")
            f.write("patient_id 一致性验证\n")
            f.write("-" * 80 + "\n")
            f.write(f"PATIENT_LIST_FILE 中病人数: {validation_result['total_in_patient_list']}\n")
            f.write(f"  有 CSF 值的病人: {validation_result['with_csf_value']}\n")
            f.write(f"  缺少 CSF 值的病人: {len(validation_result['missing_csf_value'])}\n")
            if validation_result["missing_csf_value"]:
                f.write(f"    列表: {validation_result['missing_csf_value'][:10]}")
                if len(validation_result["missing_csf_value"]) > 10:
                    f.write(f" 等 {len(validation_result['missing_csf_value'])} 人")
                f.write("\n")
            f.write(f"找到预处理目录的病人: {validation_result['preprocessed_dirs_found']}\n")
            f.write(f"  缺少预处理目录的病人: {len(validation_result['missing_preprocessed'])}\n")
            if validation_result["missing_preprocessed"]:
                f.write(f"    列表: {validation_result['missing_preprocessed'][:10]}")
                if len(validation_result["missing_preprocessed"]) > 10:
                    f.write(f" 等 {len(validation_result['missing_preprocessed'])} 人")
                f.write("\n")
            f.write(f"找到标签文件的病人: {validation_result['label_files_found']}\n")
            f.write(f"  缺少标签文件的病人: {len(validation_result['missing_labels'])}\n")
            if validation_result["missing_labels"]:
                f.write(f"    列表: {validation_result['missing_labels'][:10]}")
                if len(validation_result["missing_labels"]) > 10:
                    f.write(f" 等 {len(validation_result['missing_labels'])} 人")
                f.write("\n")
            if validation_result["csf_range"]["min"] is not None:
                f.write(f"CSF 值范围: [{validation_result['csf_range']['min']:.2f}, {validation_result['csf_range']['max']:.2f}]\n")
                f.write(f"CSF 平均值: {validation_result['csf_range']['mean']:.2f}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("总体统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"总病人数: {total_patients}\n")
        success_rate = len(successful_patients) / total_patients * 100 if total_patients > 0 else 0
        fail_rate = len(failed_patients) / total_patients * 100 if total_patients > 0 else 0
        f.write(f"成功提取: {len(successful_patients)} ({success_rate:.1f}%)\n")
        f.write(f"失败提取: {len(failed_patients)} ({fail_rate:.1f}%)\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("成功提取特征的病人（按 patient_id 排序）\n")
        f.write("-" * 80 + "\n")
        if successful_patients:
            for i, patient_id in enumerate(sorted(successful_patients), 1):
                f.write(f"{i}. {patient_id}\n")
        else:
            f.write("无\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("未能提取特征的病人\n")
        f.write("-" * 80 + "\n")
        if failed_patients:
            for i, fail_info in enumerate(failed_patients, 1):
                f.write(f'{i}. {fail_info["patient_id"]}\n')
                f.write(f'   失败原因: {fail_info["reason"]}\n')
                if "error_type" in fail_info:
                    f.write(f'   错误类型: {fail_info["error_type"]}\n')
                if "suggestion" in fail_info:
                    f.write(f'   处理建议: {fail_info["suggestion"]}\n')
                f.write("\n")
        else:
            f.write("无\n")
        f.write("\n")

        if consistency_result:
            f.write("-" * 80 + "\n")
            f.write("四类特征表格病人一致性检查\n")
            f.write("-" * 80 + "\n")

            f.write("\n各表格病人数统计:\n")
            f.write(f'  预期成功病人数: {len(consistency_result["expected_patients"])}\n')
            f.write(f'  2D特征表格病人数: {len(consistency_result["csv_patients_2d"])}\n')
            f.write(f'  3D特征表格病人数: {len(consistency_result["csv_patients_3d"])}\n')
            f.write(f'  Level3.1特征表格病人数: {len(consistency_result["csv_patients_level3_cross"])}\n')
            f.write(f'  Level3.2-3.5特征表格病人数: {len(consistency_result["csv_patients_level3_multi"])}\n')

            has_missing = (
                consistency_result["missing_in_2d"]
                or consistency_result["missing_in_3d"]
                or consistency_result["missing_in_level3_cross"]
                or consistency_result["missing_in_level3_multi"]
            )

            if has_missing:
                f.write("\n各表格缺失病人情况（相对于成功提取列表）:\n")
                if consistency_result["missing_in_2d"]:
                    f.write(f'  2D特征缺失: {sorted(consistency_result["missing_in_2d"])}\n')
                if consistency_result["missing_in_3d"]:
                    f.write(f'  3D特征缺失: {sorted(consistency_result["missing_in_3d"])}\n')
                if consistency_result["missing_in_level3_cross"]:
                    f.write(
                        f'  Level3.1特征缺失: {sorted(consistency_result["missing_in_level3_cross"])}\n'
                    )
                if consistency_result["missing_in_level3_multi"]:
                    f.write(
                        f'  Level3.2-3.5特征缺失: {sorted(consistency_result["missing_in_level3_multi"])}\n'
                    )
            else:
                f.write("\n各表格无缺失病人\n")

            if consistency_result["inconsistent_between_csv"]:
                f.write("\n表格间不一致情况:\n")
                for compare_key, diff_info in consistency_result[
                    "inconsistent_between_csv"
                ].items():
                    f.write(f"  {compare_key}:\n")
                    for diff_label, patients in diff_info.items():
                        if patients:
                            f.write(f"    {diff_label}: {patients}\n")
            else:
                f.write("\n所有表格间病人ID一致\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")

    return report_path


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 patient_id 移到第一列。"""
    if "patient_id" in df.columns:
        cols = ["patient_id"] + [c for c in df.columns if c != "patient_id"]
        df = df[cols]
    return df


def batch_extract(
    preprocessed_dir,
    labels_dir,
    output_dir,
    patient_list_file,
    continue_on_error: bool = True,
    generate_report: bool = True,
):
    """批量提取特征（以 patient_id 为索引）。

    Args:
        preprocessed_dir: 预处理后图像目录
        labels_dir: 分割标签目录
        output_dir: 输出目录
        patient_list_file: 病人列表 Excel
        continue_on_error: 单个病人失败时是否继续（默认 True）
        generate_report: 是否生成检测报告和一致性检查（默认 True）

    Returns:
        汇总字典
    """
    patient_ids, csf_values = load_patient_list_with_csf(patient_list_file)
    preprocessed_root, labels_root, output_root = map(
        Path, (preprocessed_dir, labels_dir, output_dir)
    )
    available = {
        canonical_patient_id(path.name): path
        for path in preprocessed_root.iterdir() if path.is_dir()
    }
    missing = set(patient_ids) - set(available)
    if missing:
        names = [patient_ids[key] for key in sorted(missing)]
        raise FileNotFoundError(f"缺少预处理数据的患者: {names}")

    successful_patients: List[str] = []
    failed_patients: List[Dict] = []
    all_2d, all_3d, all_cross, all_multi = [], [], [], []

    for key in tqdm(sorted(patient_ids.keys()), desc="v7 特征提取"):
        patient_id = patient_ids[key]
        patient_dir = available[key]
        csf_value = csf_values.get(key)
        try:
            image_path = _find_normalized(patient_dir, patient_id)
            label_path = _find_label(labels_root, patient_id)
            rows_2d, rows_3d, rows_cross, row_multi = extract_patient(
                image_path, label_path, patient_id, csf_value=csf_value
            )
            all_2d.extend(rows_2d)
            all_3d.extend(rows_3d)
            all_cross.extend(rows_cross)
            all_multi.append(row_multi)
            successful_patients.append(patient_id)
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            failed_patients.append({
                "patient_id": patient_id,
                "reason": error_msg,
                "error_type": error_type,
            })
            tqdm.write(f"  ⚠️  {patient_id} 提取失败: {error_type} - {error_msg}")
            if not continue_on_error:
                raise

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "2d": (all_2d, "muscle_features_2d_v7.csv"),
        "3d": (all_3d, "muscle_features_3d_v7.csv"),
        "level3_cross": (all_cross, "muscle_features_level3_cross_v7.csv"),
        "level3_multi": (all_multi, "muscle_features_level3_multi_v7.csv"),
    }
    for rows, filename in outputs.values():
        df = pd.DataFrame(rows)
        df = _reorder_columns(df)
        df.to_csv(output_root / filename, index=False, encoding="utf-8-sig")

    consistency_result = None
    report_path = None
    validation_result = None
    if generate_report:
        validation_result = validate_patient_consistency(
            patient_ids, csf_values, preprocessed_root, labels_root
        )
        tqdm.write(f"patient_id 一致性验证完成")
        tqdm.write(f"  PATIENT_LIST_FILE: {validation_result['total_in_patient_list']} 人")
        tqdm.write(f"  有 CSF 值: {validation_result['with_csf_value']} 人")
        tqdm.write(f"  预处理目录: {validation_result['preprocessed_dirs_found']} 人")
        tqdm.write(f"  标签文件: {validation_result['label_files_found']} 人")

    if generate_report and successful_patients:
        consistency_result = check_csv_consistency(
            successful_patients, all_2d, all_3d, all_cross, all_multi
        )
        report_path = save_extraction_report(
            str(output_root),
            successful_patients,
            failed_patients,
            len(patient_ids),
            consistency_result,
            validation_result,
        )
        tqdm.write(f"检测报告已保存: {report_path}")

    summary = {
        "version": "v7",
        "total_requested": len(patient_ids),
        "success_count": len(successful_patients),
        "failed_count": len(failed_patients),
        "successful_patients": successful_patients,
        "failed_patients": failed_patients,
        "row_counts": {key: len(rows) for key, (rows, _) in outputs.items()},
        "texture_bin_width": 0.025,
        "fat_threshold_policy": "2D v6 shared across all feature levels",
        "index_by": "patient_id",
        "continue_on_error": continue_on_error,
        "report_generated": report_path is not None,
        "validation": validation_result,
    }
    (output_root / "feature_extraction_log_v7.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    """使用文件顶部的固定路径提取目标患者的特征。"""
    summary = batch_extract(
        preprocessed_dir=PREPROCESSED_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        patient_list_file=PATIENT_LIST_FILE,
        continue_on_error=True,
        generate_report=True,
    )
    print(f"\n特征提取完成：成功 {summary['success_count']}/{summary['total_requested']} 人")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
