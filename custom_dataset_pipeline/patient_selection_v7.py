"""v7 流水线使用的患者 ID 读取与严格匹配工具。

文件命名格式：
  - images: "{patient_id}_{姓名拼音}_0000.nii.gz"
  - labels: "{patient_id}_{姓名拼音}.nii.gz"

patient_id 为数字编号，是整个流水线的唯一标识符。
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd


def canonical_patient_id(value) -> str:
    """仅统一首尾空格和大小写，不进行模糊姓名匹配。"""
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def patient_id_from_nifti(path: Path) -> str:
    """从 NIfTI 文件名中提取 patient_id（下划线前的数字部分）。

    支持格式：
      - "P001_zhangsan_0000.nii.gz" -> "P001"
      - "P001_zhangsan.nii.gz" -> "P001"
      - "123_lisi.nii" -> "123"
    """
    name = path.name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    elif name.lower().endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    # 按下划线分割，取第一段作为 patient_id
    if "_" in name:
        name = name.split("_")[0]
    return name.strip()


def load_patient_list_with_csf(
    path: str,
    patient_id_col: int = 0,
    csf_col: int = 3,
    label_col: int = 4,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """从 PATIENT_LIST_FILE 读取病人列表和 CSF 值。

    Args:
        path: CSV 或 Excel 文件路径
        patient_id_col: patient_id 所在列索引（0-based，第1列）
        csf_col: CSF 值所在列索引（0-based，第4列）
        label_col: label 所在列索引（0-based，第5列）

    Returns:
        (patient_ids 映射, csf_values 映射)
        - patient_ids: {canonical_id: original_id}
        - csf_values: {canonical_id: csf_mean_value}
        label 为 #N/A 的病人会被跳过
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"患者名单不存在：{source}")

    suffix = source.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(source, header=None)
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(source, header=None, sep=sep, encoding="gbk")
    else:
        raise ValueError(f"不支持的患者名单格式：{suffix}")

    patient_ids: Dict[str, str] = {}
    csf_values: Dict[str, float] = {}
    skipped_na = []

    for _, row in frame.iterrows():
        raw_id = row.iloc[patient_id_col]
        if pd.isna(raw_id):
            continue

        original_id = str(raw_id).strip()
        canonical = canonical_patient_id(original_id)
        if not canonical:
            continue

        # 检查第5列（label）是否为 #N/A
        label_val = row.iloc[label_col] if label_col < len(row) else None
        if pd.isna(label_val):
            skipped_na.append(original_id)
            continue

        # 读取第4列 CSF 值
        csf_val = row.iloc[csf_col] if csf_col < len(row) else None
        try:
            csf_float = float(csf_val)
            if not np.isfinite(csf_float) or csf_float <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(
                f"患者 {original_id} 的 CSF 值无效: {csf_val}"
            )

        if canonical in patient_ids:
            raise ValueError(f"重复的 patient_id: {original_id}")

        patient_ids[canonical] = original_id
        csf_values[canonical] = csf_float

    print(f"从 PATIENT_LIST_FILE 加载了 {len(patient_ids)} 名有效患者")
    if skipped_na:
        print(f"  跳过 label 为 #N/A 的患者 {len(skipped_na)} 人")

    return patient_ids, csf_values


def load_patient_ids(
    path: Optional[str], id_column: str = "patient_id"
) -> Optional[Dict[str, str]]:
    """读取患者白名单，返回"标准化 ID → 原始 ID"的映射。"""
    if not path:
        return None
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"患者名单不存在：{source}")
    if source.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    elif source.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(source, sep="\t" if source.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError(f"不支持的患者名单格式：{source.suffix}")
    if id_column not in frame.columns:
        raise KeyError(f"患者名单缺少 {id_column!r} 列；现有列={list(frame.columns)}")

    result: Dict[str, str] = {}
    for raw in frame[id_column]:
        canonical = canonical_patient_id(raw)
        if not canonical:
            raise ValueError("患者名单中存在空 patient_id")
        original = str(raw).strip()
        if canonical in result:
            raise ValueError(f"标准化后出现重复 patient_id：{original}")
        result[canonical] = original
    return result


def filter_paths_to_patients(
    paths: Iterable[Path], patient_ids: Optional[Dict[str, str]]
):
    """为每个目标 ID 严格选择一个 NIfTI，并返回缺失 ID。"""
    paths = list(paths)
    if patient_ids is None:
        return sorted(paths), set()

    selected = {}
    for path in paths:
        canonical = canonical_patient_id(patient_id_from_nifti(path))
        if canonical not in patient_ids:
            continue
        if canonical in selected:
            raise ValueError(
                f"患者 {patient_ids[canonical]!r} 匹配到多个影像："
                f"{selected[canonical]} 和 {path}"
            )
        selected[canonical] = path
    missing = set(patient_ids) - set(selected)
    return sorted(selected.values()), missing
