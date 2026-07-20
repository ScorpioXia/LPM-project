"""v7 流水线使用的患者 ID 读取与严格匹配工具。"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import pandas as pd


def canonical_patient_id(value) -> str:
    """仅统一首尾空格和大小写，不进行模糊姓名匹配。"""
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def patient_id_from_nifti(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    elif name.lower().endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name.strip()


def load_patient_ids(path: Optional[str], id_column: str = "patient_id") -> Optional[Dict[str, str]]:
    """读取患者白名单，返回“标准化 ID → 原始 ID”的映射。"""
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


def filter_paths_to_patients(paths: Iterable[Path], patient_ids: Optional[Dict[str, str]]):
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
