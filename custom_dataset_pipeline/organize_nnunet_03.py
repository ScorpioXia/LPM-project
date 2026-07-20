import os
import shutil
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import Counter


def validate_image_label_pair(image_path, label_path, expected_labels=None):
    """
    验证图像和标注配对
    """
    try:
        img = nib.load(str(image_path))
        lbl = nib.load(str(label_path))
        
        img_data = img.get_fdata()
        lbl_data = lbl.get_fdata()
        
        # 转换为整数标签
        lbl_data_int = lbl_data.astype(np.int32)
        
        errors = []
        label_info = {}
        
        if img_data.shape != lbl_data.shape:
            errors.append(f"形状不匹配: 图像 {img_data.shape}, 标注 {lbl_data.shape}")
        
        if not np.allclose(img.affine, lbl.affine):
            errors.append("Affine矩阵不匹配")
        
        unique_labels = np.unique(lbl_data_int)
        label_counts = Counter(lbl_data_int.flatten())
        label_info = {
            "unique_labels": sorted(unique_labels),
            "label_counts": dict(label_counts),
            "shape": lbl_data.shape
        }
        
        if expected_labels is not None:
            unexpected_labels = set(unique_labels) - set(expected_labels)
            if unexpected_labels:
                errors.append(f"意外的标签值: {sorted(unexpected_labels)}")
        
        is_valid = len(errors) == 0
        error_message = "; ".join(errors) if errors else None
        
        return is_valid, error_message, label_info
        
    except Exception as e:
        return False, str(e), None


def organize_to_nnunet(
    images_dir,
    labels_dir,
    nnunet_raw_dir,
    dataset_id,
    dataset_name,
    expected_labels=None,
    auto_validate=True
):
    """
    组织数据集到nnUNet v2格式
    
    Args:
        images_dir: 图像目录
        labels_dir: 标注目录
        nnunet_raw_dir: nnUNet_raw目录
        dataset_id: 数据集ID (3位数字)
        dataset_name: 数据集名称
        expected_labels: 预期的标签值列表
        auto_validate: 是否自动验证
    """
    dataset_dir = Path(nnunet_raw_dir) / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"
    
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir.mkdir(parents=True, exist_ok=True)
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    image_files = sorted(list(images_path.glob("*_0000.nii.gz")))
    label_files = sorted(list(labels_path.glob("*.nii.gz")) + list(labels_path.glob("*.nii")))
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"找到 {len(label_files)} 个标注文件")
    
    training_cases = []
    valid_cases = []
    invalid_cases = []
    
    for img_file in image_files:
        case_id = img_file.stem.replace("_0000", "")
        print(f"\n处理: {case_id}")
        
        label_file = labels_path / f"{case_id}.nii.gz"
        
        if not label_file.exists():
            label_file = labels_path / f"{case_id}.nii"
            if not label_file.exists():
                print(f"  ✗ 未找到标注文件，跳过")
                continue
        
        if auto_validate:
            is_valid, error_msg, label_info = validate_image_label_pair(
                img_file, label_file, expected_labels
            )
            
            if not is_valid:
                print(f"  ✗ 验证失败 - {error_msg}")
                invalid_cases.append({
                    "case_id": case_id,
                    "error": error_msg
                })
            else:
                print(f"  ✓ 验证通过")
                valid_cases.append({
                    "case_id": case_id,
                    "labels": label_info["unique_labels"]
                })
        
        dest_img = images_tr_dir / img_file.name
        dest_label = labels_tr_dir / f"{case_id}.nii.gz"
        
        shutil.copy2(img_file, dest_img)
        shutil.copy2(label_file, dest_label)
        
        training_cases.append({
            "image": f"./imagesTr/{img_file.name}",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })
        
        print(f"  ✓ 已处理")
    
    print(f"\n{'='*60}")
    print(f"组织完成!")
    print(f"{'='*60}")
    print(f"成功: {len(training_cases)} 个训练病例")
    
    if auto_validate:
        print(f"验证通过: {len(valid_cases)} 个")
        print(f"验证失败: {len(invalid_cases)} 个")
        
        if invalid_cases:
            print(f"\n验证失败的病例:")
            for case in invalid_cases[:10]:
                print(f"  - {case['case_id']}: {case['error']}")
            if len(invalid_cases) > 10:
                print(f"  ... 还有 {len(invalid_cases) - 10} 个")
    
    return dataset_dir, training_cases


def create_dataset_json(dataset_dir, training_cases, channel_names, labels, file_ending=".nii.gz"):
    """
    创建dataset.json文件
    """
    dataset_json = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": len(training_cases),
        "file_ending": file_ending
    }
    
    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)
    
    print(f"\n已创建: {json_path}")
    return json_path


if __name__ == "__main__":
    print("="*60)
    print("步骤3: 组织到nnUNet格式")
    print("="*60)
    
    print("\n请修改以下配置后运行:")
    print("""
    # 配置
    IMAGES_DIR = "path/to/your/images"
    LABELS_DIR = "path/to/your/labels"
    NNUNET_RAW_DIR = "path/to/nnUNet_raw"
    DATASET_ID = 101
    DATASET_NAME = "LumbarMuscle"
    
    # 预期的标签值（用于验证）
    EXPECTED_LABELS = [0, 1, 2, 3, 4, 5, 6]
    
    # 组织数据集
    dataset_dir, training_cases = organize_to_nnunet(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        nnunet_raw_dir=NNUNET_RAW_DIR,
        dataset_id=DATASET_ID,
        dataset_name=DATASET_NAME,
        expected_labels=EXPECTED_LABELS,
        auto_validate=True
    )
    
    # 创建dataset.json
    channel_names = {"0": "MRI"}
    labels = {
        "background": 0,
        "psoas_left": 1,
        "psoas_right": 2,
        "erector_spinae_left": 3,
        "erector_spinae_right": 4,
        "multifidus_left": 5,
        "multifidus_right": 6
    }
    
    create_dataset_json(
        dataset_dir=dataset_dir,
        training_cases=training_cases,
        channel_names=channel_names,
        labels=labels
    )
    """)
