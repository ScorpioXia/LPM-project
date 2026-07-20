"""
一键运行完整流程
请根据实际情况修改配置部分
"""

print("="*70)
print("自建数据集 → nnUNet v2 完整流程")
print("="*70)

print("\n请先编辑此文件，修改配置部分，然后运行!")
print("="*70)

# ==========================================
# 配置部分（请根据您的实际情况修改）
# ==========================================

# 步骤1: DICOM转换配置
INPUT_ROOT = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset\LS"  # 包含所有病人文件夹
EXCEL_PATH = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset\LS\series_number_851.xlsx"   # Excel文件
OUTPUT_IMAGES_DIR = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_output\images"  # 输出图像目录（注意：不要在INPUT_ROOT里面！）
OUTPUT_LABELS_DIR = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_output\labels"  # 输出标注目录（注意：不要在INPUT_ROOT里面！）

# 步骤2: 标签转换配置（如果需要）
# 如果标签已经是0-6，可以跳过此步
NEED_LABEL_CONVERSION = False
LABEL_MAPPING = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
CONVERTED_LABELS_DIR = r"E:\dataset\yaozhui\QiLuhospital\2026年2月20日数据\labels_converted"

# 步骤3: nnUNet组织配置
NNUNET_RAW_DIR = r"E:\code\nnUNet-master\nnUNet_raw"
DATASET_ID = 201
DATASET_NAME = "LumbarMuscle"
EXPECTED_LABELS = [0, 1, 2, 3, 4, 5, 6]

# ==========================================
# 流程开始
# ==========================================

print("\n" + "="*70)
print("步骤1: DICOM转换")
print("="*70)

from convert_dicom_01 import process_patient_dataset
process_patient_dataset(INPUT_ROOT, EXCEL_PATH, OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR)

print("\n" + "="*70)
print("步骤2: 标签检查")
print("="*70)

from check_and_convert_labels_02 import analyze_labels
label_stats = analyze_labels(OUTPUT_LABELS_DIR, max_files=10)

if NEED_LABEL_CONVERSION:
    print("\n" + "="*70)
    print("步骤2a: 标签转换")
    print("="*70)
    from check_and_convert_labels_02 import convert_labels
    convert_labels(OUTPUT_LABELS_DIR, CONVERTED_LABELS_DIR, LABEL_MAPPING)
    FINAL_LABELS_DIR = CONVERTED_LABELS_DIR
else:
    print("\n✓ 跳过标签转换（标签已正确）")
    FINAL_LABELS_DIR = OUTPUT_LABELS_DIR

print("\n" + "="*70)
print("步骤3: 组织到nnUNet格式 + 自动验证")
print("="*70)

from organize_nnunet_03 import organize_to_nnunet, create_dataset_json

dataset_dir, training_cases = organize_to_nnunet(
    images_dir=OUTPUT_IMAGES_DIR,
    labels_dir=FINAL_LABELS_DIR,
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

print("\n" + "="*70)
print("✓ 所有步骤完成!")
print("="*70)
print(f"\n数据集已创建: {dataset_dir}")
print("\n接下来可以运行:")
print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID}")
print(f"  nnUNetv2_train {DATASET_ID} 3d_fullres 0")
