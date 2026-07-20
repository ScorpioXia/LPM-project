
"""
全面分析数据集结构
"""
from pathlib import Path
import pandas as pd

# 数据集根目录
root_dir = Path(r"E:\dataset\yaozhui\QiLuhospital\2026年2月20日数据")
excel_path = root_dir / "SERIES_NUMBER.xlsx"

print("="*70)
print("数据集分析")
print("="*70)

# 1. 读取Excel
print("\n1. 读取Excel文件:")
print("-"*70)
if excel_path.exists():
    df = pd.read_excel(excel_path)
    print(f"Excel列名: {list(df.columns)}")
    print(f"Excel记录数: {len(df)}")
    print("\n前5行:")
    print(df.head())
    
    excel_names = set(str(row.iloc[0]).strip() for _, row in df.iterrows())
    print(f"\nExcel中的病人数: {len(excel_names)}")
else:
    print("未找到Excel文件!")
    excel_names = set()

# 2. 检查目录结构
print("\n2. 检查目录结构:")
print("-"*70)

all_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
print(f"根目录下的文件夹数: {len(all_dirs)}")

patient_dirs = []
other_dirs = []
for d in all_dirs:
    if d.name in ['images', 'labels']:
        other_dirs.append(d)
    else:
        patient_dirs.append(d)

print(f"病人文件夹: {len(patient_dirs)}")
print(f"其他文件夹: {[d.name for d in other_dirs]}")

# 3. 分析每个病人文件夹
print("\n3. 分析每个病人文件夹:")
print("-"*70)

stats = {
    'has_dicom_and_label': [],
    'has_only_dicom': [],
    'has_only_label': [],
    'has_nothing': [],
    'not_in_excel': []
}

for patient_dir in sorted(patient_dirs):
    name = patient_dir.name
    
    # 检查是否在Excel中
    in_excel = name in excel_names
    
    # 检查文件
    dicom_files = list(patient_dir.glob("*.dcm")) + list(patient_dir.glob("*.DCM"))
    label_files = list(patient_dir.glob("*.nii.gz")) + list(patient_dir.glob("*.nii"))
    
    has_dicom = len(dicom_files) > 0
    has_label = len(label_files) > 0
    
    status = ""
    if not in_excel:
        stats['not_in_excel'].append(name)
        status = " (不在Excel中)"
    elif has_dicom and has_label:
        stats['has_dicom_and_label'].append(name)
    elif has_dicom:
        stats['has_only_dicom'].append(name)
    elif has_label:
        stats['has_only_label'].append(name)
    else:
        stats['has_nothing'].append(name)
    
    print(f"{name}: DICOM={len(dicom_files)}, Label={len(label_files)}{status}")

# 4. 统计汇总
print("\n4. 统计汇总:")
print("-"*70)
print(f"有DICOM和标注: {len(stats['has_dicom_and_label'])} 个")
if stats['has_dicom_and_label']:
    print(f"  例如: {stats['has_dicom_and_label'][:5]}")

print(f"\n只有DICOM: {len(stats['has_only_dicom'])} 个")
if stats['has_only_dicom']:
    print(f"  例如: {stats['has_only_dicom'][:5]}")

print(f"\n只有标注: {len(stats['has_only_label'])} 个")
if stats['has_only_label']:
    print(f"  例如: {stats['has_only_label'][:5]}")

print(f"\n什么都没有: {len(stats['has_nothing'])} 个")
if stats['has_nothing']:
    print(f"  例如: {stats['has_nothing'][:5]}")

print(f"\n不在Excel中: {len(stats['not_in_excel'])} 个")
if stats['not_in_excel']:
    print(f"  例如: {stats['not_in_excel'][:5]}")

print("\n" + "="*70)
print("分析完成!")
print("="*70)
print("\n建议:")
print("1. 确认'只有标注'的病人是否有对应的DICOM图像")
print("2. 确认'不在Excel中'的病人是否需要处理")
print("3. 对于有DICOM和标注的病人，可以继续处理")
