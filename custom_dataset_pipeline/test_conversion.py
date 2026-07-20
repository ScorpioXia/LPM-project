"""
测试DICOM转换功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from convert_dicom_01 import read_dicom_series, create_affine_matrix, convert_dicom_series_to_nifti

# 测试参数
TEST_DIR = r"E:\dataset\yaozhui\QiLuhospital\2026年2月20日数据"
TEST_PATIENT = "陈秀英1"  # 选择一个测试病人
TEST_SERIES = 5  # 从Excel中获取的Series Number
OUTPUT_TEST = r"E:\dataset\yaozhui\QiLuhospital\test_output"

print("="*60)
print("测试DICOM转换功能")
print("="*60)

# 确保输出目录存在
Path(OUTPUT_TEST).mkdir(parents=True, exist_ok=True)

# 测试读取DICOM序列
print(f"\n测试读取DICOM序列: {TEST_PATIENT}")
patient_dir = Path(TEST_DIR) / TEST_PATIENT

# 测试1: 读取DICOM序列
dicom_volume, slice_positions, pixel_spacing, slice_thickness, image_position, image_orientation = read_dicom_series(patient_dir, TEST_SERIES)

if dicom_volume is not None:
    print(f"✓ 成功读取DICOM序列")
    print(f"  体积形状: {dicom_volume.shape}")
    print(f"  像素间距: {pixel_spacing}")
    print(f"  切片厚度: {slice_thickness}")
    print(f"  图像位置: {image_position}")
    print(f"  图像方向: {image_orientation}")
else:
    print(f"✗ 无法读取DICOM序列")

# 测试2: 创建affine矩阵
if dicom_volume is not None:
    print(f"\n测试创建affine矩阵")
    affine = create_affine_matrix(pixel_spacing, slice_thickness, image_position, image_orientation)
    print(f"✓ 成功创建affine矩阵")
    print(f"  Affine矩阵:")
    print(affine)

# 测试3: 完整转换
if dicom_volume is not None:
    print(f"\n测试完整转换")
    output_file = Path(OUTPUT_TEST) / f"{TEST_PATIENT}_0000.nii.gz"
    try:
        convert_dicom_series_to_nifti(patient_dir, TEST_SERIES, output_file)
        print(f"✓ 成功转换DICOM为NIfTI")
        print(f"  输出文件: {output_file}")
        
        # 验证文件是否存在
        if output_file.exists():
            print(f"✓ 文件已创建，大小: {output_file.stat().st_size / 1024:.2f} KB")
        else:
            print(f"✗ 文件未创建")
            
    except Exception as e:
        print(f"✗ 转换失败: {e}")

print(f"\n{'='*60}")
print("测试完成!")
print(f"{'='*60}")
