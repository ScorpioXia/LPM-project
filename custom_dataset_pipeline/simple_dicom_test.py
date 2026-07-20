
"""
简单的DICOM文件测试
"""
from pathlib import Path
import pydicom

# 测试一个有DICOM文件的病人
test_patient = "曹怀月"
test_dir = Path(r"E:\dataset\yaozhui\QiLuhospital\2026年2月20日数据") / test_patient

print(f"测试病人: {test_patient}")
print(f"目录: {test_dir}")
print("="*70)

# 列出所有文件
print("\n目录中的所有文件:")
all_files = list(test_dir.iterdir())
for f in sorted(all_files):
    print(f"  {f.name}")

# 查找.dcm和.DCM文件
print("\n\n查找DICOM文件:")
dicom_files = []
for f in test_dir.iterdir():
    if f.is_file():
        # 检查扩展名
        if f.suffix.lower() in ['.dcm']:
            dicom_files.append(f)
            print(f"  ✓ 找到: {f.name}")

print(f"\n总共找到 {len(dicom_files)} 个DICOM文件")

if dicom_files:
    print("\n\n尝试读取第一个文件:")
    try:
        first_file = dicom_files[0]
        print(f"文件: {first_file.name}")
        
        ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
        print(f"  ✓ 读取成功!")
        
        print(f"\n文件信息:")
        print(f"  SeriesNumber: {getattr(ds, 'SeriesNumber', 'N/A')}")
        print(f"  SeriesDescription: {getattr(ds, 'SeriesDescription', 'N/A')}")
        print(f"  PatientName: {getattr(ds, 'PatientName', 'N/A')}")
        print(f"  Rows: {getattr(ds, 'Rows', 'N/A')}")
        print(f"  Columns: {getattr(ds, 'Columns', 'N/A')}")
        
        # 检查几个文件的SeriesNumber
        print(f"\n检查前10个文件的SeriesNumber:")
        series_numbers = set()
        for f in dicom_files[:10]:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                sn = getattr(ds, 'SeriesNumber', None)
                if sn is not None:
                    series_numbers.add(int(sn))
            except:
                pass
        
        print(f"  发现的SeriesNumber: {sorted(series_numbers)}")
        
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        import traceback
        traceback.print_exc()
