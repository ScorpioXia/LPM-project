import os
import re
import pydicom
from pydicom import uid
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def read_excel_series_info(excel_path):
    """
    读取Excel文件，获取病人姓名和对应的series number
    
    Args:
        excel_path: Excel文件路径
        
    Returns:
        dict: {病人姓名: series_number}
    """
    df = pd.read_excel(excel_path)
    print(f"读取Excel文件: {excel_path}")
    print(f"共 {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")
    
    series_info = {}
    for _, row in df.iterrows():
        name = str(row.iloc[0]).strip()
        series_num = int(row.iloc[1])
        series_info[name] = series_num
    
    print(f"\n成功读取 {len(series_info)} 个病人的series信息")
    return series_info


def read_dicom_series(dicom_dir, series_number):
    """
    读取指定Series Number的DICOM序列
    确保正确处理DICOM元数据和切片排序
    
    Args:
        dicom_dir: DICOM文件所在目录
        series_number: 指定的Series Number
        
    Returns:
        tuple: (dicom_volume, slice_positions, pixel_spacing, slice_thickness, image_position, image_orientation)
    """
    dicom_files = list(Path(dicom_dir).glob("*.dcm"))
    if not dicom_files:
        print(f"  未找到任何DICOM文件")
        return None, None, None, None, None, None
    
    print(f"  找到 {len(dicom_files)} 个DICOM文件")
    
    # 收集指定Series Number的DICOM文件
    selected_files = []
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(str(dicom_file), force=True)
            # 处理缺失的TransferSyntaxUID
            if not hasattr(ds, 'TransferSyntaxUID'):
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            
            # 检查Series Number
            if hasattr(ds, 'SeriesNumber'):
                sn = int(ds.SeriesNumber)
                if sn == series_number:
                    selected_files.append(dicom_file)
        except Exception as e:
            print(f"  警告: 无法读取文件 {dicom_file.name}: {e}")
            continue
    
    if not selected_files:
        print(f"  未找到Series Number为 {series_number} 的DICOM文件")
        return None, None, None, None, None, None
    
    print(f"  找到 {len(selected_files)} 个Series {series_number} 的DICOM文件")
    
    # 读取并处理DICOM切片
    slices = []
    slice_positions = []
    
    for dicom_file in selected_files:
        try:
            ds = pydicom.dcmread(str(dicom_file), force=True)
            # 处理缺失的TransferSyntaxUID
            if not hasattr(ds, 'TransferSyntaxUID'):
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            
            slices.append(ds)
            # 获取切片位置（Z轴坐标）
            if hasattr(ds, 'ImagePositionPatient'):
                slice_positions.append(float(ds.ImagePositionPatient[2]))
            elif hasattr(ds, 'SliceLocation'):
                slice_positions.append(float(ds.SliceLocation))
            else:
                slice_positions.append(0.0)
        except Exception as e:
            print(f"  警告: 无法处理文件 {dicom_file.name}: {e}")
            continue
    
    if not slices:
        print(f"  没有成功读取任何DICOM切片")
        return None, None, None, None, None, None
    
    # 按切片位置排序
    sorted_indices = np.argsort(slice_positions)
    slices = [slices[i] for i in sorted_indices]
    slice_positions = [slice_positions[i] for i in sorted_indices]
    
    # 获取第一个切片的元数据
    first_slice = slices[0]
    
    # 获取像素间距
    pixel_spacing = [1.0, 1.0]
    if hasattr(first_slice, 'PixelSpacing') and len(first_slice.PixelSpacing) >= 2:
        pixel_spacing = [float(first_slice.PixelSpacing[0]), float(first_slice.PixelSpacing[1])]
    
    # 获取切片厚度
    slice_thickness = pixel_spacing[0]
    if hasattr(first_slice, 'SliceThickness'):
        slice_thickness = float(first_slice.SliceThickness)
    
    # 获取图像位置
    image_position = [0.0, 0.0, 0.0]
    if hasattr(first_slice, 'ImagePositionPatient'):
        image_position = [float(x) for x in first_slice.ImagePositionPatient]
    
    # 获取图像方向
    image_orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if hasattr(first_slice, 'ImageOrientationPatient'):
        image_orientation = [float(x) for x in first_slice.ImageOrientationPatient]
    
    # 创建图像数组
    rows = int(first_slice.Rows)
    cols = int(first_slice.Columns)
    depth = len(slices)
    
    image_array = np.zeros((depth, rows, cols), dtype=np.int16)
    
    valid_count = 0
    for i, slice_ds in enumerate(slices):
        try:
            # 检查尺寸是否匹配
            if int(slice_ds.Rows) != rows or int(slice_ds.Columns) != cols:
                print(f"  警告: 切片 {i} 尺寸不匹配 ({slice_ds.Rows}x{slice_ds.Columns} != {rows}x{cols})，跳过")
                continue
            
            # 读取像素数据
            pixel_data = slice_ds.PixelData
            bits_allocated = getattr(slice_ds, 'BitsAllocated', 16)
            pixel_representation = getattr(slice_ds, 'PixelRepresentation', 0)
            
            if bits_allocated == 16:
                if pixel_representation == 1:
                    dtype = np.int16
                else:
                    dtype = np.uint16
                
                img_slice = np.frombuffer(pixel_data, dtype=dtype)
                img_slice = img_slice.reshape((rows, cols))
                image_array[i, :, :] = img_slice.astype(np.int16)
                valid_count += 1
            else:
                print(f"  警告: 不支持的BitsAllocated: {bits_allocated}")
                continue
                
        except Exception as e:
            print(f"  警告: 无法处理切片 {i}: {e}")
            continue
    
    print(f"  成功处理 {valid_count}/{len(slices)} 个切片")
    
    if valid_count == 0:
        print(f"  没有有效的切片数据")
        return None, None, None, None, None, None
    
    return image_array, slice_positions, pixel_spacing, slice_thickness, image_position, image_orientation


def create_affine_matrix(pixel_spacing, slice_thickness, image_position, image_orientation):
    """
    创建正确的affine矩阵
    
    Args:
        pixel_spacing: 像素间距 [x, y]
        slice_thickness: 切片厚度
        image_position: 图像位置 [x, y, z]
        image_orientation: 图像方向 [x1, y1, z1, x2, y2, z2]
        
    Returns:
        np.ndarray: 4x4 affine矩阵
    """
    # 计算方向向量
    row_vector = np.array(image_orientation[:3])
    col_vector = np.array(image_orientation[3:])
    slice_vector = np.cross(row_vector, col_vector)
    
    # 计算spacing
    row_spacing = pixel_spacing[0]
    col_spacing = pixel_spacing[1]
    slice_spacing = slice_thickness
    
    # 构建affine矩阵
    affine = np.eye(4)
    affine[0, 0] = row_vector[0] * row_spacing
    affine[1, 0] = row_vector[1] * row_spacing
    affine[2, 0] = row_vector[2] * row_spacing
    
    affine[0, 1] = col_vector[0] * col_spacing
    affine[1, 1] = col_vector[1] * col_spacing
    affine[2, 1] = col_vector[2] * col_spacing
    
    affine[0, 2] = slice_vector[0] * slice_spacing
    affine[1, 2] = slice_vector[1] * slice_spacing
    affine[2, 2] = slice_vector[2] * slice_spacing
    
    affine[0, 3] = image_position[0]
    affine[1, 3] = image_position[1]
    affine[2, 3] = image_position[2]
    
    return affine


def convert_dicom_series_to_nifti(dicom_dir, series_number, output_file):
    """
    将DICOM序列转换为NIfTI文件
    确保正确处理元数据，避免数据拉伸
    """
    # 确保输出目录存在
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取DICOM序列
    dicom_volume, slice_positions, pixel_spacing, slice_thickness, image_position, image_orientation = read_dicom_series(dicom_dir, series_number)
    
    if dicom_volume is None:
        raise RuntimeError("无法读取DICOM序列")
    
    # 转换维度顺序为 (x, y, z) 以匹配标注格式
    # 原始形状: (z, y, x)
    # 目标形状: (x, y, z)
    dicom_volume = np.transpose(dicom_volume, (2, 1, 0))
    
    # 翻转x轴数据以纠正左右方向（L->R）
    # 翻转y轴数据以纠正前后方向（P->A）
    # 这样可以确保图像内容正确，同时保持与标注的匹配
    dicom_volume = np.flip(dicom_volume, axis=0)  # 翻转x轴
    dicom_volume = np.flip(dicom_volume, axis=1)  # 翻转y轴
    
    # 创建affine矩阵，保持正常方向
    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[0]  # x方向
    affine[1, 1] = pixel_spacing[1]  # y方向
    affine[2, 2] = slice_thickness    # z方向
    
    # 图像位置
    affine[0, 3] = image_position[0]
    affine[1, 3] = image_position[1]
    affine[2, 3] = image_position[2]
    
    # 保存为NIfTI文件
    img = nib.Nifti1Image(dicom_volume, affine)
    nib.save(img, str(output_file))
    
    print(f"  ✓ 已保存: {output_file.name}")
    print(f"    数组形状: {dicom_volume.shape} (x, y, z)")
    print(f"    像素间距: {pixel_spacing}")
    print(f"    切片厚度: {slice_thickness}")


def process_patient_dataset(input_root, excel_path, output_images_dir, output_labels_dir):
    """
    处理整个数据集
    
    Args:
        input_root: 输入根目录（包含所有病人文件夹）
        excel_path: Excel文件路径
        output_images_dir: 输出图像目录
        output_labels_dir: 输出标签目录
    """
    Path(output_images_dir).mkdir(parents=True, exist_ok=True)
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)
    
    series_info = read_excel_series_info(excel_path)
    
    input_path = Path(input_root)
    
    # 更严格的过滤：只处理Excel中存在的文件夹
    all_items = list(input_path.iterdir())
    patient_dirs = []
    
    print(f"\n输入目录中共有 {len(all_items)} 个项目")
    
    for item in all_items:
        if not item.is_dir():
            continue  # 跳过文件
            
        item_name = item.name
        
        # 跳过已知的非病人文件夹
        if item_name in ['images', 'labels', 'nunet_prepared', 'labels_converted']:
            continue
            
        # 跳过隐藏文件夹
        if item_name.startswith('.'):
            continue
            
        # 只处理Excel中存在的病人
        if item_name in series_info:
            patient_dirs.append(item)
    
    print(f"识别出 {len(patient_dirs)} 个病人文件夹（在Excel中）")
    
    success_count = 0
    skip_count = 0
    
    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        
        print(f"\n处理: {patient_name}")
        
        target_series = series_info[patient_name]
        
        # 先检查是否有标注文件
        label_files = list(patient_dir.glob("*.nii.gz")) + list(patient_dir.glob("*.nii"))
        
        if not label_files:
            print(f"  ✗ 未找到标注文件，跳过")
            skip_count += 1
            continue
        
        # 转换DICOM为NIfTI
        image_output = Path(output_images_dir) / f"{patient_name}_0000.nii.gz"
        
        try:
            convert_dicom_series_to_nifti(patient_dir, target_series, image_output)
        except Exception as e:
            print(f"  ✗ DICOM转换失败: {e}")
            skip_count += 1
            continue
        
        # 复制并处理标注文件
        label_file = label_files[0]
        label_output = Path(output_labels_dir) / f"{patient_name}.nii.gz"
        
        # 读取标注文件
        label_img = nib.load(label_file)
        label_data = label_img.get_fdata()
        
        # 对标注数据进行与图像相同的翻转处理
        # 翻转x轴（axis=0）纠正左右方向（L->R）
        # 翻转y轴（axis=1）纠正前后方向（P->A）
        label_data = np.flip(label_data, axis=0)
        label_data = np.flip(label_data, axis=1)
        
        # 保存处理后的标注文件
        new_label_img = nib.Nifti1Image(label_data, label_img.affine)
        nib.save(new_label_img, str(label_output))
        print(f"  ✓ 标注已处理并保存: {label_output.name}")
        
        # 验证图像和标注尺寸是否匹配
        try:
            img = nib.load(image_output)
            label = nib.load(label_output)
            
            img_shape = img.shape
            label_shape = label.shape
            
            if img_shape == label_shape:
                print(f"  ✓ 图像和标注尺寸匹配: {img_shape}")
            else:
                print(f"  ⚠ 图像和标注尺寸不匹配: 图像={img_shape}, 标注={label_shape}")
                print(f"  请检查数据一致性")
                
        except Exception as e:
            print(f"  ⚠ 验证尺寸时出错: {e}")
        
        success_count += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"{'='*60}")
    print(f"成功: {success_count} 个")
    print(f"跳过: {skip_count} 个")


if __name__ == "__main__":
    print("="*60)
    print("步骤1: DICOM转换")
    print("="*60)
    
    print("\n请修改以下配置后运行:")
    print("""
    INPUT_ROOT = "path/to/your/dataset"  # 包含所有病人文件夹
    EXCEL_PATH = "path/to/your/excel.xlsx"   # Excel文件
    OUTPUT_IMAGES_DIR = "path/to/output/images"
    OUTPUT_LABELS_DIR = "path/to/output/labels"
    
    process_patient_dataset(INPUT_ROOT, EXCEL_PATH, OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR)
    """)
