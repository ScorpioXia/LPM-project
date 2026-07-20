import os
import glob
import numpy as np
import pydicom
from pydicom import uid
import nibabel as nib
import cv2
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免图形显示问题
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import pandas as pd
import warnings

# 设置中文显示，添加更多常见的中文字体选项
plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 忽略matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

def read_dicom_series(dicom_dir, target_size=None, series_number=None):
    """
    读取DICOM序列，支持处理不同尺寸的切片和指定Series Number
    :param dicom_dir: DICOM文件所在目录
    :param target_size: 可选，目标尺寸 (rows, cols)，所有切片将被调整到这个尺寸
    :param series_number: 可选，指定要读取的Series Number，None表示自动检测最常见的序列号
    :return: 3D DICOM数据, 切片位置信息, 像素面积, pixel_spacing_x, pixel_spacing_y, 使用的Series Number
    """
    dicom_files = sorted(glob.glob(os.path.join(dicom_dir, '*.dcm')))
    if not dicom_files:
        print(f"调试: 在 {dicom_dir} 中未找到任何 .dcm 文件")
        return None, None, None, None, None, None

    print(f"调试: 找到 {len(dicom_files)} 个DICOM文件")
    
    # 收集所有DICOM文件的Series Number信息
    series_info = {}
    for dicom_file in dicom_files:
        try:
            dcm = pydicom.dcmread(dicom_file, force=True)
            if not hasattr(dcm, 'TransferSyntaxUID'):
                dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            
            # 获取Series Number
            if hasattr(dcm, 'SeriesNumber'):
                sn = int(dcm.SeriesNumber)
                if sn in series_info:
                    series_info[sn].append(dicom_file)
                else:
                    series_info[sn] = [dicom_file]
            else:
                print(f"警告: DICOM文件 {dicom_file} 中未找到SeriesNumber属性")
        except Exception as e:
            print(f"错误: 处理DICOM文件 {dicom_file} 时出错: {str(e)}")
            continue
    
    # 确定要使用的Series Number
    if not series_info:
        print(f"警告: 在 {dicom_dir} 中未找到任何包含SeriesNumber的DICOM文件")
        return None, None, None, None, None, None
    
    print(f"调试: 找到 {len(series_info)} 个不同的Series Number")
    
    # 如果用户指定了Series Number
    if series_number is not None:
        if series_number in series_info:
            selected_files = series_info[series_number]
            print(f"使用指定的Series Number: {series_number}，包含 {len(selected_files)} 个DICOM文件")
        else:
            print(f"错误: 指定的Series Number {series_number} 不存在于目录 {dicom_dir} 中")
            print(f"可用的Series Number: {list(series_info.keys())}")
            return None, None, None, None, None, None
    else:
        # 自动检测最常见的Series Number
        max_count = 0
        most_common_series = None
        for sn, files in series_info.items():
            if len(files) > max_count:
                max_count = len(files)
                most_common_series = sn
        
        selected_files = series_info[most_common_series]
        print(f"自动检测到Series Number: {most_common_series}，包含 {max_count} 个DICOM文件")
        series_number = most_common_series
    
    # 使用选定Series中的第一个文件来获取元数据（像素间距和目标尺寸）
    first_selected_dcm = pydicom.dcmread(selected_files[0], force=True)
    if not hasattr(first_selected_dcm, 'TransferSyntaxUID'):
        first_selected_dcm.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian

    # 获取像素间距
    pixel_spacing_x = None
    pixel_spacing_y = None
    pixel_area = None
    if hasattr(first_selected_dcm, 'PixelSpacing') and len(first_selected_dcm.PixelSpacing) >= 2:
        pixel_spacing_x = float(first_selected_dcm.PixelSpacing[0])
        pixel_spacing_y = float(first_selected_dcm.PixelSpacing[1])
        pixel_area = pixel_spacing_x * pixel_spacing_y
    else:
        print(f"警告: DICOM文件 {selected_files[0]} 中未找到PixelSpacing属性")

    # 如果没有指定目标尺寸，则使用选定Series中第一个切片的尺寸
    if target_size is None:
        rows, cols = first_selected_dcm.Rows, first_selected_dcm.Columns
        print(f"调试: 未指定目标尺寸，使用Series {series_number} 第一个切片的尺寸: {rows}x{cols}")
    else:
        rows, cols = target_size
        print(f"调试: 使用指定的目标尺寸: {rows}x{cols}")

    # 初始化3D数组
    slice_positions = []
    valid_slices_count = 0
    dicom_volume = []
    
    # 读取选定Series Number的DICOM文件
    skipped_count = 0
    for i, dicom_file in enumerate(selected_files):
        try:
            dcm = pydicom.dcmread(dicom_file, force=True)
            # Handle missing TransferSyntaxUID
            if not hasattr(dcm, 'TransferSyntaxUID'):
                dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

            # 获取切片数据
            pixel_array = dcm.pixel_array

            # 如果切片尺寸与目标尺寸不同，则跳过
            if pixel_array.shape != (rows, cols):
                skipped_count += 1
                if skipped_count <= 3:  # 只打印前3个跳过的文件
                    print(f"调试: 跳过文件 {dicom_file}，尺寸 {pixel_array.shape} 不匹配目标尺寸 ({rows}, {cols})")
                continue
            dicom_volume.append(pixel_array)
            # 获取切片位置（Z轴坐标）
            slice_positions.append(float(dcm.ImagePositionPatient[2]))
            valid_slices_count += 1
        except Exception as e:
            print(f"错误: 处理DICOM文件 {dicom_file} 时出错: {str(e)}")
            continue

    if skipped_count > 3:
        print(f"调试: 还有 {skipped_count - 3} 个文件因尺寸不匹配被跳过")
    
    print(f"调试: 有效切片数量: {valid_slices_count}")

    # 如果没有有效的切片，则返回None
    if valid_slices_count == 0:
        print(f"警告: 没有找到尺寸匹配的DICOM切片，所有切片都被跳过")
        return None, None, None, None, None, None

    # 使用stack将列表中的二维数组堆叠成三维数组 (深度, 高度, 宽度)
    dicom_volume = np.stack(dicom_volume, axis=0)
    # 按切片位置排序(升序)
    sorted_indices = np.argsort(slice_positions)
    dicom_volume = dicom_volume[sorted_indices]
    slice_positions = [slice_positions[i] for i in sorted_indices]

    return dicom_volume, slice_positions, pixel_area, pixel_spacing_x, pixel_spacing_y, series_number


def read_annotation(annotation_path):
    """
    读取标注数据
    :param annotation_path: 标注文件路径（.nii.gz）
    :return: 标注数据和非全0切片的索引列表
    """
    if not os.path.exists(annotation_path):
        print(f"标注文件不存在: {annotation_path}")
        return None, None

    try:
        nii_img = nib.load(annotation_path)
        annotation_data = nii_img.get_fdata()

        # 标注数据维度顺序为(W, H, D)
        if len(annotation_data.shape) == 3:
            # 交换维度顺序为(D, H, W)以匹配DICOM数据
            annotation_data = np.transpose(annotation_data, (2, 1, 0))
            depth = annotation_data.shape[0]
            # 筛选非全0的切片
            non_zero_indices = []

            for i in range(depth):
                slice_data = annotation_data[i]
                if not np.all(slice_data == 0):
                    non_zero_indices.append(i)

            # 打印维度信息，帮助调试
            print(f"转换后标注数据维度: {annotation_data.shape}")
            print(f"非全0标注切片数量: {len(non_zero_indices)}")
            return annotation_data, non_zero_indices
        else:
            print(f"标注数据维度不符合预期: {annotation_data.shape}")
            return None, None
    except Exception as e:
        print(f"读取标注文件出错: {e}")
        return None, None


def normalize_image(image):
    """
    归一化图像数据到0-255
    :param image: 原始图像数据
    :return: 归一化后的图像数据
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        normalized = ((image - min_val) / (max_val - min_val)) * 255
    else:
        normalized = np.zeros_like(image)
    return normalized.astype(np.uint8)


def save_dicom_as_png(dicom_volume, slice_indices, save_dir, pixel_area=None):
    """
    保存指定DICOM切片为PNG灰度图
    :param dicom_volume: DICOM体积数据
    :param slice_indices: 要保存的切片索引列表
    :param save_dir: 保存目录
    :param pixel_area: 像素面积（可选）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果提供了像素面积，打印出来
    if pixel_area is not None:
        print(f"像素面积: {pixel_area:.4f} mm²")

    for i in slice_indices:
        if 0 <= i < dicom_volume.shape[0]:
            slice_data = dicom_volume[i]
            # normalized_slice = normalize_image(slice_data) # 暂时不做归一化
            normalized_slice = slice_data
            save_path = os.path.join(save_dir, f'slice_{i:03d}.png')
            plt.imsave(save_path, normalized_slice, cmap='gray')


def save_annotation_as_pngWdat(annotation_data, slice_indices, save_dir):
    """
    保存标注数据为PNG图像
    :param annotation_data: 标注数据
    :param slice_indices: 要保存的切片索引列表
    :param save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    for i in slice_indices:
        slice_data = annotation_data[i]

        save_path = os.path.join(save_dir, f'annotation_{i:03d}.png')
        # 保存为伪彩色图以便区分不同标签
        plt.imsave(save_path, slice_data, cmap='jet')

        # 同时保存灰度图用于参考
        gray_save_path = os.path.join(save_dir, f'annotation_gray_{i:03d}.png')
        plt.imsave(gray_save_path, slice_data, cmap='gray')

        dat_save_path = os.path.join(save_dir, f'annotation_{i:03d}.dat')
        np.save(dat_save_path, slice_data)


# def save_annotation_as_dat(annotation_data, slice_indices, save_dir):
#     """
#     保存标注数据为二进制dat格式
#     :param annotation_data: 标注数据
#     :param slice_indices: 要保存的切片索引列表
#     :param save_dir: 保存目录
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     for i in slice_indices:
#         slice_data = annotation_data[i]
        
#         save_path = os.path.join(save_dir, f'annotation_{i:03d}.dat')
#         # 保存为二进制格式
#         np.save(save_path, slice_data)
#         # 如果需要无.npy后缀的纯二进制文件，取消下面注释
#         # with open(save_path, 'wb') as f:
#         #     np.array(slice_data).tofile(f)


def save_overlay_images(dicom_volume, annotation_data, matched_indices, output_dir):
    """
    将分割掩膜和对应图像叠加并保存
    :param dicom_volume: DICOM体积数据
    :param annotation_data: 标注数据
    :param matched_indices: 匹配的切片索引列表
    :param output_dir: 输出目录
    """
    # 检查DICOM和标注的尺寸是否匹配
    dicom_size = (dicom_volume.shape[1], dicom_volume.shape[2])
    annotation_size = (annotation_data.shape[1], annotation_data.shape[2])
    
    if dicom_size != annotation_size:
        print(f'警告: DICOM尺寸 {dicom_size} 与标注尺寸 {annotation_size} 不一致，跳过叠加图像生成')
        print('（叠加图像需要相同尺寸，后续模型输入时会统一处理）')
        return
    
    overlay_save_dir = os.path.join(output_dir, 'overlay_png')
    os.makedirs(overlay_save_dir, exist_ok=True)

    for i in matched_indices:
        # 获取对应的DICOM切片和标注
        dicom_slice = dicom_volume[i]
        annotation_slice = annotation_data[i]

        # 归一化DICOM图像
        normalized_dicom = normalize_image(dicom_slice)  # 执行归一化以提高对比度

        # 转换为PIL图像
        # 将 numpy 数组 (H, W) 转换为 PIL 图像
        original_image = Image.fromarray(normalized_dicom.astype(np.uint8))
        original_image = original_image.convert('RGB')

        # 创建红色叠加层
        red_overlay = Image.new('RGB', original_image.size, (255, 0, 0))

        # 混合原始图像和红色叠加层
        blended = Image.blend(original_image, red_overlay, 0.3)

        # 创建掩码 (将标注转换为二值掩码)
        mask = Image.fromarray((annotation_slice > 0).astype(np.uint8) * 255)

        # 在混合图像上粘贴红色，使用掩码
        overlay = blended.copy()
        overlay.paste(Image.new('RGB', original_image.size, (255, 0, 0)), mask=mask)

        # 保存叠加图像
        overlay_path = os.path.join(overlay_save_dir, f'overlay_{i:03d}.png')
        overlay.save(overlay_path)

        print(f"分割叠加图像已保存至 {overlay_path}")


def match_annotated_slices(dicom_volume, annotation_indices):
    """
    匹配有标注的DICOM切片
    :param dicom_volume: DICOM体积数据
    :param annotation_indices: 非全0标注切片的索引列表
    :return: 匹配的DICOM切片索引列表
    """
    # 假设标注索引与DICOM切片索引直接对应
    # 但需要确保索引不超出DICOM体积的范围
    num_dicom_slices = dicom_volume.shape[0]
    matched_indices = [idx for idx in annotation_indices if idx < num_dicom_slices]

    # 打印匹配信息
    print(f"原始标注索引数量: {len(annotation_indices)}")
    print(f"有效DICOM切片索引数量: {len(matched_indices)}")
    return matched_indices


def process_patient(patient_dir, output_base_dir, series_number=None):
    """
    处理单个病人的数据
    :param patient_dir: 病人数据目录
    :param output_base_dir: 输出基础目录
    :param series_number: 可选，指定要读取的Series Number，None表示自动检测
    :return: 病人数据信息列表
    """
    patient_id = os.path.basename(patient_dir)
    print(f'处理病人: {patient_id}')
    patient_data = []

    # 创建输出目录（统一在函数开头创建）
    output_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(output_dir, exist_ok=True)

    # 创建保存所有DICOM切片的目录
    all_dicom_save_dir = os.path.join(output_dir, 'All_dicom_png')
    os.makedirs(all_dicom_save_dir, exist_ok=True)

    # 先读取标注数据
    annotation_path = os.path.join(patient_dir, 'Untitled.nii.gz')
    annotation_data, non_zero_annotation_indices = read_annotation(annotation_path)

    # 如果没有标注数据，直接使用DICOM自身尺寸处理
    if annotation_data is None:
        print(f'警告: 未在{patient_dir}中找到标注文件，将使用DICOM原始尺寸')
        dicom_volume, dicom_positions, pixel_area, pixel_spacing_x, pixel_spacing_y, used_series_number = read_dicom_series(patient_dir, series_number=series_number)
        if dicom_volume is None:
            print(f'警告: 未在{patient_dir}中找到DICOM文件')
            return
        
        # 保存所有DICOM切片
        all_slice_indices = list(range(dicom_volume.shape[0]))
        save_dicom_as_png(dicom_volume, all_slice_indices, all_dicom_save_dir, pixel_area)
        print(f'已将所有 {dicom_volume.shape[0]} 个DICOM切片保存至 {all_dicom_save_dir}')
        
        # 收集没有标注时的DICOM切片信息
        for i in all_slice_indices:
            slice_path = os.path.join(all_dicom_save_dir, f'slice_{i:03d}.png')
            slice_info = {
                '病人ID': patient_id,
                'DICOM切片路径': slice_path,
                '标注图像路径': None,
                '标注数据路径': None,
                '切片编号': i,
                'pixel_spacing_x': pixel_spacing_x,
                'pixel_spacing_y': pixel_spacing_y
            }
            patient_data.append(slice_info)
        return patient_data

    # 获取标注数据的尺寸作为目标尺寸
    target_size = (annotation_data.shape[1], annotation_data.shape[2])
    print(f'使用标注数据尺寸作为目标尺寸: {target_size}')

    # 使用标注尺寸读取DICOM数据
    dicom_volume, dicom_positions, pixel_area, pixel_spacing_x, pixel_spacing_y, used_series_number = read_dicom_series(
        patient_dir, target_size=target_size, series_number=series_number
    )

    # 如果使用目标尺寸读取失败（没有匹配尺寸的切片），则对原始DICOM进行resize
    if dicom_volume is None:
        print(f'警告: 使用标注尺寸 {target_size} 读取DICOM时没有匹配切片，将对DICOM数据进行尺寸调整')
        # 先用原始尺寸读取DICOM
        dicom_volume_original, _, _, pixel_spacing_x, pixel_spacing_y, used_series_number = read_dicom_series(
            patient_dir, series_number=series_number
        )
        
        if dicom_volume_original is None:
            print(f'错误: 无法读取病人 {patient_id} 的DICOM数据')
            return
        
        # 对每个切片进行resize到标注尺寸
        dicom_volume = np.zeros((dicom_volume_original.shape[0], target_size[0], target_size[1]), dtype=dicom_volume_original.dtype)
        for i in range(dicom_volume_original.shape[0]):
            dicom_volume[i] = resize(
                dicom_volume_original[i], 
                target_size, 
                anti_aliasing=True
            )
        print(f'已将DICOM数据从 {dicom_volume_original.shape[1:]} 调整到 {target_size}')
        
        # 计算resize后的像素面积（近似值）
        if pixel_spacing_x is not None and pixel_spacing_y is not None:
            original_size = (dicom_volume_original.shape[1], dicom_volume_original.shape[2])
            scale_x = original_size[0] / target_size[0]
            scale_y = original_size[1] / target_size[1]
            pixel_area = (pixel_spacing_x * scale_x) * (pixel_spacing_y * scale_y)
            print(f'调整后像素面积: {pixel_area:.4f} mm² (近似值)')
    else:
        print(f'DICOM数据尺寸与标注尺寸匹配: {target_size}')

    # 保存所有DICOM切片到All_dicom_png文件夹
    all_slice_indices = list(range(dicom_volume.shape[0]))
    save_dicom_as_png(dicom_volume, all_slice_indices, all_dicom_save_dir, pixel_area)
    print(f'已将所有 {dicom_volume.shape[0]} 个DICOM切片保存至 {all_dicom_save_dir}')

    # 匹配有标注的DICOM切片
    matched_indices = match_annotated_slices(dicom_volume, non_zero_annotation_indices)
    print(f'找到{len(matched_indices)}个匹配的切片')

    # 创建保存有标注DICOM切片的目录
    annotated_dicom_save_dir = os.path.join(output_dir, 'Annotated_dicom_png')
    os.makedirs(annotated_dicom_save_dir, exist_ok=True)

    # 保存有标注的DICOM切片到Annotated_dicom_png文件夹
    save_dicom_as_png(dicom_volume, matched_indices, annotated_dicom_save_dir, pixel_area)
    print(f'已将 {len(matched_indices)} 个有标注的DICOM切片保存至 {annotated_dicom_save_dir}')

    # 保存标注为PNG
    annotation_save_dir = os.path.join(output_dir, 'annotation_png')
    save_annotation_as_pngWdat(annotation_data, matched_indices, annotation_save_dir)

    # 收集标注图像路径信息
    annotation_paths = [os.path.join(annotation_save_dir, f'annotation_{i:03d}.png') for i in matched_indices]

    # 收集标注数据路径信息
    annotation_dat_paths = [os.path.join(annotation_save_dir, f'annotation_{i:03d}.dat') for i in matched_indices]

    # 调用函数保存叠加图像
    save_overlay_images(dicom_volume, annotation_data, matched_indices, output_dir)

    # 收集当前病人的所有切片信息（有标注的切片）
    for i, (annotation_path, annotation_dat_path, idx) in enumerate(zip(annotation_paths, annotation_dat_paths, matched_indices)):
        dicom_path = os.path.join(annotated_dicom_save_dir, f'slice_{idx:03d}.png')
        slice_info = {
            '病人ID': patient_id,
            'DICOM切片路径': dicom_path,
            '标注图像路径': annotation_path,
            '标注数据路径': annotation_dat_path,
            '切片编号': idx,
            'pixel_spacing_x': pixel_spacing_x,
            'pixel_spacing_y': pixel_spacing_y
        }
        patient_data.append(slice_info)

    print(f'病人{patient_id}处理完成')
    return patient_data


def batch_process(input_dir, output_base_dir, series_number_excel=None):
    """
    批量处理所有病人数据
    :param input_dir: 输入数据目录
    :param output_base_dir: 输出基础目录
    :param series_number_excel: 可选，包含病人ID和对应Series Number的Excel文件路径
    """
    # 获取所有病人目录
    patient_dirs = [d for d in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(d)]

    if not patient_dirs:
        print(f'错误: 在{input_dir}中未找到病人目录')
        return

    print(f'找到{len(patient_dirs)}个病人数据')

    # 创建输出基础目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 从Excel读取病人ID和Series Number的映射关系
    patient_series_map = {}
    if series_number_excel and os.path.exists(series_number_excel):
        try:
            df = pd.read_excel(series_number_excel)
            # 假设Excel中有'病人ID'和'SeriesNumber'列
            if '病人ID' in df.columns and 'SeriesNumber' in df.columns:
                for _, row in df.iterrows():
                    patient_id = str(row['病人ID']).strip()
                    series_number = int(row['SeriesNumber'])
                    patient_series_map[patient_id] = series_number
                print(f"成功从Excel读取{len(patient_series_map)}个病人的Series Number信息")
            else:
                print(f"警告: Excel文件{series_number_excel}中缺少'病人ID'或'SeriesNumber'列")
        except Exception as e:
            print(f"错误: 读取Excel文件{series_number_excel}时出错: {str(e)}")
    elif series_number_excel:
        print(f"警告: Excel文件{series_number_excel}不存在")

    # 存储所有病人的切片信息
    all_patient_data = []

    # 批量处理每个病人
    for patient_dir in tqdm(patient_dirs):
        patient_id = os.path.basename(patient_dir)
        # 获取当前病人的Series Number
        series_number = patient_series_map.get(patient_id, None)
        patient_data = process_patient(patient_dir, output_base_dir, series_number)
        if patient_data:
            all_patient_data.extend(patient_data)

    # 将所有数据保存到Excel
    if all_patient_data:
        df = pd.DataFrame(all_patient_data, columns=['病人ID', 'DICOM切片路径', '标注图像路径','标注数据路径', '切片编号', 'pixel_spacing_x', 'pixel_spacing_y'])
        excel_path = os.path.join(output_base_dir, 'processed_data_info.xlsx')
        df.to_excel(excel_path, index=False)
        print(f'所有病人数据信息已保存至: {excel_path}')
    else:
        print('没有数据可保存到Excel')

    print('所有病人数据处理完成')


if __name__ == '__main__':
    # 设置输入和输出路径
    input_dir = r'E:\dataset\yaozhui\segdataset-20251225process\20250819'
    output_base_dir = r'E:\dataset\yaozhui\segdataset-20251225process\20250819processed'
    
    # 直接从固定路径读取Excel文件
    series_number_excel = r'E:\dataset\yaozhui\segdataset-20251225process\series_numbers.xlsx'

    # 批量处理
    batch_process(input_dir, output_base_dir, series_number_excel)