"""
图像预处理使用示例
请根据您的实际情况修改配置参数后运行
"""

import os
from pathlib import Path
from image_preprocessing import batch_process, process_single_patient
from visualize_preprocessing import visualize_single_patient, plot_fat_signal_statistics


def example_single_patient():
    """
    示例：处理单个病人
    """
    print('=' * 80)
    print('示例1: 处理单个病人')
    print('=' * 80)
    
    # ==========================================
    # 请修改以下配置
    # ==========================================
    IMAGE_PATH = r"path\to\your\image.nii.gz"  # 原始图像路径
    LABEL_PATH = r"path\to\your\label.nii.gz"  # 分割标签路径
    # 自动从分割标签文件名提取病人ID
    PATIENT_ID = Path(LABEL_PATH).stem.replace('.nii', '')  # 从 kongqingrong.nii.gz 提取 kongqingrong
    OUTPUT_DIR = r"path\to\output\single_patient"  # 输出目录
    
    # 像素物理间距 (根据您的数据修改，单位mm)
    PIXEL_SPACING = (1.0, 1.0)
    
    # ==========================================
    # 处理单个病人
    # ==========================================
    if not Path(IMAGE_PATH).exists():
        print(f'错误: 图像文件不存在: {IMAGE_PATH}')
        return
    
    if not Path(LABEL_PATH).exists():
        print(f'错误: 标签文件不存在: {LABEL_PATH}')
        return
    
    patient_stats, slice_stats = process_single_patient(
        image_path=IMAGE_PATH,
        label_path=LABEL_PATH,
        patient_id=PATIENT_ID,
        output_dir=OUTPUT_DIR,
        pixel_spacing=PIXEL_SPACING,
        n4_shrink_factor=4,
        fat_percentile_threshold=95.0,
        back_ratio=0.2,
        min_fat_area_ratio=0.05,
        max_fat_area_ratio=0.2,
        min_csa=10.0,
        max_fat_ratio=0.95
    )
    
    print('\n病人统计:')
    print(patient_stats)
    
    print(f'\n处理完成! 结果保存在: {OUTPUT_DIR}')


def example_batch_process():
    """
    示例：批量处理所有病人
    """
    print('=' * 80)
    print('示例2: 批量处理所有病人')
    print('=' * 80)
    
    # ==========================================
    # 配置参数（与image_preprocessing.py保持一致）
    # ==========================================
    IMAGES_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\images"  # 原始图像目录
    LABELS_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\results"  # 分割标签目录
    OUTPUT_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\preprocessed"  # 输出目录
    
    # 像素物理间距 (单位mm)
    PIXEL_SPACING = (1.0, 1.0)
    
    # ==========================================
    # 批量处理
    # ==========================================
    if not Path(IMAGES_DIR).exists():
        print(f'错误: 图像目录不存在: {IMAGES_DIR}')
        return
    
    if not Path(LABELS_DIR).exists():
        print(f'错误: 标签目录不存在: {LABELS_DIR}')
        return
    
    patient_df, slice_df = batch_process(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        pixel_spacing=PIXEL_SPACING,
        n4_shrink_factor=4,
        fat_percentile_threshold=95.0,
        back_ratio=0.2,
        min_fat_area_ratio=0.05,
        max_fat_area_ratio=0.2,
        min_csa=10.0,
        max_fat_ratio=0.95
    )
    
    print('\n病人统计DataFrame:')
    print(patient_df)
    
    print(f'\n处理完成! 结果保存在: {OUTPUT_DIR}')
    print(f'病人统计CSV: {OUTPUT_DIR}/patient_statistics.csv')
    print(f'切片统计CSV: {OUTPUT_DIR}/slice_statistics.csv')


def example_visualization():
    """
    示例：可视化预处理结果
    """
    print('=' * 80)
    print('示例3: 可视化预处理结果')
    print('=' * 80)
    
    # ==========================================
    # 配置参数（与image_preprocessing.py保持一致）
    # ==========================================
    IMAGES_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\images"  # 原始图像目录
    LABELS_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\results"  # 分割标签目录
    OUTPUT_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\preprocessed"  # 输出目录
    VIS_OUTPUT_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\visualizations"  # 可视化输出目录
    
    # 自动找到第一个病人
    patient_dirs = list(Path(OUTPUT_DIR).glob('patient_*'))
    if not patient_dirs:
        print(f'错误: 未找到病人目录: {OUTPUT_DIR}')
        return
    
    PATIENT_DIR = str(patient_dirs[0])
    patient_id = PATIENT_DIR.split('_')[-1]
    
    # 找到对应的原始图像和标签
    image_files = list(Path(IMAGES_DIR).glob(f'*{patient_id}*'))
    label_files = list(Path(LABELS_DIR).glob(f'*{patient_id}*'))
    
    if not image_files:
        print(f'错误: 未找到病人 {patient_id} 的原始图像')
        return
    
    if not label_files:
        print(f'错误: 未找到病人 {patient_id} 的分割标签')
        return
    
    ORIGINAL_IMAGE = str(image_files[0])
    LABEL_PATH = str(label_files[0])
    SLICE_INDEX = 10  # 要查看的切片索引
    
    # 创建可视化输出目录
    Path(VIS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 可视化单个病人
    # ==========================================
    if not Path(PATIENT_DIR).exists():
        print(f'错误: 病人目录不存在: {PATIENT_DIR}')
        return
    
    visualize_single_patient(
        patient_dir=PATIENT_DIR,
        original_image_path=ORIGINAL_IMAGE,
        label_path=LABEL_PATH,
        slice_index=SLICE_INDEX,
        output_dir=VIS_OUTPUT_DIR
    )


def example_statistics_plot():
    """
    示例：绘制统计图表
    """
    print('=' * 80)
    print('示例4: 绘制脂肪信号统计图表')
    print('=' * 80)
    
    # ==========================================
    # 配置参数（与image_preprocessing.py保持一致）
    # ==========================================
    PREPROCESSED_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\preprocessed"  # 预处理输出目录
    VIS_OUTPUT_DIR = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\visualizations"  # 可视化输出目录
    
    # 切片统计CSV路径
    SLICE_CSV_PATH = str(Path(PREPROCESSED_DIR) / "slice_statistics.csv")
    
    # 创建可视化输出目录
    Path(VIS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 绘制统计图表
    # ==========================================
    if not Path(SLICE_CSV_PATH).exists():
        print(f'错误: CSV文件不存在: {SLICE_CSV_PATH}')
        return
    
    plot_fat_signal_statistics(
        slice_csv_path=SLICE_CSV_PATH,
        output_dir=VIS_OUTPUT_DIR
    )


if __name__ == '__main__':
    print('=' * 80)
    print('图像预处理 - 自动运行2-4功能')
    print('=' * 80)
    
    print('\n开始自动执行完整流程:')
    print('1. 批量处理所有病人 (示例2)')
    print('2. 可视化预处理结果 (示例3)')
    print('3. 绘制统计图表 (示例4)')
    
    # ==========================================
    # 执行示例2: 批量处理所有病人
    # ==========================================
    print('\n' + '=' * 60)
    print('步骤1: 执行批量处理')
    print('=' * 60)
    example_batch_process()
    
    # ==========================================
    # 执行示例3: 可视化预处理结果
    # ==========================================
    print('\n' + '=' * 60)
    print('步骤2: 执行可视化')
    print('=' * 60)
    example_visualization()
    
    # ==========================================
    # 执行示例4: 绘制统计图表
    # ==========================================
    print('\n' + '=' * 60)
    print('步骤3: 执行统计图表绘制')
    print('=' * 60)
    example_statistics_plot()
    
    print('\n' + '=' * 80)
    print('所有功能执行完成!')
    print('=' * 80)
