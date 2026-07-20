"""
验证所有图像的像素间距
"""

import nibabel as nib
from pathlib import Path
from collections import defaultdict


def verify_pixel_spacing(images_dir):
    """
    验证指定目录下所有图像的像素间距
    """
    spacings = set()
    spacing_details = defaultdict(list)
    
    print(f'扫描目录: {images_dir}')
    print('=' * 80)
    
    # 遍历所有.nii.gz文件
    nii_files = list(Path(images_dir).glob("*.nii.gz"))
    
    if not nii_files:
        print('未找到任何.nii.gz文件')
        return
    
    print(f'找到 {len(nii_files)} 个图像文件')
    print('=' * 80)
    
    for nii_file in nii_files:
        try:
            img = nib.load(nii_file)
            zooms = img.header.get_zooms()
            
            # 获取x, y方向的像素间距（保留4位小数）
            spacing_xy = (round(zooms[0], 4), round(zooms[1], 4))
            spacing_xyz = (round(zooms[0], 4), round(zooms[1], 4), round(zooms[2], 4))
            
            spacings.add(spacing_xy)
            spacing_details[spacing_xy].append(nii_file.name)
            
            print(f'{nii_file.name}:')
            print(f'  X间距: {zooms[0]:.4f} mm')
            print(f'  Y间距: {zooms[1]:.4f} mm')
            print(f'  Z间距: {zooms[2]:.4f} mm')
            
        except Exception as e:
            print(f'处理 {nii_file.name} 时出错: {e}')
    
    print('=' * 80)
    print(f'发现的XY像素间距: {spacings}')
    print('=' * 80)
    
    # 按像素间距分组显示文件
    for spacing, files in spacing_details.items():
        print(f'\n像素间距 {spacing}:')
        for file in files:
            print(f'  - {file}')
    
    print('=' * 80)
    print(f'总共发现 {len(spacings)} 种不同的像素间距')
    
    if len(spacings) == 1:
        print('所有图像的像素间距一致')
    else:
        print('警告: 发现多种不同的像素间距，可能需要统一处理')


def main():
    print('=' * 80)
    print('像素间距验证工具')
    print('=' * 80)
    print()
    
    # 验证原始图像目录
    images_dir = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\images"
    if Path(images_dir).exists():
        print('验证原始图像目录:')
        verify_pixel_spacing(images_dir)
    else:
        print(f'原始图像目录不存在: {images_dir}')
    
    print()
    
    # 验证标准化图像目录
    preprocessed_dir = r"E:\dataset\yaozhui\QiLuhospital\20260320testdata-1\preprocessed-ds-55"
    if Path(preprocessed_dir).exists():
        print('验证标准化图像目录:')
        for patient_dir in Path(preprocessed_dir).iterdir():
            if patient_dir.is_dir():
                print(f'\n病人: {patient_dir.name}')
                verify_pixel_spacing(patient_dir)
    else:
        print(f'标准化图像目录不存在: {preprocessed_dir}')


if __name__ == '__main__':
    main()
