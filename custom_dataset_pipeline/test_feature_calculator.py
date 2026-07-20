"""
测试特征计算器
验证 muscle_feature_calculator 模块是否正常工作
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from muscle_feature_calculator import (
    calculate_morphological_features,
    calculate_intensity_features,
    calculate_all_muscle_features
)


def create_test_image():
    """创建测试用的假图像和掩码"""
    h, w = 200, 200
    # 创建黑色背景
    image = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.bool_)
    
    # 创建一个圆形肌肉掩码
    center_y, center_x = 100, 100
    radius = 50
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask[dist_from_center <= radius] = True
    
    # 创建信号：肌肉部分 = 0.7，脂肪部分 = 1.2
    image[mask] = 0.7
    # 在圆形内添加一个小的脂肪区域（高信号）
    fat_center_y, fat_center_x = 100, 100
    fat_radius = 20
    dist_fat = np.sqrt((x - fat_center_x)**2 + (y - fat_center_y)**2)
    fat_mask = (dist_fat <= fat_radius) & mask
    image[fat_mask] = 1.2
    
    return image, mask, fat_mask


def test_morphological_features():
    print("测试形态学特征...")
    
    _, mask, _ = create_test_image()
    pixel_spacing = (0.5, 0.5)  # 0.5mm 像素间距
    
    features = calculate_morphological_features(mask, pixel_spacing)
    
    print("形态学特征计算结果:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value:.4f}")
    
    print()


def test_intensity_features():
    print("测试灰度/信号特征...")
    
    image, muscle_mask, fat_mask = create_test_image()
    pixel_spacing = (0.5, 0.5)
    
    features = calculate_intensity_features(image, muscle_mask, fat_mask, pixel_spacing)
    
    print("灰度特征计算结果:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value:.4f}")
    
    print()


def test_all_features():
    print("测试完整特征提取...")
    
    image, muscle_mask, _ = create_test_image()
    pixel_spacing = (0.5, 0.5)
    fat_threshold = 1.0  # 假设的脂肪阈值
    
    features = calculate_all_muscle_features(
        normalized_image=image,
        muscle_mask=muscle_mask,
        pixel_spacing=pixel_spacing,
        fat_threshold=fat_threshold
    )
    
    print(f"提取到的特征数: {len(features)}")
    print("\n所有特征:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value:.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('标准化图像')
    
    plt.subplot(1, 3, 2)
    plt.imshow(muscle_mask, cmap='gray')
    plt.title('肌肉掩码')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image >= fat_threshold, cmap='gray')
    plt.title(f'脂肪掩码 (阈值={fat_threshold})')
    
    plt.tight_layout()
    plt.savefig('test_visualization.png', dpi=150)
    print("\n可视化结果已保存到 test_visualization.png")


if __name__ == '__main__':
    print("="*60)
    print("肌肉特征计算器测试")
    print("="*60)
    print()
    
    test_morphological_features()
    test_intensity_features()
    test_all_features()
    
    print("\n测试完成！")
