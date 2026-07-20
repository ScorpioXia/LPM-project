import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os
import matplotlib.pyplot as plt
import cv2

def enhance_image_visibility(img_array):
    """
    应用多种增强方法来提高图像可见性
    """
    enhanced_images = {}
    
    # 1. 原始图像
    enhanced_images['original'] = img_array
    
    # 2. 自动对比度拉伸（基于百分位数）
    p2, p98 = np.percentile(img_array, (2, 98))
    if p98 > p2:
        img_contrast = np.clip((img_array - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    else:
        img_contrast = img_array.copy()
    enhanced_images['auto_contrast'] = img_contrast
    
    # 3. 直方图均衡化
    img_eq = cv2.equalizeHist(img_array.astype(np.uint8))
    enhanced_images['histogram_equalization'] = img_eq
    
    # 4. CLAHE（限制对比度自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_array.astype(np.uint8))
    enhanced_images['clahe'] = img_clahe
    
    # 5. Z-score归一化
    mean = img_array.mean()
    std = img_array.std()
    if std > 0:
        img_zscore = ((img_array - mean) / std * 50 + 128).astype(np.uint8)
    else:
        img_zscore = img_array.copy()
    enhanced_images['zscore_normalized'] = img_zscore
    
    # 6. Gamma校正（增强暗部）
    gamma = 0.5
    img_gamma = np.power(img_array / 255.0, gamma) * 255.0
    img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)
    enhanced_images['gamma_correction'] = img_gamma
    
    # 7. 对数变换（增强低亮度区域）
    epsilon = 1e-6
    img_log = np.log(img_array + epsilon)
    img_log = (img_log - img_log.min()) / (img_log.max() - img_log.min()) * 255
    img_log = np.clip(img_log, 0, 255).astype(np.uint8)
    enhanced_images['log_transform'] = img_log
    
    # 8. 自适应阈值（突出边缘）
    img_adaptive = cv2.adaptiveThreshold(
        img_array.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    enhanced_images['adaptive_threshold'] = img_adaptive
    
    return enhanced_images

def visualize_enhanced_images(image_path, output_dir, img_name):
    """
    可视化并保存增强后的图像
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        print(f"\n{'='*70}")
        print(f"处理图像: {img_name}")
        print(f"{'='*70}")
        print(f"形状: {img_array.shape}")
        print(f"数据类型: {img_array.dtype}")
        print(f"像素值范围: [{img_array.min()}, {img_array.max()}]")
        print(f"均值: {img_array.mean():.2f}")
        print(f"中位数: {np.median(img_array):.2f}")
        print(f"标准差: {img_array.std():.2f}")
        print(f"第5百分位: {np.percentile(img_array, 5):.2f}")
        print(f"第95百分位: {np.percentile(img_array, 95):.2f}")
        
        # 应用增强方法
        enhanced = enhance_image_visibility(img_array)
        
        # 创建可视化
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        
        methods = [
            ('原始图像', 'original', 0, 255),
            ('自动对比度', 'auto_contrast', 0, 255),
            ('直方图均衡化', 'histogram_equalization', 0, 255),
            ('CLAHE', 'clahe', 0, 255),
            ('Z-score归一化', 'zscore_normalized', 0, 255),
            ('Gamma校正(γ=0.5)', 'gamma_correction', 0, 255),
            ('对数变换', 'log_transform', 0, 255),
            ('自适应阈值', 'adaptive_threshold', 0, 255),
            ('直方图', None, None, None)
        ]
        
        for idx, (title, method, vmin, vmax) in enumerate(methods):
            if method is None or title == '直方图':
                # 绘制直方图
                axes[idx].hist(img_array.flatten(), bins=256, range=(0, 256), 
                             alpha=0.7, color='blue', edgecolor='black')
                axes[idx].set_title('像素值直方图', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('像素值', fontsize=10)
                axes[idx].set_ylabel('频数', fontsize=10)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].axvline(img_array.mean(), color='red', linestyle='--', 
                                linewidth=2, label=f'均值: {img_array.mean():.1f}')
                axes[idx].legend()
            else:
                # 显示增强后的图像
                axes[idx].imshow(enhanced[method], cmap='gray', vmin=vmin, vmax=vmax)
                axes[idx].set_title(title, fontsize=12, fontweight='bold')
                axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化结果
        output_path = os.path.join(output_dir, f"{img_name}_enhanced.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {output_path}")
        plt.close()
        
        # 单独保存每种增强方法的结果
        for method_name, img_enhanced in enhanced.items():
            method_output_path = os.path.join(output_dir, f"{img_name}_{method_name}.png")
            cv2.imwrite(method_output_path, img_enhanced)
        
        print(f"所有增强方法已保存到: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"处理 {img_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    base_path = r"e:\code\nnUNet-master\nnUNet_raw\Dataset101_SpineSegmentation\imagesTs"
    output_dir = r"e:\code\nnUNet-master\enhanced_images"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # baixiuhua 系列
    baixiuhua_images = [
        "baixiuhua_slice000_0000.png",
        "baixiuhua_slice004_0000.png",
        "baixiuhua_slice009_0000.png",
        "baixiuhua_slice013_0000.png",
        "baixiuhua_slice018_0000.png"
    ]
    
    # dongdeshuiLS 系列
    dongdeshui_images = [
        "dongdeshuiLS_slice001_0000.png",
        "dongdeshuiLS_slice005_0000.png",
        "dongdeshuiLS_slice010_0000.png",
        "dongdeshuiLS_slice014_0000.png",
        "dongdeshuiLS_slice018_0000.png"
    ]
    
    # 对比：caizhenmei 系列（表现好的样本）
    caizhenmei_images = [
        "caizhenmei_slice005_0000.png",
        "caizhenmei_slice011_0000.png",
        "caizhenmei_slice016_0000.png",
        "caizhenmei_slice019_0000.png",
        "caizhenmei_slice024_0000.png"
    ]
    
    print("="*70)
    print("开始处理 baixiuhua 系列（表现差的样本）")
    print("="*70)
    for img_name in baixiuhua_images:
        img_path = os.path.join(base_path, img_name)
        visualize_enhanced_images(img_path, output_dir, img_name.replace('_0000.png', ''))
    
    print("\n" + "="*70)
    print("开始处理 dongdeshuiLS 系列（表现差的样本）")
    print("="*70)
    for img_name in dongdeshui_images:
        img_path = os.path.join(base_path, img_name)
        visualize_enhanced_images(img_path, output_dir, img_name.replace('_0000.png', ''))
    
    print("\n" + "="*70)
    print("开始处理 caizhenmei 系列（表现好的样本 - 对比）")
    print("="*70)
    for img_name in caizhenmei_images:
        img_path = os.path.join(base_path, img_name)
        visualize_enhanced_images(img_path, output_dir, img_name.replace('_0000.png', ''))
    
    print("\n" + "="*70)
    print("处理完成！")
    print(f"所有增强后的图像已保存到: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
