"""
肌肉特征提取脚本 (版本 v3)
基于 feature_extraction_lv1_1.py 扩展，增加以下信息：
1. 像素间距 (pixel_spacing_x, pixel_spacing_y) - 记录图像物理分辨率
2. 实际肌肉面积 (muscle_area_mm2) - 以平方毫米为单位的肌肉面积

提取的特征包括：
1. 基础形态学特征（13项）
2. 灰度/信号特征（14项）
3. 空间分布特征（8项）
4. 纹理特征（12项，需要PyRadiomics）
5. Level 2: 3D 体积与全局特征（21项）
6. Level 3.1: 跨层梯度特征（5项，每病人每肌肉）
7. Level 3.2-3.5: 多肌肉关系特征（13项，每病人一行）

分别统计六块肌肉：
- 多裂肌（左右）
- 竖脊肌（左右）
- 腰大肌（左右）

输出文件：
- muscle_features_2d_v3.csv: 2D切片级特征（每切片每肌肉一行），包含像素间距和实际面积
- muscle_features_3d_v3.csv: 3D病人级特征
- muscle_features_level3_cross_v3.csv: Level 3.1跨层梯度特征
- muscle_features_level3_multi_v3.csv: Level 3.2-3.5多肌肉关系特征
"""

import os
# 设置环境变量解决 Windows 上 KMeans 内存泄漏问题
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from typing import Dict, List, Tuple

# 导入特征计算模块
from muscle_feature_calculator import (
    calculate_all_muscle_features,
    calculate_3d_features,
    calculate_cross_layer_gradient_features,
    calculate_multi_muscle_features
)


# 肌肉标签映射 - 修正标签名称对应关系
# 修改映射：
# 1 左侧多裂肌 --> 左侧腰大肌
# 2 右侧多裂肌 --> 右侧腰大肌
# 3 左侧竖脊肌 --> 左侧多裂肌
# 4 右侧竖脊肌 --> 右侧多裂肌
# 5 左侧腰大肌 --> 左侧竖脊肌
# 6 右侧腰大肌 --> 右侧竖脊肌
MUSCLE_LABELS = {
    1: "psoas_left",         # 左侧腰大肌（原左侧多裂肌）
    2: "psoas_right",        # 右侧腰大肌（原右侧多裂肌）
    3: "multifidus_left",    # 左侧多裂肌（原左侧竖脊肌）
    4: "multifidus_right",   # 右侧多裂肌（原右侧竖脊肌）
    5: "erector_spinae_left", # 左侧竖脊肌（原左侧腰大肌）
    6: "erector_spinae_right" # 右侧竖脊肌（原右侧腰大肌）
}


def is_bimodal_valid(pix_vals, min_samples=30):
    """
    检验双峰性。返回 (是否有效, 推荐阈值或None)
    
    两种互补的检验，任意一个通过即认为"双峰有效"：
    - 检验A: 峰谷比 (Peak-to-Valley Ratio)
    - 检验B: Ashman's D
    """
    if len(pix_vals) < min_samples:
        return False, None   # 像素太少，不做检验，交给回退

    # --- 检验A: 峰谷比 ---
    hist, bin_edges = np.histogram(pix_vals, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, _ = find_peaks(hist, height=0.05 * hist.max(), distance=len(hist)//10)
    valid_a = False
    if len(peaks) >= 2:
        # 取最高的两个峰
        peak_heights = hist[peaks]
        top2_idx = np.argsort(peak_heights)[-2:]
        p1, p2 = peaks[top2_idx[0]], peaks[top2_idx[1]]
        # 两峰之间的山谷最低点
        valley_slice = hist[min(p1,p2):max(p1,p2)+1]
        valley_min = valley_slice.min()
        valley_depth = min(hist[p1], hist[p2]) - valley_min
        if valley_depth > 0.1 * max(hist[p1], hist[p2]):
            valid_a = True

    # --- 检验B: Ashman's D ---
    valid_b = False
    try:
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(pix_vals.reshape(-1, 1))
        means = gmm.means_.flatten()
        covs = gmm.covariances_.flatten()
        if np.all(covs > 0):
            D = np.sqrt(2) * np.abs(means[0] - means[1]) / np.sqrt(covs[0] + covs[1])
            if D > 2.0:
                valid_b = True
    except:
        pass

    return (valid_a or valid_b), None


def get_fat_threshold_per_muscle(pix_vals):
    """
    返回脂肪阈值。pix_vals 是标准化后该肌肉内的像素值。
    
    算法流程：
    1. 检验双峰性
    2. 若双峰有效 → 使用 Otsu 阈值
    3. 若双峰无效 → 使用回退策略（中位数 + 1.5*IQR）
    """
    if len(pix_vals) < 10:
        return 0.5  # 默认阈值
    
    bimodal, _ = is_bimodal_valid(pix_vals)
    if bimodal:
        try:
            thresh = threshold_otsu(pix_vals)
            # 安全钳：Otsu给出的阈值若极端，则退回
            low_bound = np.percentile(pix_vals, 10)
            high_bound = np.percentile(pix_vals, 90)
            if thresh < low_bound or thresh > high_bound:
                bimodal = False
        except:
            bimodal = False

    if not bimodal:
        # 回退：中位数 + 1.5*IQR
        med = np.median(pix_vals)
        iqr = np.subtract(*np.percentile(pix_vals, [75, 25]))
        thresh = med + 1.5 * iqr
        # 额外限制：不低于最小值，不高于最大值
        thresh = np.clip(thresh, pix_vals.min(), pix_vals.max())
    
    return thresh


def calculate_features(
    normalized_image: np.ndarray,
    label: np.ndarray,
    muscle_label: int,
    pixel_spacing: Tuple[float, float] = None
) -> Dict:
    """
    计算单个切片中单个肌肉的所有特征（形态学 + 灰度/信号）
    
    Args:
        normalized_image: 标准化后的图像切片
        label: 分割标签切片
        muscle_label: 肌肉标签编号
        pixel_spacing: 像素物理间距（如果为None，使用(1.0, 1.0)）
    
    Returns:
        特征字典，包含像素间距和实际肌肉面积
    """
    if pixel_spacing is None:
        pixel_spacing = (1.0, 1.0)
    
    # 提取特定肌肉的掩码
    muscle_mask = label == muscle_label
    total_csa_pixels = np.sum(muscle_mask)  # 原始像素数量
    
    # 计算动态脂肪阈值
    if total_csa_pixels > 0:
        muscle_pixels = normalized_image[muscle_mask]
        fat_threshold = get_fat_threshold_per_muscle(muscle_pixels)
    else:
        fat_threshold = 0.5
    
    # 使用新模块计算所有特征
    features = calculate_all_muscle_features(
        normalized_image=normalized_image,
        muscle_mask=muscle_mask,
        pixel_spacing=pixel_spacing,
        fat_threshold=fat_threshold
    )
    
    # ========== 新增：像素间距信息 ==========
    features['pixel_spacing_x'] = pixel_spacing[0]  # X方向像素间距（mm）
    features['pixel_spacing_y'] = pixel_spacing[1]  # Y方向像素间距（mm）
    
    # ========== 新增：实际肌肉面积（平方毫米） ==========
    # Area 已经是用像素间距计算的实际面积（mm²）
    features['muscle_area_mm2'] = features.get('Area', 0.0)
    
    # ========== Total_CSA 改为原始像素数量 ==========
    # 为了向后兼容保留 Total_CSA，但修改为原始像素数
    features['Total_CSA'] = total_csa_pixels
    
    # 添加动态阈值记录
    features['fat_threshold_used'] = fat_threshold
    
    return features


def process_single_patient(
    normalized_image_path: str,
    label_path: str,
    patient_id: str,
    output_dir: str
) -> List[Dict]:
    """
    处理单个病人的所有切片，计算每块肌肉的特征
    像素间距自动从图像头文件中读取
    """
    # 加载数据
    img = nib.load(normalized_image_path)
    normalized_data = img.get_fdata()
    
    # 从 nifti 头文件中读取像素间距 (dx, dy)
    # get_zooms() 返回 (x, y, z) 三个方向的实际间距
    zooms = img.header.get_zooms()
    pixel_spacing = (zooms[0], zooms[1])   # 单位：mm
    
    lbl = nib.load(label_path)
    label_data = lbl.get_fdata().astype(np.int32)
    
    assert normalized_data.shape == label_data.shape, \
        f'Image shape {normalized_data.shape} != label shape {label_data.shape}'
    
    slice_features = []
    
    # 处理每个切片
    for z in range(normalized_data.shape[2]):
        norm_slice = normalized_data[:, :, z]
        label_slice = label_data[:, :, z]
        
        # 处理每块肌肉
        for muscle_label, muscle_name in MUSCLE_LABELS.items():
            features = calculate_features(norm_slice, label_slice, muscle_label, pixel_spacing)
            features['patient_id'] = patient_id
            features['slice_index'] = z
            features['muscle_name'] = muscle_name
            
            slice_features.append(features)
    
    return slice_features


def process_single_patient_3d(
    normalized_image_path: str,
    label_path: str,
    patient_id: str
) -> List[Dict]:
    """
    处理单个病人的3D特征（病人级别）

    Returns:
        每个肌肉的3D特征列表
    """
    img = nib.load(normalized_image_path)
    normalized_data = img.get_fdata()

    zooms = img.header.get_zooms()
    pixel_spacing = (zooms[0], zooms[1])
    slice_thickness = zooms[2] if len(zooms) > 2 else 1.0

    lbl = nib.load(label_path)
    label_data = lbl.get_fdata().astype(np.int32)

    features_3d = []

    for muscle_label, muscle_name in MUSCLE_LABELS.items():
        features = calculate_3d_features(
            volume_data=normalized_data,
            label_data=label_data,
            muscle_label=muscle_label,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness
        )
        features['patient_id'] = patient_id
        features['muscle_name'] = muscle_name
        features['pixel_spacing_x'] = pixel_spacing[0]
        features['pixel_spacing_y'] = pixel_spacing[1]
        features_3d.append(features)

    return features_3d


def process_single_patient_level3_cross_layer(
    normalized_image_path: str,
    label_path: str,
    patient_id: str
) -> List[Dict]:
    """
    处理单个病人的 Level 3.1 跨层梯度特征（每病人每肌肉一行）

    Returns:
        每个肌肉的跨层梯度特征列表
    """
    img = nib.load(normalized_image_path)
    normalized_data = img.get_fdata()

    zooms = img.header.get_zooms()
    pixel_spacing = (zooms[0], zooms[1])
    slice_thickness = zooms[2] if len(zooms) > 2 else 1.0

    lbl = nib.load(label_path)
    label_data = lbl.get_fdata().astype(np.int32)

    features_level3 = []

    for muscle_label, muscle_name in MUSCLE_LABELS.items():
        features = calculate_cross_layer_gradient_features(
            volume_data=normalized_data,
            label_data=label_data,
            muscle_label=muscle_label,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness
        )
        features['patient_id'] = patient_id
        features['muscle_name'] = muscle_name
        features_level3.append(features)

    return features_level3


def process_single_patient_level3_multi_muscle(
    normalized_image_path: str,
    label_path: str,
    patient_id: str
) -> Dict:
    """
    处理单个病人的 Level 3.2-3.5 多肌肉关系特征（每病人一行）

    Returns:
        病人级综合特征字典
    """
    img = nib.load(normalized_image_path)
    normalized_data = img.get_fdata()

    zooms = img.header.get_zooms()
    pixel_spacing = (zooms[0], zooms[1])
    slice_thickness = zooms[2] if len(zooms) > 2 else 1.0

    lbl = nib.load(label_path)
    label_data = lbl.get_fdata().astype(np.int32)

    features = calculate_multi_muscle_features(
        volume_data=normalized_data,
        label_data=label_data,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness
    )
    features['patient_id'] = patient_id

    return features


def main():
    print('=' * 80)
    print('肌肉特征提取脚本 (版本 v3)')
    print('新增：像素间距和实际肌肉面积（mm²）')
    print('=' * 80)
    
    # 配置参数
    PREPROCESSED_DIR = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\test_anzhonghua\output"
    LABELS_DIR = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\test_anzhonghua\labels"
    OUTPUT_DIR = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\test_anzhonghua\features_v3"
    
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f'输出目录: {OUTPUT_DIR}')
    
    # 获取所有病人目录
    patient_dirs = [d for d in Path(PREPROCESSED_DIR).iterdir() if d.is_dir()]
    print(f'找到 {len(patient_dirs)} 个病人目录')
    
    all_features = []
    all_features_3d = []
    all_features_level3_cross = []
    all_features_level3_multi = []

    # 处理每个病人
    for patient_dir in tqdm(patient_dirs, desc='Processing patients'):
        patient_id = patient_dir.name

        # 找到标准化图像文件
        normalized_files = list(patient_dir.glob('*_normalized.nii.gz'))
        if not normalized_files:
            print(f'跳过 {patient_id}: 未找到标准化图像')
            continue

        normalized_path = str(normalized_files[0])

        # 找到分割标签文件
        label_path = Path(LABELS_DIR) / f'{patient_id}.nii.gz'
        if not label_path.exists():
            label_path = Path(LABELS_DIR) / f'{patient_id}.nii'

        if not label_path.exists():
            print(f'跳过 {patient_id}: 未找到分割标签')
            continue

        # 处理病人数据
        print(f'处理病人: {patient_id}')
        try:
            # 2D 切片级特征
            patient_features = process_single_patient(
                normalized_path,
                str(label_path),
                patient_id,
                OUTPUT_DIR
            )
            all_features.extend(patient_features)

            # 3D 病人级特征
            patient_features_3d = process_single_patient_3d(
                normalized_path,
                str(label_path),
                patient_id
            )
            all_features_3d.extend(patient_features_3d)

            # Level 3.1: 跨层梯度特征（每病人每肌肉）
            patient_features_level3_cross = process_single_patient_level3_cross_layer(
                normalized_path,
                str(label_path),
                patient_id
            )
            all_features_level3_cross.extend(patient_features_level3_cross)

            # Level 3.2-3.5: 多肌肉关系特征（每病人一行）
            patient_features_level3_multi = process_single_patient_level3_multi_muscle(
                normalized_path,
                str(label_path),
                patient_id
            )
            all_features_level3_multi.append(patient_features_level3_multi)
        except Exception as e:
            print(f'处理 {patient_id} 时出错: {e}')
            import traceback
            traceback.print_exc()

    # 保存 2D 切片级特征（包含新增的像素间距和肌肉面积）
    if all_features:
        df = pd.DataFrame(all_features)
        output_file = Path(OUTPUT_DIR) / 'muscle_features_2d_v3.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f'\n2D特征已保存到: {output_file}')

        print('\n2D特征提取完成!')
        print(f'处理了 {len(df["patient_id"].unique())} 个病人')
        print(f'计算了 {len(df)} 个切片的特征')
        
        # 显示新增列的示例
        print('\n新增特征示例:')
        sample_cols = ['patient_id', 'slice_index', 'muscle_name', 
                      'pixel_spacing_x', 'pixel_spacing_y', 'muscle_area_mm2']
        print(df[sample_cols].head().to_string())

    # 保存 3D 病人级特征
    if all_features_3d:
        df_3d = pd.DataFrame(all_features_3d)
        output_file_3d = Path(OUTPUT_DIR) / 'muscle_features_3d_v3.csv'
        df_3d.to_csv(output_file_3d, index=False, encoding='utf-8-sig')
        print(f'\n3D特征已保存到: {output_file_3d}')

    # 保存 Level 3.1: 跨层梯度特征（每病人每肌肉）
    if all_features_level3_cross:
        df_level3_cross = pd.DataFrame(all_features_level3_cross)
        output_file_level3_cross = Path(OUTPUT_DIR) / 'muscle_features_level3_cross_v3.csv'
        df_level3_cross.to_csv(output_file_level3_cross, index=False, encoding='utf-8-sig')
        print(f'Level 3.1跨层梯度特征已保存到: {output_file_level3_cross}')

    # 保存 Level 3.2-3.5: 多肌肉关系特征（每病人一行）
    if all_features_level3_multi:
        df_level3_multi = pd.DataFrame(all_features_level3_multi)
        output_file_level3_multi = Path(OUTPUT_DIR) / 'muscle_features_level3_multi_v3.csv'
        df_level3_multi.to_csv(output_file_level3_multi, index=False, encoding='utf-8-sig')
        print(f'Level 3.2-3.5多肌肉关系特征已保存到: {output_file_level3_multi}')

    if not all_features and not all_features_3d and not all_features_level3_cross and not all_features_level3_multi:
        print('未提取到任何特征')


if __name__ == '__main__':
    main()