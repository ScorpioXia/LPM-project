import os

V__TXT_ = """
肌肉特征提取脚本 (版本 v4)
基于 feature_extraction_v3.py 扩展，新增以下功能：
1. 病人特征提取成功/失败检测
2. 将检测结果保存为txt文档

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
- muscle_features_2d_v4.csv: 2D切片级特征（每切片每肌肉一行），包含像素间距和实际面积
- muscle_features_3d_v4.csv: 3D病人级特征
- muscle_features_level3_cross_v4.csv: Level 3.1跨层梯度特征
- muscle_features_level3_multi_v4.csv: Level 3.2-3.5多肌肉关系特征
- extraction_report_v4.txt: 特征提取检测报告
"""
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
from datetime import datetime

from muscle_feature_calculator import (
    calculate_all_muscle_features,
    calculate_3d_features,
    calculate_cross_layer_gradient_features,
    calculate_multi_muscle_features
)


MUSCLE_LABELS = {
    1: "psoas_left",
    2: "psoas_right",
    3: "multifidus_left",
    4: "multifidus_right",
    5: "erector_spinae_left",
    6: "erector_spinae_right"
}


def is_bimodal_valid(pix_vals, min_samples=30):
    """
    检验双峰性。返回 (是否有效, 推荐阈值或None)

    两种互补的检验，任意一个通过即认为"双峰有效"：
    - 检验A: 峰谷比 (Peak-to-Valley Ratio)
    - 检验B: Ashman's D
    """
    if len(pix_vals) < min_samples:
        return False, None

    hist, bin_edges = np.histogram(pix_vals, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    distance = max(1, len(hist)//10)
    peaks, _ = find_peaks(hist, height=0.05 * hist.max(), distance=distance)
    valid_a = False
    if len(peaks) >= 2:
        peak_heights = hist[peaks]
        top2_idx = np.argsort(peak_heights)[-2:]
        p1, p2 = peaks[top2_idx[0]], peaks[top2_idx[1]]
        valley_slice = hist[min(p1,p2):max(p1,p2)+1]
        valley_min = valley_slice.min()
        valley_depth = min(hist[p1], hist[p2]) - valley_min
        if valley_depth > 0.1 * max(hist[p1], hist[p2]):
            valid_a = True

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
        return 0.5

    bimodal, _ = is_bimodal_valid(pix_vals)
    if bimodal:
        try:
            thresh = threshold_otsu(pix_vals)
            low_bound = np.percentile(pix_vals, 10)
            high_bound = np.percentile(pix_vals, 90)
            if thresh < low_bound or thresh > high_bound:
                bimodal = False
        except:
            bimodal = False

    if not bimodal:
        med = np.median(pix_vals)
        iqr = np.subtract(*np.percentile(pix_vals, [75, 25]))
        thresh = med + 1.5 * iqr
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

    muscle_mask = label == muscle_label
    total_csa_pixels = np.sum(muscle_mask)

    if total_csa_pixels > 0:
        muscle_pixels = normalized_image[muscle_mask]
        fat_threshold = get_fat_threshold_per_muscle(muscle_pixels)
    else:
        fat_threshold = 0.5

    features = calculate_all_muscle_features(
        normalized_image=normalized_image,
        muscle_mask=muscle_mask,
        pixel_spacing=pixel_spacing,
        fat_threshold=fat_threshold
    )

    features['pixel_spacing_x'] = pixel_spacing[0]
    features['pixel_spacing_y'] = pixel_spacing[1]
    features['muscle_area_mm2'] = features.get('Area', 0.0)
    features['Total_CSA'] = total_csa_pixels
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
    img = nib.load(normalized_image_path)
    normalized_data = img.get_fdata()

    zooms = img.header.get_zooms()
    pixel_spacing = (zooms[0], zooms[1])

    lbl = nib.load(label_path)
    label_data = lbl.get_fdata().astype(np.int32)

    assert normalized_data.shape == label_data.shape, \
        f'Image shape {normalized_data.shape} != label shape {label_data.shape}'

    slice_features = []

    for z in range(normalized_data.shape[2]):
        norm_slice = normalized_data[:, :, z]
        label_slice = label_data[:, :, z]

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


def check_csv_consistency(
    output_dir: str,
    successful_patients: List[str],
    all_features: List[Dict],
    all_features_3d: List[Dict],
    all_features_level3_cross: List[Dict],
    all_features_level3_multi: List[Dict]
) -> Dict:
    """
    检查四类特征表格的病人一致性

    Args:
        output_dir: 输出目录
        successful_patients: 成功提取特征的病人ID列表
        all_features: 2D特征列表
        all_features_3d: 3D特征列表
        all_features_level3_cross: Level 3.1特征列表
        all_features_level3_multi: Level 3.2-3.5特征列表

    Returns:
        一致性检查结果字典
    """
    result = {
        'expected_patients': set(successful_patients),
        'csv_patients_2d': set(),
        'csv_patients_3d': set(),
        'csv_patients_level3_cross': set(),
        'csv_patients_level3_multi': set(),
        'missing_in_2d': set(),
        'missing_in_3d': set(),
        'missing_in_level3_cross': set(),
        'missing_in_level3_multi': set(),
        'inconsistent_between_csv': {}
    }

    if all_features:
        df_2d = pd.DataFrame(all_features)
        result['csv_patients_2d'] = set(df_2d['patient_id'].unique())

    if all_features_3d:
        df_3d = pd.DataFrame(all_features_3d)
        result['csv_patients_3d'] = set(df_3d['patient_id'].unique())

    if all_features_level3_cross:
        df_cross = pd.DataFrame(all_features_level3_cross)
        result['csv_patients_level3_cross'] = set(df_cross['patient_id'].unique())

    if all_features_level3_multi:
        df_multi = pd.DataFrame(all_features_level3_multi)
        result['csv_patients_level3_multi'] = set(df_multi['patient_id'].unique())

    result['missing_in_2d'] = result['expected_patients'] - result['csv_patients_2d']
    result['missing_in_3d'] = result['expected_patients'] - result['csv_patients_3d']
    result['missing_in_level3_cross'] = result['expected_patients'] - result['csv_patients_level3_cross']
    result['missing_in_level3_multi'] = result['expected_patients'] - result['csv_patients_level3_multi']

    all_csv_patients = (
        result['csv_patients_2d'] |
        result['csv_patients_3d'] |
        result['csv_patients_level3_cross'] |
        result['csv_patients_level3_multi']
    )

    csv_names = ['csv_patients_2d', 'csv_patients_3d', 'csv_patients_level3_cross', 'csv_patients_level3_multi']
    csv_labels = ['2D特征', '3D特征', 'Level3.1特征', 'Level3.2-3.5特征']

    for i in range(len(csv_names)):
        for j in range(i + 1, len(csv_names)):
            set_i = result[csv_names[i]]
            set_j = result[csv_names[j]]
            if set_i != set_j:
                diff_i_not_j = set_i - set_j
                diff_j_not_i = set_j - set_i
                key = f'{csv_labels[i]} vs {csv_labels[j]}'
                result['inconsistent_between_csv'][key] = {
                    f'仅在{csv_labels[i]}': sorted(diff_i_not_j) if diff_i_not_j else [],
                    f'仅在{csv_labels[j]}': sorted(diff_j_not_i) if diff_j_not_i else []
                }

    return result


def save_extraction_report(
    output_dir: str,
    successful_patients: List[str],
    failed_patients: List[Dict],
    total_patients: int,
    consistency_result: Dict = None
):
    """
    保存特征提取检测报告

    Args:
        output_dir: 输出目录
        successful_patients: 成功提取特征的病人ID列表
        failed_patients: 失败病人的信息列表，每个元素为 {patient_id, reason} 字典
        total_patients: 总病人数
    """
    report_path = Path(output_dir) / 'extraction_report_v4.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('肌肉特征提取检测报告 (版本 v4)\n')
        f.write('=' * 80 + '\n\n')

        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

        f.write('-' * 80 + '\n')
        f.write('总体统计\n')
        f.write('-' * 80 + '\n')
        f.write(f'总病人数: {total_patients}\n')
        f.write(f'成功提取: {len(successful_patients)} ({len(successful_patients)/total_patients*100:.1f}%)\n')
        f.write(f'失败提取: {len(failed_patients)} ({len(failed_patients)/total_patients*100:.1f}%)\n')
        f.write('\n')

        f.write('-' * 80 + '\n')
        f.write('成功提取特征的病人\n')
        f.write('-' * 80 + '\n')
        if successful_patients:
            for i, patient_id in enumerate(sorted(successful_patients), 1):
                f.write(f'{i}. {patient_id}\n')
        else:
            f.write('无\n')
        f.write('\n')

        f.write('-' * 80 + '\n')
        f.write('未能提取特征的病人\n')
        f.write('-' * 80 + '\n')
        if failed_patients:
            for i, fail_info in enumerate(failed_patients, 1):
                f.write(f'{i}. {fail_info["patient_id"]}\n')
                f.write(f'   失败原因: {fail_info["reason"]}\n')
                if 'error_type' in fail_info:
                    f.write(f'   错误类型: {fail_info["error_type"]}\n')
                if 'suggestion' in fail_info:
                    f.write(f'   处理建议: {fail_info["suggestion"]}\n')
                f.write('\n')
        else:
            f.write('无\n')
        f.write('\n')

        if consistency_result:
            f.write('-' * 80 + '\n')
            f.write('四类特征表格病人一致性检查\n')
            f.write('-' * 80 + '\n')

            f.write('\n各表格病人数统计:\n')
            f.write(f'  预期成功病人数: {len(consistency_result["expected_patients"])}\n')
            f.write(f'  2D特征表格病人数: {len(consistency_result["csv_patients_2d"])}\n')
            f.write(f'  3D特征表格病人数: {len(consistency_result["csv_patients_3d"])}\n')
            f.write(f'  Level3.1特征表格病人数: {len(consistency_result["csv_patients_level3_cross"])}\n')
            f.write(f'  Level3.2-3.5特征表格病人数: {len(consistency_result["csv_patients_level3_multi"])}\n')

            has_missing = (
                consistency_result['missing_in_2d'] or
                consistency_result['missing_in_3d'] or
                consistency_result['missing_in_level3_cross'] or
                consistency_result['missing_in_level3_multi']
            )

            if has_missing:
                f.write('\n各表格缺失病人情况（相对于成功提取列表）:\n')
                if consistency_result['missing_in_2d']:
                    f.write(f'  2D特征缺失: {sorted(consistency_result["missing_in_2d"])}\n')
                if consistency_result['missing_in_3d']:
                    f.write(f'  3D特征缺失: {sorted(consistency_result["missing_in_3d"])}\n')
                if consistency_result['missing_in_level3_cross']:
                    f.write(f'  Level3.1特征缺失: {sorted(consistency_result["missing_in_level3_cross"])}\n')
                if consistency_result['missing_in_level3_multi']:
                    f.write(f'  Level3.2-3.5特征缺失: {sorted(consistency_result["missing_in_level3_multi"])}\n')
            else:
                f.write('\n各表格无缺失病人\n')

            if consistency_result['inconsistent_between_csv']:
                f.write('\n表格间不一致情况:\n')
                for compare_key, diff_info in consistency_result['inconsistent_between_csv'].items():
                    f.write(f'  {compare_key}:\n')
                    for diff_label, patients in diff_info.items():
                        if patients:
                            f.write(f'    {diff_label}: {patients}\n')
            else:
                f.write('\n所有表格间病人姓名一致\n')
            f.write('\n')

        f.write('=' * 80 + '\n')
        f.write('报告结束\n')
        f.write('=' * 80 + '\n')

    return report_path


def main():
    print('=' * 80)
    print('肌肉特征提取脚本 (版本 v4)')
    print('新增：病人特征提取成功/失败检测')
    print('=' * 80)

    PREPROCESSED_DIR = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\output"
    LABELS_DIR = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\labels"
    OUTPUT_DIR = r"E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\features_v4_202605172124"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f'输出目录: {OUTPUT_DIR}')

    patient_dirs = [d for d in Path(PREPROCESSED_DIR).iterdir() if d.is_dir()]
    print(f'找到 {len(patient_dirs)} 个病人目录')

    successful_patients = []
    failed_patients = []

    all_features = []
    all_features_3d = []
    all_features_level3_cross = []
    all_features_level3_multi = []

    for patient_dir in tqdm(patient_dirs, desc='Processing patients'):
        patient_id = patient_dir.name

        normalized_files = list(patient_dir.glob('*_normalized.nii.gz'))
        if not normalized_files:
            failed_patients.append({
                'patient_id': patient_id,
                'reason': '未找到标准化图像'
            })
            print(f'\n⚠️ 跳过病人: {patient_id} (原因: 未找到标准化图像)')
            continue

        normalized_path = str(normalized_files[0])

        label_path = Path(LABELS_DIR) / f'{patient_id}.nii.gz'
        if not label_path.exists():
            label_path = Path(LABELS_DIR) / f'{patient_id}.nii'

        if not label_path.exists():
            failed_patients.append({
                'patient_id': patient_id,
                'reason': '未找到分割标签'
            })
            print(f'\n⚠️ 跳过病人: {patient_id} (原因: 未找到分割标签)')
            continue

        print(f'处理病人: {patient_id}')
        try:
            patient_features = process_single_patient(
                normalized_path,
                str(label_path),
                patient_id,
                OUTPUT_DIR
            )
            all_features.extend(patient_features)

            patient_features_3d = process_single_patient_3d(
                normalized_path,
                str(label_path),
                patient_id
            )
            all_features_3d.extend(patient_features_3d)

            patient_features_level3_cross = process_single_patient_level3_cross_layer(
                normalized_path,
                str(label_path),
                patient_id
            )
            all_features_level3_cross.extend(patient_features_level3_cross)

            patient_features_level3_multi = process_single_patient_level3_multi_muscle(
                normalized_path,
                str(label_path),
                patient_id
            )
            all_features_level3_multi.append(patient_features_level3_multi)

            successful_patients.append(patient_id)
            print(f'\n✅ 完成病人: {patient_id}')

        except FileNotFoundError as e:
            reason = f'文件未找到: {str(e)}'
            failed_patients.append({
                'patient_id': patient_id,
                'reason': reason,
                'error_type': '文件错误',
                'suggestion': '请检查文件路径是否正确，文件是否存在'
            })
            print(f'❌ 处理 {patient_id} 失败: {reason}')
            print(f'   建议: 请检查文件路径是否正确，文件是否存在')
            import traceback
            traceback.print_exc()

        except (FileNotFoundError, IOError) as e:
            reason = f'文件读取错误: {str(e)}'
            failed_patients.append({
                'patient_id': patient_id,
                'reason': reason,
                'error_type': '文件错误',
                'suggestion': '请检查文件路径是否正确，文件是否存在'
            })
            print(f'❌ 处理 {patient_id} 失败: {reason}')
            print(f'   建议: 请检查文件路径是否正确，文件是否存在')
            import traceback
            traceback.print_exc()

        except (AssertionError, ValueError) as e:
            reason = f'数据处理错误: {str(e)}'
            failed_patients.append({
                'patient_id': patient_id,
                'reason': reason,
                'error_type': '数据处理错误',
                'suggestion': '请检查图像数据格式是否正确'
            })
            print(f'❌ 处理 {patient_id} 失败: {reason}')
            print(f'   建议: 请检查图像数据格式是否正确')
            import traceback
            traceback.print_exc()

        except MemoryError as e:
            reason = f'内存不足错误: {str(e)}'
            failed_patients.append({
                'patient_id': patient_id,
                'reason': reason,
                'error_type': '内存错误',
                'suggestion': '请尝试减少同时处理的数据量或增加系统内存'
            })
            print(f'❌ 处理 {patient_id} 失败: {reason}')
            print(f'   建议: 请尝试减少同时处理的数据量或增加系统内存')
            import traceback
            traceback.print_exc()

        except Exception as e:
            reason = f'处理异常: {str(e)}'
            error_type = str(type(e).__name__)
            failed_patients.append({
                'patient_id': patient_id,
                'reason': reason,
                'error_type': error_type,
                'suggestion': '请查看详细错误信息以定位问题'
            })
            print(f'❌ 处理 {patient_id} 失败: {reason}')
            print(f'   错误类型: {error_type}')
            print(f'   建议: 请查看详细错误信息以定位问题')
            import traceback
            traceback.print_exc()

    if all_features:
        df = pd.DataFrame(all_features)
        output_file = Path(OUTPUT_DIR) / 'muscle_features_2d_v4.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f'\n2D特征已保存到: {output_file}')

        print('\n2D特征提取完成!')
        print(f'处理了 {len(df["patient_id"].unique())} 个病人')
        print(f'计算了 {len(df)} 个切片的特征')

        print('\n新增特征示例:')
        sample_cols = ['patient_id', 'slice_index', 'muscle_name',
                      'pixel_spacing_x', 'pixel_spacing_y', 'muscle_area_mm2']
        print(df[sample_cols].head().to_string())

    if all_features_3d:
        df_3d = pd.DataFrame(all_features_3d)
        output_file_3d = Path(OUTPUT_DIR) / 'muscle_features_3d_v4.csv'
        df_3d.to_csv(output_file_3d, index=False, encoding='utf-8-sig')
        print(f'\n3D特征已保存到: {output_file_3d}')

    if all_features_level3_cross:
        df_level3_cross = pd.DataFrame(all_features_level3_cross)
        output_file_level3_cross = Path(OUTPUT_DIR) / 'muscle_features_level3_cross_v4.csv'
        df_level3_cross.to_csv(output_file_level3_cross, index=False, encoding='utf-8-sig')
        print(f'Level 3.1跨层梯度特征已保存到: {output_file_level3_cross}')

    if all_features_level3_multi:
        df_level3_multi = pd.DataFrame(all_features_level3_multi)
        output_file_level3_multi = Path(OUTPUT_DIR) / 'muscle_features_level3_multi_v4.csv'
        df_level3_multi.to_csv(output_file_level3_multi, index=False, encoding='utf-8-sig')
        print(f'Level 3.2-3.5多肌肉关系特征已保存到: {output_file_level3_multi}')

    consistency_result = check_csv_consistency(
        OUTPUT_DIR,
        successful_patients,
        all_features,
        all_features_3d,
        all_features_level3_cross,
        all_features_level3_multi
    )

    report_path = save_extraction_report(
        OUTPUT_DIR,
        successful_patients,
        failed_patients,
        len(patient_dirs),
        consistency_result
    )
    print(f'\n特征提取检测报告已保存到: {report_path}')

    print('\n' + '=' * 80)
    print('特征提取检测摘要')
    print('=' * 80)
    print(f'总病人数: {len(patient_dirs)}')
    print(f'成功提取: {len(successful_patients)} ({len(successful_patients)/len(patient_dirs)*100:.1f}%)')
    print(f'失败提取: {len(failed_patients)} ({len(failed_patients)/len(patient_dirs)*100:.1f}%)')

    if not all_features and not all_features_3d and not all_features_level3_cross and not all_features_level3_multi:
        print('未提取到任何特征')


if __name__ == '__main__':
    main()
