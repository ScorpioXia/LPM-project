import nibabel as nib
import numpy as np

# 加载一个标签文件
label_path = r'E:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\labels\anzhonghua.nii.gz'
img = nib.load(label_path)
data = img.get_fdata()

print('标签文件形状:', data.shape)
print()
print('各标签值的像素统计:')
for label_val in range(7):
    count = np.sum(data == label_val)
    percentage = count / data.size * 100
    print(f'  标签 {label_val}: {count:>10} 像素 ({percentage:>6.2f}%)')

# 检查非零标签的分布
print()
print('='*60)
print('根据 dataset.json 和 run_all_00.py，标签映射为:')
print('  1: psoas_left (腰大肌左)')
print('  2: psoas_right (腰大肌右)')
print('  3: erector_spinae_left (竖脊肌左)')
print('  4: erector_spinae_right (竖脊肌右)')
print('  5: multifidus_left (多裂肌左)')
print('  6: multifidus_right (多裂肌右)')
print('='*60)

# 分析中间切片的标签分布
mid_slice = data.shape[2] // 2
print(f'\n中间切片 (z={mid_slice}) 的标签分布:')
for label_val in range(1, 7):
    count = np.sum(data[:, :, mid_slice] == label_val)
    print(f'  标签 {label_val}: {count:>8} 像素')