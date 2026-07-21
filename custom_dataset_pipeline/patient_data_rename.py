import os
import csv
import re
from pathlib import Path

# ==================== 配置区域 ====================
IMAGE_DIR = r"D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\images"
CSV_PATH = r"D:\dataset\yaozhui\QiLuhospital\feature_extraction_20260421dataset_prediction_results\Dataset201_LumbarMuscle_predictions800\PATIENT_LIST_FILE.csv"
HAS_HEADER = True   # 如果CSV第一行是表头，设为True
DRY_RUN = True      # 试运行模式（改为False时实际执行）
# ================================================

def main():
    # ---------- 读取CSV，构建拼音->ID映射 ----------
    pinyin_to_id = {}
    try:
        # 关键修改：使用 gbk 编码
        with open(CSV_PATH, 'r', encoding='gbk') as f:
            reader = csv.reader(f)
            if HAS_HEADER:
                header = next(reader)  # 跳过表头
            for row in reader:
                if len(row) < 3:
                    continue
                patient_id = row[0].strip()
                chinese_name = row[1].strip()   # 暂未使用
                pinyin = row[2].strip()
                if pinyin:
                    if pinyin in pinyin_to_id:
                        print(f"警告：拼音 '{pinyin}' 重复，将覆盖之前的ID {pinyin_to_id[pinyin]} -> {patient_id}")
                    pinyin_to_id[pinyin] = patient_id
        print(f"成功读取 {len(pinyin_to_id)} 个病人记录。")
    except FileNotFoundError:
        print(f"错误：找不到CSV文件 '{CSV_PATH}'，请检查路径。")
        return
    except Exception as e:
        print(f"读取CSV时出错：{e}")
        return

    # ---------- 遍历并重命名 ----------
    image_path = Path(IMAGE_DIR)
    if not image_path.exists():
        print(f"错误：图像文件夹 '{IMAGE_DIR}' 不存在。")
        return

    pattern = re.compile(r'^(.+)_0000\.nii\.gz$')
    renamed_count = 0
    missing_count = 0
    skip_count = 0

    for file_path in image_path.glob("*.nii.gz"):
        filename = file_path.name
        match = pattern.match(filename)
        if not match:
            print(f"跳过不符合格式的文件: {filename}")
            skip_count += 1
            continue

        pinyin = match.group(1)
        if pinyin in pinyin_to_id:
            patient_id = pinyin_to_id[pinyin]
            new_filename = f"{patient_id}_{pinyin}_0000.nii.gz"
            new_path = file_path.parent / new_filename

            if new_path.exists():
                print(f"警告：目标文件已存在，跳过 {new_filename}")
                skip_count += 1
                continue

            if DRY_RUN:
                print(f"[试运行] 将重命名: {filename} -> {new_filename}")
            else:
                try:
                    file_path.rename(new_path)
                    print(f"已重命名: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"重命名失败: {filename} -> {new_filename}, 错误: {e}")
        else:
            print(f"警告：未找到拼音 '{pinyin}' 对应的ID，跳过文件: {filename}")
            missing_count += 1

    if DRY_RUN:
        print(f"\n试运行结束，共需重命名 {renamed_count} 个文件，缺失映射 {missing_count} 个，跳过 {skip_count} 个。")
    else:
        print(f"\n重命名完成，成功重命名 {renamed_count} 个文件，缺失映射 {missing_count} 个，跳过 {skip_count} 个。")

if __name__ == "__main__":
    main()