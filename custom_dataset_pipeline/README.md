# 自建数据集 → nnUNet v2 处理流程

专为单一数据集设计，去掉了LPM相关的冗余部分。

## 📁 文件说明

| 文件 | 功能 |
|-----|------|
| `00_run_all.py` | 一键运行完整流程（推荐） |
| `01_convert_dicom.py` | DICOM→nii.gz转换（支持Excel读取series number） |
| `02_check_and_convert_labels.py` | 标签检查和转换 |
| `03_organize_nnunet.py` | 组织到nnUNet格式 + 自动验证 |

## 🚀 快速开始

### 前置依赖
```bash
pip install pydicom nibabel numpy pandas openpyxl
```

### 方法1：一键运行（推荐）

1. 编辑 `00_run_all.py`，修改配置部分
2. 运行：
```bash
python 00_run_all.py
```

### 方法2：分步运行

#### 步骤1：DICOM转换
编辑 `01_convert_dicom.py` 并运行：
```bash
python convert_dicom_01.py
```

#### 步骤2：标签检查
编辑 `02_check_and_convert_labels.py` 并运行：
```bash
python check_and_convert_labels_02.py
```

#### 步骤3：组织到nnUNet
编辑 `03_organize_nnunet.py` 并运行：
```bash
python organize_nnunet_03.py
```

## 📊 数据流程

```
原始数据
  ↓
[01] DICOM → nii.gz (读取Excel的series number)
  ↓
[02] 标签检查 (确认标签值)
  ↓ (可选) 标签转换
[03] 组织到nnUNet格式 + 自动验证
  ↓
nnUNet_raw/DatasetXXX_Name/ ✓
```

## ✅ 自动验证功能

组织阶段会自动验证：
- ✓ 图像和标注形状一致
- ✓ 图像和标注Affine矩阵一致
- ✓ 标签值在预期范围内

## 📋 数据集结构要求

```
您的数据集根目录/
├── 病人信息.xlsx           (姓名, 轴位序列号Se, 影片数量)
├── 张三/
│   ├── IM-0001-0001.dcm
│   ├── IM-0001-0002.dcm
│   ├── ...
│   └── zhangsan.nii.gz     (标注文件)
├── 李四/
│   ├── IM-0001-0001.dcm
│   ├── ...
│   └── lisi.nii.gz
└── ...
```

## 🎯 输出结果

```
nnUNet_raw/Dataset101_LumbarMuscle/
├── dataset.json
├── imagesTr/
│   ├── zhangsan_0000.nii.gz
│   ├── lisi_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── zhangsan.nii.gz
    ├── lisi.nii.gz
    └── ...
```

## 🏃 接下来

```bash
# 设置环境变量
nnUNetv2_plan_and_preprocess -d 101

# 开始训练
nnUNetv2_train 101 3d_fullres 0
```
