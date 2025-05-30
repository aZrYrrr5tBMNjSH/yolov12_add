import os
import nibabel as nib
import numpy as np
import imageio
from tqdm import tqdm
import json

# 设置路径
img_root = "D:/CAMUS_YOLO/cropped_all"
mask_root = "D:/CAMUS_YOLO/cropped_all_masks"
output_dir = "D:/CAMUS_YOLO/nnunet_dataset"
imagesTr_dir = os.path.join(output_dir, "imagesTr")
labelsTr_dir = os.path.join(output_dir, "labelsTr")
imagesTs_dir = os.path.join(output_dir, "imagesTs")

# 创建必要的文件夹
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)
os.makedirs(imagesTs_dir, exist_ok=True)

# 函数：图像/mask 转为 NIfTI 格式
def save_nifti(image_array, save_path):
    nifti_img = nib.Nifti1Image(image_array, affine=np.eye(4))
    nib.save(nifti_img, save_path)

# 收集用于生成 dataset.json 的 ID 列表
training_cases = []
test_cases = []

# 处理 train 和 val（训练数据）
for subset in ['train', 'val']:
    img_dir = os.path.join(img_root, subset)
    mask_dir = os.path.join(mask_root, subset)
    for fname in tqdm(os.listdir(img_dir), desc=f'Processing {subset}'):
        if not fname.endswith('.jpg'):
            continue
        case_id = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, case_id + ".png")

        # 读取并保存图像和 mask
        image = imageio.v2.imread(img_path)
        mask = imageio.v2.imread(mask_path)

        # 添加通道维度
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        else:
            image = image.transpose(2, 0, 1)

        save_nifti(image, os.path.join(imagesTr_dir, f"{case_id}_0000.nii.gz"))
        save_nifti(mask, os.path.join(labelsTr_dir, f"{case_id}.nii.gz"))
        training_cases.append(case_id)

# 处理 test（测试数据）
test_dir = os.path.join(img_root, 'test')
for fname in tqdm(os.listdir(test_dir), desc='Processing test'):
    if not fname.endswith('.jpg'):
        continue
    case_id = os.path.splitext(fname)[0]
    img_path = os.path.join(test_dir, fname)
    image = imageio.v2.imread(img_path)

    if image.ndim == 2:
        image = image[np.newaxis, ...]
    else:
        image = image.transpose(2, 0, 1)

    save_nifti(image, os.path.join(imagesTs_dir, f"{case_id}_0000.nii.gz"))
    test_cases.append(case_id)

# 创建 dataset.json
dataset_dict = {
    "name": "CAMUS_YOLO",
    "description": "CAMUS cropped and labeled by YOLOv12",
    "tensorImageSize": "3D",
    "reference": "Generated from YOLO",
    "licence": "CC BY-SA 4.0",
    "release": "1.0",
    "modality": {"0": "RGB"},
    "labels": {
        "0": "background",
        "1": "left_ventricle",
        "2": "left_atrium",
        "3": "left_ventricle_wall"
    },
    "numTraining": len(training_cases),
    "numTest": len(test_cases),
    "training": [
        {
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        } for case_id in training_cases
    ],
    "test": [
        f"./imagesTs/{case_id}_0000.nii.gz" for case_id in test_cases
    ]
}

with open(os.path.join(output_dir, "dataset.json"), "w") as f:
    json.dump(dataset_dict, f, indent=4)

output_dir
