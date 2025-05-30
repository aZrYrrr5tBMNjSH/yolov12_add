import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

# ==== 1. 设置路径 ====
nifti_root = r"D:\CAMUS_public\database_nifti"
split_root = r"D:\CAMUS_public\database_split"
output_root = r"D:\CAMUS_YOLO1"

output_img_dir = os.path.join(output_root, "images")
output_label_dir = os.path.join(output_root, "labels")
yaml_path = os.path.join(output_root, "data.yaml")

# ==== 2. 读取分组信息 ====
def load_split_txt(txt_path):
    with open(txt_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

train_patients = load_split_txt(os.path.join(split_root, "subgroup_training.txt"))
val_patients = load_split_txt(os.path.join(split_root, "subgroup_validation.txt"))
test_patients = load_split_txt(os.path.join(split_root, "subgroup_testing.txt"))

def get_split_folder(patient_id):
    if patient_id in train_patients:
        return "train"
    elif patient_id in val_patients:
        return "val"
    elif patient_id in test_patients:
        return "test"
    return None

# ==== 3. 创建输出目录 ====
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_img_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_label_dir, split), exist_ok=True)

# ==== 4. 遍历患者，转换数据 ====
for patient in tqdm(sorted(os.listdir(nifti_root))):
    p_dir = os.path.join(nifti_root, patient)
    if not os.path.isdir(p_dir):
        continue

    split = get_split_folder(patient)
    if split is None:
        continue

    for phase in ["ED", "ES"]:
        for view in ["2CH", "4CH"]:
            img_file = f"{patient}_{view}_{phase}.nii.gz"
            gt_file = f"{patient}_{view}_{phase}_gt.nii.gz"

            img_path = os.path.join(p_dir, img_file)
            gt_path = os.path.join(p_dir, gt_file)

            if not os.path.exists(img_path) or not os.path.exists(gt_path):
                continue

            img_nifti = nib.load(img_path)
            gt_nifti = nib.load(gt_path)
            img_data = img_nifti.get_fdata()
            gt_data = gt_nifti.get_fdata()

            if img_data.ndim == 2:
                slices = [(img_data, gt_data, 0)]
            else:
                slices = [(img_data[:, :, i], gt_data[:, :, i], i) for i in range(img_data.shape[-1])]

            for img_slice, gt_slice, i in slices:
                img_uint8 = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_name = f"{patient}_{view}_{phase}_{i}.png"
                img_out_path = os.path.join(output_img_dir, split, img_name)
                label_out_path = os.path.join(output_label_dir, split, img_name.replace(".png", ".txt"))

                h_img, w_img = img_uint8.shape
                objects_found = 0

                with open(label_out_path, "w") as f:
                    for cls_id in [1, 2, 3]:  # 1: LV, 2: LA, 3: LVM
                        mask = (gt_slice == cls_id).astype(np.uint8)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            x_center = (x + w / 2) / w_img
                            y_center = (y + h / 2) / h_img
                            w_norm = w / w_img
                            h_norm = h / h_img
                            f.write(f"{cls_id - 1} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                            objects_found += 1

                if objects_found > 0:
                    cv2.imwrite(img_out_path, img_uint8)
                else:
                    os.remove(label_out_path)  # 没有对象则删除 label 文件

# ==== 5. 写入 data.yaml ====
train_path = os.path.join(output_img_dir, "train").replace("\\", "/")
val_path = os.path.join(output_img_dir, "val").replace("\\", "/")
test_path = os.path.join(output_img_dir, "test").replace("\\", "/")

with open(yaml_path, "w") as f:
    f.write("train: " + train_path + "\n")
    f.write("val: " + val_path + "\n")
    f.write("test: " + test_path + "\n\n")
    f.write("nc: 3\n")
    f.write("names: ['left_ventricle', 'left_atrium', 'left_ventricle_wall']\n")

print("\n✅ CAMUS 多类数据集转换完成！可用于 YOLOv12 训练。")
