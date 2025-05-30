import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

model_path = r'D:\yolov12\runs\detect\camus_yolov12_multi_class2\weights\best.pt'
root_img_dir = r'D:\CAMUS_YOLO\images'        # 含 train/val/test文件夹
save_cropped_img_dir = r'D:\CAMUS_YOLO\cropped_all'        # 裁剪图像保存目录
save_cropped_mask_dir = r'D:\CAMUS_YOLO\cropped_all_masks' # 生成mask保存目录
bbox_json_path = r'D:\CAMUS_YOLO\image_bboxes.json'        # bbox json路径

model = YOLO(model_path)

image_bboxes = {}

# Step 1: 推理 + 保存bbox
for subset in ['train', 'val', 'test']:
    img_dir = os.path.join(root_img_dir, subset)
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    print(f"推理 {subset} 集合，共 {len(img_files)} 张图片")
    for img_name in tqdm(img_files):
        img_path = os.path.join(img_dir, img_name)
        results = model(img_path)
        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        bboxes_list = []
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = box.astype(int)
            bboxes_list.append([int(x1), int(y1), int(x2), int(y2), int(cls), i])

        image_bboxes[f"{subset}/{img_name}"] = bboxes_list

# 保存bbox json
with open(bbox_json_path, 'w') as f:
    json.dump(image_bboxes, f)
print(f"✅ BBox信息已保存到 {bbox_json_path}")

# Step 2: 根据bbox生成对应mask和裁剪图像
with open(bbox_json_path, 'r') as f:
    image_bboxes = json.load(f)

for subset in ['train', 'val', 'test']:
    img_dir = os.path.join(root_img_dir, subset)
    cropped_img_dir = os.path.join(save_cropped_img_dir, subset)
    cropped_mask_dir = os.path.join(save_cropped_mask_dir, subset)
    os.makedirs(cropped_img_dir, exist_ok=True)
    os.makedirs(cropped_mask_dir, exist_ok=True)

    print(f"开始裁剪并生成mask：{subset}")
    for key in tqdm(image_bboxes.keys()):
        if not key.startswith(subset + '/'):
            continue
        img_name = key.split('/', 1)[1]
        bboxes = image_bboxes[key]

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"图片读取失败: {img_path}")
            continue

        h, w = img.shape[:2]

        for bbox in bboxes:
            x1, y1, x2, y2, cls, idx = bbox

            # 裁剪图像
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue
            crop_img_name = f"{os.path.splitext(img_name)[0]}_cls{cls}_idx{idx}.jpg"
            cv2.imwrite(os.path.join(cropped_img_dir, crop_img_name), crop_img)

            # 生成整图大小的mask，bbox区域为1，其余为0，单通道uint8
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255  # 注意255，方便查看与保存为png

            # 裁剪对应的mask区域，大小和crop_img一致
            crop_mask = mask[y1:y2, x1:x2]
            if crop_mask.size == 0:
                continue
            crop_mask_name = f"{os.path.splitext(img_name)[0]}_cls{cls}_idx{idx}.png"
            cv2.imwrite(os.path.join(cropped_mask_dir, crop_mask_name), crop_mask)

print("✅ 所有子集裁剪和mask生成完成！")
