import os
from collections import Counter

label_dir = r"D:\CAMUS_YOLO1\labels\train"  # 修改为你的训练标签路径

counter = Counter()
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(label_dir, filename), "r") as f:
            lines = f.readlines()
            for line in lines:
                cls_id = line.strip().split()[0]
                counter[cls_id] += 1

print("训练标签类别分布：")
for cls_id in sorted(counter.keys()):
    print(f"类别 {cls_id} 数量：{counter[cls_id]}")

