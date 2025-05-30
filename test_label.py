import os
from collections import Counter

def test_label_distribution():
    label_root = r"D:/CAMUS_YOLO1/labels"  # 你的labels目录
    splits = ['train', 'val', 'test']

    # 类别名称顺序，必须和data.yaml对应
    class_names = ['left_atrium', 'left_ventricle', 'left_ventricle_wall']

    for split in splits:
        split_dir = os.path.join(label_root, split)
        counter = Counter()
        total_files = 0
        for fname in os.listdir(split_dir):
            if not fname.endswith('.txt'):
                continue
            total_files += 1
            fpath = os.path.join(split_dir, fname)
            with open(fpath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                    class_id = int(line.split()[0])
                    counter[class_id] += 1

        print(f"---- {split} ----")
        print(f"Total label files: {total_files}")
        for cid, cname in enumerate(class_names):
            print(f"Class {cid} ({cname}): {counter[cid]} boxes")
        print()

    # 如果你想做简单断言，确保标签数量不为0（任选一条）
    assert sum(counter.values()) > 0

if __name__ == "__main__":
    test_label_distribution()
