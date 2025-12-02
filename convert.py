import os
import glob
import shutil
from PIL import Image


IMG_DIR = "/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/VisDrone2019-DET-train/images"
ANN_DIR = "/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/VisDrone2019-DET-train/annotations"

OUT_IMG_DIR = "/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/visdrone_yolo/images/train"
OUT_LAB_DIR = "/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/visdrone_yolo/labels/train"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LAB_DIR, exist_ok=True)

def visdrone_to_yolo_cls(cat_id: int):
    """
    把 VisDrone 的 category_id 映射到你的 YOLO 類別：
    0: person, 1: ignore, 2: car
    """
    if cat_id in [1, 2]:      # pedestrian, people
        return 0              # person
    elif cat_id == 4:         # car
        return 2              # car
    else:
        return None           # 其他類別忽略

ann_files = sorted(glob.glob(os.path.join(ANN_DIR, "*.txt")))
print(f"Found {len(ann_files)} annotation files")

for ann_path in ann_files:
    base = os.path.splitext(os.path.basename(ann_path))[0]
    img_name = base + ".jpg"  # VisDrone 是 jpg
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"[WARN] Image not found for {ann_path}, skip")
        continue

    # 讀圖片尺寸
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    yolo_lines = []

    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            if len(vals) < 6:
                continue

            x, y, w, h = map(float, vals[:4])
            # score = float(vals[4])  # 如果要用可以讀
            cat_id = int(vals[5])

            # 忽略 category 0 (ignored region)
            if cat_id == 0:
                continue

            cls_id = visdrone_to_yolo_cls(cat_id)
            if cls_id is None:
                continue  # 不要的類別

            # 轉成 YOLO normalized format
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h

            # 避免超出 0~1
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if not yolo_lines:
        continue

    out_img_path = os.path.join(OUT_IMG_DIR, img_name)
    shutil.copy(img_path, out_img_path)


    out_lab_path = os.path.join(OUT_LAB_DIR, base + ".txt")
    with open(out_lab_path, "w") as f:
        f.write("\n".join(yolo_lines))

print("Done converting VisDrone → YOLO.")
