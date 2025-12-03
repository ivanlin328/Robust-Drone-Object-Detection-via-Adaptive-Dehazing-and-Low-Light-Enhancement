import os
import shutil
import glob
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ==================== 使用者設定區 ====================

# 1. 輸入路徑：請修改為您 RTTS 資料集解壓縮後的真實路徑
# 通常 RTTS 資料夾結構內會有 'Annotations' (放XML) 和 'JPEGImages' (放圖片)
SOURCE_XML_DIR = r"/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/RTTS/Annotations"  
SOURCE_IMG_DIR = r"/Users/ivanlin328/Desktop/UCSD/Fall 2025/ECE 253/ECE 253 Final Project/RTTS/JPEGImages"

# 2. 輸出路徑：轉換後的資料要存在哪裡
OUTPUT_DIR = r"new_dataset/rtts_car_person_yolo"

# 3. 類別過濾與 ID 映射
# YOLO 需要從 0 開始的整數 ID
# 這裡指定只保留 car 和 person，並重新編號
TARGET_CLASSES = {
    'car': 2,
    'person': 0
}

# 4. 分割比例 (0.8 代表 80% 訓練集, 20% 驗證集)
TRAIN_RATIO = 0.08

# 5. 是否保留「沒有目標」的圖片？
# True: 即使一張圖裡沒有車和人，也保留該圖 (產生空的 txt)，作為負樣本 (建議 True 以降低誤判)
# False: 只保留包含車或人的圖片
KEEP_NO_TARGET_IMAGES = False 

# ====================================================

def convert_to_yolo_bbox(size, box):
    """將 VOC xml 座標 (xmin, xmax, ymin, ymax) 轉為 YOLO (x_center, y_center, w, h)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    # 計算中心點與長寬
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    # 歸一化
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)

def setup_directories():
    """建立 YOLOv8 標準資料夾結構"""
    if os.path.exists(OUTPUT_DIR):
        user_input = input(f"警告：輸出目錄 '{OUTPUT_DIR}' 已存在。是否刪除並重新建立？(y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(OUTPUT_DIR)
        else:
            print("已取消操作。")
            exit()

    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
    print(f"資料夾結構已建立於：{OUTPUT_DIR}")

def find_image_file(base_name, search_dir):
    """嘗試尋找對應的圖片檔案 (支援 .png, .jpg, .jpeg)"""
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        file_path = os.path.join(search_dir, base_name + ext)
        if os.path.exists(file_path):
            return file_path
    return None

def process_dataset():
    setup_directories()

    # 取得所有 XML 檔案
    xml_files = glob.glob(os.path.join(SOURCE_XML_DIR, "*.xml"))
    if not xml_files:
        print("錯誤：找不到任何 XML 檔案，請檢查 SOURCE_XML_DIR 路徑是否正確。")
        return

    # 打亂順序以隨機分配
    random.shuffle(xml_files)
    
    split_point = int(len(xml_files) * TRAIN_RATIO)
    train_files = xml_files[:split_point]
    val_files = xml_files[split_point:]
    
    splits = [('train', train_files), ('val', val_files)]
    
    counts = {'car': 0, 'person': 0, 'images_kept': 0, 'images_skipped': 0}

    print(f"開始處理... 總共 {len(xml_files)} 張標註檔")
    
    for split_name, files in splits:
        print(f"正在處理 {split_name} 集...")
        
        for xml_file in tqdm(files):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 取得圖片尺寸
            size_node = root.find('size')
            img_w = int(size_node.find('width').text)
            img_h = int(size_node.find('height').text)
            
            # 準備 YOLO 格式資料
            yolo_lines = []
            has_target_class = False
            
            for obj in root.iter('object'):
                class_name = obj.find('name').text.lower().strip()
                
                if class_name in TARGET_CLASSES:
                    class_id = TARGET_CLASSES[class_name]
                    xmlbox = obj.find('bndbox')
                    
                    # 讀取座標並限制在圖片範圍內 (防呆)
                    xmin = max(0, float(xmlbox.find('xmin').text))
                    xmax = min(img_w, float(xmlbox.find('xmax').text))
                    ymin = max(0, float(xmlbox.find('ymin').text))
                    ymax = min(img_h, float(xmlbox.find('ymax').text))
                    
                    # 轉換
                    bbox = convert_to_yolo_bbox((img_w, img_h), (xmin, xmax, ymin, ymax))
                    
                    # 格式：class_id x y w h
                    yolo_lines.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                    
                    counts[class_name] += 1
                    has_target_class = True
            
            # 判斷是否保存此圖片
            # 1. 圖片中有我們要的類別 -> 存
            # 2. 圖片中沒有，但設定了 KEEP_NO_TARGET_IMAGES -> 存 (作為背景負樣本)
            should_save = has_target_class or KEEP_NO_TARGET_IMAGES
            
            if should_save:
                # 找出對應圖片
                file_basename = os.path.splitext(os.path.basename(xml_file))[0]
                img_src_path = find_image_file(file_basename, SOURCE_IMG_DIR)
                
                if img_src_path:
                    # 複製圖片
                    img_dst_path = os.path.join(OUTPUT_DIR, 'images', split_name, os.path.basename(img_src_path))
                    shutil.copy2(img_src_path, img_dst_path)
                    
                    # 寫入 Label txt
                    txt_dst_path = os.path.join(OUTPUT_DIR, 'labels', split_name, file_basename + ".txt")
                    with open(txt_dst_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    
                    counts['images_kept'] += 1
                else:
                    print(f"\n警告：找不到 XML 對應的圖片：{file_basename}")
            else:
                counts['images_skipped'] += 1

    print("\n" + "="*30)
    print("處理完成！")
    print(f"輸出目錄：{OUTPUT_DIR}")
    print(f"保留圖片數：{counts['images_kept']}")
    print(f"跳過圖片數：{counts['images_skipped']}")
    print(f"標註統計 - Car: {counts['car']}, Person: {counts['person']}")
    print("="*30)
    print("\n下一步：請建立 data.yaml 檔案並指向上述輸出目錄。")

if __name__ == "__main__":
    process_dataset()