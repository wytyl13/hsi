#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/27 14:30
@Author  : weiyutao
@File    : label_studio_export.py
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/02 18:43
@Author  : weiyutao
@File    : data_build.py
"""

import os
import json
import shutil
import glob
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_single_file(args):
    file_stem, source_dir, label_dir, output_dir, target_subset, classes = args
    
    # 1. 寻找图片 (从 source_dir 获取)
    img_path = os.path.join(source_dir, f"{file_stem}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(source_dir, f"{file_stem}.jpg")
        if not os.path.exists(img_path):
            return False
            
    dst_img = os.path.join(output_dir, 'images', target_subset, os.path.basename(img_path))
    dst_label = os.path.join(output_dir, 'labels', target_subset, f"{file_stem}.txt")
    
    # 标签文件路径候选
    txt_path = os.path.join(label_dir, f"{file_stem}.txt")
    json_path = os.path.join(label_dir, f"{file_stem}.json")
    
    has_valid_label = False
    
    # 2. 策略 A：修改点 1 - 只要 TXT 存在就直接拷贝（哪怕是空白的，YOLO 视其为负样本）
    if os.path.exists(txt_path):
        shutil.copy(txt_path, dst_label)
        has_valid_label = True
        
    # 3. 策略 B：如果没有 TXT，则检查 JSON 并进行解析转换
    elif os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        img_w, img_h = data.get('imageWidth'), data.get('imageHeight')
        
        with open(dst_label, 'w', encoding='utf-8') as f:
            for shape in data.get('shapes', []):
                label = shape['label']
                if label not in classes: continue
                
                class_id = classes.index(label)
                points = shape['points']
                if len(points) < 3: continue
                
                # --- 直接对原始多边形坐标进行归一化 ---
                normalized_points = []
                for pt in points:
                    x = max(0.0, min(1.0, pt[0] / img_w))
                    y = max(0.0, min(1.0, pt[1] / img_h))
                    normalized_points.extend([f"{x:.6f}", f"{y:.6f}"])
                    
                f.write(f"{class_id} " + " ".join(normalized_points) + "\n")
                
        # 修改点 2 - 无论 JSON 里有没有符合要求的目标，只要有对应的图片和生成的空/非空 txt，都保留
        has_valid_label = True
            
    # 4. 拷贝图片 (现在包含正常标注的图和无目标的负样本背景图)
    if has_valid_label:
        shutil.copy(img_path, dst_img)
        return True
        
    return False



class YoloSegDatasetBuilder:
    def __init__(self, source_dir, output_dir, classes, label_dir=None, split_ratios=(0.8, 0.1, 0.1)):
        self.source_dir = source_dir
        self.label_dir = label_dir if label_dir else source_dir
        self.output_dir = output_dir
        self.classes = classes
        self.split_ratios = split_ratios
        self.sub_dirs = ['train', 'val', 'test']
        
    def setup_directories(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        for subset in self.sub_dirs:
            os.makedirs(os.path.join(self.output_dir, 'images', subset), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'labels', subset), exist_ok=True)

    def build(self):
        self.setup_directories()
        
        # 同时查找 .txt 和 .json 文件
        txt_files = glob.glob(os.path.join(self.label_dir, "*.txt"))
        json_files = glob.glob(os.path.join(self.label_dir, "*.json"))
        
        # 提取文件名并去重 (防止同一张图既有 txt 又有 json 导致重复)
        file_stems = set()
        for f in txt_files + json_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            if stem != "classes":  # 排除掉常见的类别定义文件 classes.txt
                file_stems.add(stem)
                
        file_stems = list(file_stems)
        
        print(f"🔍 发现 {len(file_stems)} 个标注文件目标(含 txt/json)，正在随机打乱...")
        random.shuffle(file_stems)
        
        # 划分切片
        total = len(file_stems)
        train_end = int(total * self.split_ratios[0])
        val_end = train_end + int(total * self.split_ratios[1])
        
        tasks_map = {
            'train': file_stems[:train_end],
            'val': file_stems[train_end:val_end],
            'test': file_stems[val_end:]
        }
        
        # 准备进程池参数
        process_args = []
        for subset, stems in tasks_map.items():
            for stem in stems:
                process_args.append((stem, self.source_dir, self.label_dir, self.output_dir, subset, self.classes))
                
        # ⚡ 启动多进程加速
        max_workers = max(1, multiprocessing.cpu_count() - 2)
        print(f"🚀 启动多进程处理引擎，调用 {max_workers} 个 CPU 核心...")
        
        success_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, arg): arg for arg in process_args}
            for future in tqdm(as_completed(futures), total=len(process_args), desc="📦 并行构建数据集"):
                if future.result():
                    success_count += 1
                    
        self.generate_yaml()
        print(f"\n✅ 数据集构建完成！成功处理并归档 {success_count} 组数据。")

    def generate_yaml(self):
        yaml_content = f"""
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val
test: images/test

nc: {len(self.classes)}
names: {self.classes}
"""
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
            f.write(yaml_content.strip())
            

if __name__ == "__main__":
    # SOURCE_DIR = "/home/weiyutao/industrial_vision_data/gold_ore_images_train" 
    # OUTPUT_DIR = "/home/weiyutao/industrial_vision_data/gold_ore_images_1_train_20260402" 
    # 假设你的标签在另一个单独的目录，如果没有可以设为 None 或者删掉这个传参
    LABEL_DIR = "label_studio_labels_txt" 
    
    SOURCE_DIR = r"F:\weiyutao\work\data\hsi\yolo_dataset\images" 
    OUTPUT_DIR = "alsi_seg_20260523_v2" 
    CLASSES = ["l", "g", "m"]
    
    builder = YoloSegDatasetBuilder(
        source_dir=SOURCE_DIR, 
        label_dir=LABEL_DIR,     # 新增：指定独立的标签目录
        output_dir=OUTPUT_DIR, 
        classes=CLASSES,
        split_ratios=(0.8, 0.1, 0.1)
        # smooth_factor 已经彻底移除
    )
    builder.build()