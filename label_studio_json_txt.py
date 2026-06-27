#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/02 18:25
@Author  : weiyutao
@File    : json_txt.py
"""

import json
import os
from urllib.parse import unquote

# 1. 配置文件路径与类别映射
json_file = 'project-13-at-2026-05-23-06-43-aa7c4f4b.json'
output_dir = 'label_studio_labels_txt'

class_mapping = {
    "l": 0,
    "g": 1,
    "m": 2
}

os.makedirs(output_dir, exist_ok=True)

with open(json_file, 'r', encoding='utf-8') as f:
    tasks = json.load(f)

# 2. 解析任务并生成 YOLO 格式 txt 文件
for task in tasks:
    image_url = task['data']['image']
    original_path = unquote(image_url.split('?d=')[-1])
    original_filename = os.path.basename(original_path)
    original_filename = original_filename.split("-")[1]
    txt_filename = os.path.splitext(original_filename)[0] + '.txt'
    txt_filepath = os.path.join(output_dir, txt_filename)
    
    has_valid_label = False
    
    with open(txt_filepath, 'w', encoding='utf-8') as f_out:
        if 'annotations' in task and len(task['annotations']) > 0:
            results = task['annotations'][0]['result']
            for res in results:
                val = res.get('value', {})
                
                # 情况 A: 目标检测 (矩形框)
                if res['type'] == 'rectanglelabels':
                    x_center = (val['x'] + val['width'] / 2) / 100.0
                    y_center = (val['y'] + val['height'] / 2) / 100.0
                    width = val['width'] / 100.0
                    height = val['height'] / 100.0
                    
                    class_name = val['rectanglelabels'][0]
                    class_id = class_mapping.get(class_name, 0)
                    if class_name not in class_mapping:
                        print(f"⚠️ 发现未配置的类别: '{class_name}'，已默认设为 0")
                    
                    f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    has_valid_label = True
                    
                # 情况 B: 实例分割 (多边形)
                elif res['type'] == 'polygonlabels':
                    points = val['points']
                    class_name = val['polygonlabels'][0]
                    class_id = class_mapping.get(class_name, 0)
                    if class_name not in class_mapping:
                        print(f"⚠️ 发现未配置的类别: '{class_name}'，已默认设为 0")
                    
                    points_str = " ".join([f"{p[0]/100.0:.6f} {p[1]/100.0:.6f}" for p in points])
                    f_out.write(f"{class_id} {points_str}\n")
                    has_valid_label = True

    if not has_valid_label:
        print(f"⚠️ 警告: 文件 {original_filename} 没有找到可解析的矩形框或多边形，生成了空文件。")

print(f"\n✅ 转换完成！YOLO 标签已保存至: {output_dir}")