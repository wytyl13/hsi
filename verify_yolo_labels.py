#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import random
import glob
from tqdm import tqdm

def verify_yolo_dataset(dataset_dir, save_dir, num_samples=30):
    """
    针对 YOLOv8 分割数据集的全集抽检器
    自动从 train, val, test 目录中均匀抽取样本进行高保真可视化
    """
    os.makedirs(save_dir, exist_ok=True)
    
    subsets = ['train', 'val', 'test']
    all_samples = []
    
    # 计算每个子目录需要抽取的数量，确保均匀覆盖
    samples_per_subset = max(1, num_samples // len(subsets))
    
    # 1. 搜集并打乱各目录的样本
    for subset in subsets:
        img_dir = os.path.join(dataset_dir, 'images', subset)
        lbl_dir = os.path.join(dataset_dir, 'labels', subset)
        
        if not os.path.exists(img_dir):
            continue
            
        # 兼容 png 和 jpg
        img_paths = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))
        
        if not img_paths:
            print(f"⚠️ 警告: {subset} 目录为空！")
            continue
            
        # 随机抽取当前子集的样本
        subset_samples = random.sample(img_paths, min(samples_per_subset, len(img_paths)))
        
        # 记录图像路径、对应的标签目录以及归属的子集名称
        for p in subset_samples:
            all_samples.append((p, lbl_dir, subset))
            
    if not all_samples:
        print(f"❌ 找不到任何图像！请检查数据集路径: {dataset_dir}")
        return

    print(f"🧐 正在从 train/val/test 中总共抽取 {len(all_samples)} 张图像进行高精度可视化核对...")

    # 2. 渲染可视化
    for img_p, lbl_dir, subset in tqdm(all_samples, desc="🎨 渲染验证图"):
        img = cv2.imread(img_p)
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # 获取对应 txt 标签路径
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        txt_p = os.path.join(lbl_dir, base_name + '.txt')
        
        if not os.path.exists(txt_p):
            continue
            
        with open(txt_p, 'r') as f:
            lines = f.readlines()
            
        overlay = img.copy()
        labels_to_draw = [] # 新增：专门用来暂存要画的字母和坐标
        
        for line in lines:
            data = line.strip().split()
            # 格式排错检查
            if len(data) < 7: 
                if len(data) == 5:
                    print(f"\n⚠️ 警告: {base_name}.txt 是矩形框格式(5个值)，不是多边形！无法绘制。")
                continue
                
            class_id = data[0]
            points = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
            abs_pts = (points * [w, h]).astype(np.int32)
            
            # 确定颜色和显示的字母 (0为铝l，1为硅g)
            color = (0, 255, 0) if class_id == '0' else (0, 0, 255)
            label_text = 'j' if class_id == '0' else 'w'
            
            # 1. 在半透明层画填充色
            cv2.fillPoly(overlay, [abs_pts], color)
            
            # 2. 直接在原图上画出 1 像素的纯色实线边界
            cv2.polylines(img, [abs_pts], isClosed=True, color=color, thickness=1)

            # 3. 计算多边形的几何中心点
            M = cv2.moments(abs_pts)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # 极端情况下如果计算失败，就取第一个点的坐标
                cX, cY = abs_pts[0][0], abs_pts[0][1]
                
            # 把字母和坐标存起来，等最后再画
            labels_to_draw.append((label_text, cX, cY))

        # 4. 融合半透明层 (将掩码的透明度设为 0.5)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # 5. 【新增】：在最顶层画上小字，避免被半透明层遮挡变暗
        for text, cx, cy in labels_to_draw:
            # 字号设为 0.35 比较小巧，微调 (-4, +4) 让字母视觉上更居中
            # 先画粗一点的黑色作为阴影/描边 (thickness=2)
            cv2.putText(img, text, (cx - 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            # 再画细一点的纯白色作为文字本体 (thickness=1)
            cv2.putText(img, text, (cx - 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # 保存结果
        save_path = os.path.join(save_dir, f"{subset}_{os.path.basename(img_p)}")
        cv2.imwrite(save_path, img)
    

if __name__ == '__main__':
    # ================= 配置区 =================
    # 指向刚才生成的 YOLO 格式数据集总目录
    DATASET_DIR = "/home/weiyutao/industrial_vision_data/gold_ore_images_3_train_20260402"
    
    # 抽检可视化结果的存放目录
    DEBUG_DIR = "/home/weiyutao/industrial_vision_data/gold_ore_images_3_train_20260402_yolo_debug"

    # 执行抽检，设置想要抽看的总图片数量 (默认30张，平分给train/val/test)
    verify_yolo_dataset(DATASET_DIR, DEBUG_DIR, num_samples=30)
    
    print(f"\n✅ 可视化完成！请前往 {DEBUG_DIR} 下载查看图片是否标注准确。")