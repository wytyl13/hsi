#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/25 18:14
@Author  : weiyutao
@File    : hsi_inference.py
"""

import os
import math
import numpy as np
import cv2
import spectral.io.envi as envi
from ultralytics import YOLO
import torch
import torch.nn as nn
import joblib

# ==========================================
# 1. 深度学习模块 (必须保留结构以便 PyTorch 加载权重)
# ==========================================
class SpectralAutoencoder(nn.Module):
    def __init__(self, input_dim=204, latent_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        latent = self.encoder(x)
        return latent, self.decoder(latent)

# ==========================================
# 2. 核心推理引擎
# ==========================================
class HSIInferenceEngine:
    def __init__(self, yolo_weights, ae_weights, gmm_weights, rgb_bands=(150, 100, 50), expected_bands=204):
        print("🚀 正在组装工业级 AI 推理引擎...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expected_bands = expected_bands
        self.rgb_bands = list(rgb_bands)
        
        # 1. 加载 YOLO (空间定位)
        self.yolo = YOLO(yolo_weights)
        
        # 2. 加载 AE (特征提纯)
        self.ae_model = SpectralAutoencoder(input_dim=expected_bands, latent_dim=5).to(self.device)
        self.ae_model.load_state_dict(torch.load(ae_weights, map_location=self.device, weights_only=True))
        self.ae_model.eval()
        
        # 3. 加载 GMM (概率分类)
        self.gmm = joblib.load(gmm_weights)
        
        # 定义画图颜色: Class 0 用红色, Class 1 用绿色
        self.colors = {0: (0, 0, 255), 1: (0, 255, 0)} 
        print("✅ 引擎组装完毕！显卡已就绪。")

    def _stretch_to_8bit(self, img_array):
        """复用强力防黑图拉伸算法"""
        img_array = img_array.astype(np.float32)
        res = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            channel = img_array[:, :, i]
            p1, p99 = np.percentile(channel, (1, 99))
            if p99 - p1 > 1e-6:
                channel = np.clip((channel - p1) / (p99 - p1), 0, 1)
            else:
                c_min, c_max = np.min(channel), np.max(channel)
                if c_max > c_min:
                    channel = (channel - c_min) / (c_max - c_min)
                else:
                    channel = np.zeros_like(channel)
            res[:, :, i] = channel
        return (res * 255).astype(np.uint8)

    def infer_and_draw(self, hdr_path, output_dir, crop_height=480, crop_width=640):
        spe_path = hdr_path.rsplit('.', 1)[0] + '.spe'
        base_name = os.path.basename(hdr_path).rsplit('.', 1)[0]
        
        if not os.path.exists(spe_path):
            print(f"找不到对应的 spe 文件: {spe_path}")
            return

        print(f"\n🔍 正在推理图像: {base_name}")
        img = envi.open(hdr_path, spe_path)
        data_cube = np.asarray(img.load())
        h, w, c = data_cube.shape

        if c != self.expected_bands:
            print(f"⚠️ 脏数据跳过: 波段数为 {c}，引擎要求 {self.expected_bands}")
            return

        # 制作全局底图画板 (Global Canvas)
        rgb_img = data_cube[:, :, self.rgb_bands]
        rgb_img_8bit = self._stretch_to_8bit(rgb_img)
        global_canvas = cv2.cvtColor(rgb_img_8bit, cv2.COLOR_RGB2BGR)

        # 补齐逻辑
        pad_h = max(0, crop_height - h)
        if pad_h > 0:
            global_canvas = cv2.copyMakeBorder(global_canvas, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            data_cube = np.pad(data_cube, ((0, pad_h), (0, 0), (0, 0)), mode='constant')
            h = crop_height

        num_crops = math.ceil(h / crop_height)
        start_y_list = [0] if num_crops == 1 else [int(round(i * (h - crop_height) / (num_crops - 1))) for i in range(num_crops)]

        particle_count = 0

        for start_y in start_y_list:
            end_y = start_y + crop_height
            crop_bgr = global_canvas[start_y:end_y, :crop_width].copy()

            # 1. YOLO 空间推断
            results = self.yolo(crop_bgr, conf=0.3, verbose=False)[0]
            if results.masks is None:
                continue

            # 2. 颗粒级处理 (极其关键：不能合并Mask了，必须逐个颗粒分析)
            for mask_data in results.masks.data:
                # 恢复 Mask 尺寸
                mask_np = mask_data.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (crop_width, crop_height)) > 0.5
                
                # 获取局部和全局坐标
                local_y, local_x = np.where(mask_resized)
                if len(local_y) < 50: # 过滤掉太小的噪点斑块
                    continue
                    
                global_y = local_y + start_y
                global_x = local_x

                # 3. 光谱提取与标准化
                particle_spectra = data_cube[global_y, global_x, :]
                X_norm = particle_spectra / (np.max(particle_spectra, axis=1, keepdims=True) + 1e-8)
                X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

                # 4. AE 降维与 GMM 概率聚合
                with torch.no_grad():
                    latent_features, _ = self.ae_model(X_tensor)
                latent_np = latent_features.cpu().numpy()
                
                # 获取该颗粒内所有像素的分类概率
                probs = self.gmm.predict_proba(latent_np)
                
                # 宏观决策：加权平均 (Soft Voting)
                mean_prob = np.mean(probs, axis=0)
                final_class = int(np.argmax(mean_prob))
                confidence = mean_prob[final_class] * 100

                # 5. 在全局画板上绘制轮廓和文字
                # 使用 OpenCV 提取该颗粒的边缘
                contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    # 将轮廓坐标映射回全局
                    cnt[:, 0, 1] += start_y 
                    
                    # 绘制多边形轮廓
                    color = self.colors.get(final_class, (255, 255, 255))
                    cv2.drawContours(global_canvas, [cnt], -1, color, thickness=2)
                    
                    # 在重心处写上类别和置信度
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        label_text = f"Type {final_class} ({confidence:.1f}%)"
                        # 黑色描边以防看不清
                        cv2.putText(global_canvas, label_text, (cX - 40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                        cv2.putText(global_canvas, label_text, (cX - 40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                particle_count += 1

        # 裁剪掉 padding 部分并保存
        final_result = global_canvas[:img.shape[0], :]
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{base_name}_inference.png")
        cv2.imencode('.png', final_result)[1].tofile(save_path)
        print(f"🎉 推理完成！共识别 {particle_count} 个矿石。可视化结果已保存至: {save_path}")

# ==========================================
# 3. 执行测试
# ==========================================
if __name__ == "__main__":
    # 配置模型路径
    YOLO_MODEL = r"F:\weiyutao\work\ai\hsi\runs\segment\runs\hsi_seg\train_v1\weights\best.pt"
    AE_MODEL = r"industrial_1d_ae.pth"
    GMM_MODEL = r"industrial_gmm.pkl"
    
    # 选一张你车间里扫的数据来测试 (可以是训练集里的，也可以是全新的)
    TEST_HDR = r"C:\Users\Administrator\Documents\SortingExpert\Files\甘总b料-20260326.hdr"
    OUTPUT_DIR = r"F:\weiyutao\work\data\hsi\inference_results"

    # 启动推理
    engine = HSIInferenceEngine(YOLO_MODEL, AE_MODEL, GMM_MODEL)
    engine.infer_and_draw(TEST_HDR, OUTPUT_DIR)