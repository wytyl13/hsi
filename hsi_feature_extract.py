#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/27 16:21
@Author  : weiyutao
@File    : hsi_feature_extract.py
"""

import os
import math
import numpy as np
import cv2
import glob
import random
from tqdm import tqdm
import spectral.io.envi as envi
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.mixture import GaussianMixture
from ultralytics import YOLO
from model import Conv1DSpectralAutoencoder, WeightedMSALoss


class HSIPixelExtractor:
    def __init__(self, yolo_model, rgb_bands=(150, 100, 50), crop_height=480, crop_width=640):
        self.yolo = yolo_model
        self.rgb_bands = list(rgb_bands)
        self.crop_height = crop_height
        self.crop_width = crop_width

    def _stretch_to_8bit(self, img_array):
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

    def process_and_extract(self, hdr_path, spe_path, expected_bands=204):
        try:
            img = envi.open(hdr_path, spe_path)
            data_cube = np.asarray(img.load())
            h, w, c = data_cube.shape

            if c != expected_bands:
                tqdm.write(f"⚠️ [脏数据拦截] 跳过: 发现 {c} 个波段。")
                return np.empty((0, expected_bands - 1)) # 返回 203 维

            # 提取 RGB 用于 YOLO (必须在一阶导数之前提取)
            rgb_img = data_cube[:, :, self.rgb_bands]
            rgb_img_8bit = self._stretch_to_8bit(rgb_img)
            bgr_img = cv2.cvtColor(rgb_img_8bit, cv2.COLOR_RGB2BGR)

            # 🚨 极其关键：全图计算一阶导数，维度变为 203
            deriv_cube = np.diff(data_cube, axis=2).astype(np.float32)

            # 对齐与 Padding (同时处理 bgr_img 和 deriv_cube)
            if h < self.crop_height:
                pad_h = self.crop_height - h
                bgr_img = cv2.copyMakeBorder(bgr_img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                deriv_cube = np.pad(deriv_cube, ((0, pad_h), (0, 0), (0, 0)), mode='constant')
                h = self.crop_height
            if w > self.crop_width:
                bgr_img = bgr_img[:, :self.crop_width]
                deriv_cube = deriv_cube[:, :self.crop_width, :] # 截断宽度
            elif w < self.crop_width:
                pad_w = self.crop_width - w
                bgr_img = cv2.copyMakeBorder(bgr_img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                deriv_cube = np.pad(deriv_cube, ((0, 0), (0, pad_w), (0, 0)), mode='constant')

            # 滑窗逻辑
            num_crops = math.ceil(h / self.crop_height)
            start_y_list = [0] if num_crops == 1 else [int(round(i * (h - self.crop_height) / (num_crops - 1))) for i in range(num_crops)]
            all_pure_pixels = []

            for start_y in start_y_list:
                end_y = start_y + self.crop_height
                crop = bgr_img[start_y:end_y, :]
                if crop.shape[0] != self.crop_height or crop.shape[1] != self.crop_width:
                    crop = cv2.resize(crop, (self.crop_width, self.crop_height))

                # YOLO 推理
                results = self.yolo(crop, conf=0.6, verbose=False, retina_masks=True)
                result = results[0]
                if result.masks is None:
                    continue

                # 合并 Mask
                slice_mask_combined = np.zeros((self.crop_height, self.crop_width), dtype=bool)
                for mask in result.masks.data:
                    slice_mask_combined |= (cv2.resize(mask.cpu().numpy(), (self.crop_width, self.crop_height)) > 0.5)

                # 提取一阶导数 203 波段
                local_y, local_x = np.where(slice_mask_combined)
                pure_spectra = deriv_cube[local_y + start_y, local_x, :]
                all_pure_pixels.append(pure_spectra)

            if len(all_pure_pixels) == 0:
                return np.empty((0, expected_bands - 1))
            return np.vstack(all_pure_pixels)

        except Exception as e:
            tqdm.write(f"❌ [读取异常] {os.path.basename(hdr_path)}: {e}")
            return np.empty((0, expected_bands - 1))
        

class HSIUnsupervisedPipeline:
    def __init__(self, yolo_weights, rgb_bands=(150, 100, 50), crop_height=480, crop_width=640, max_samples=50):
        print("正在将 YOLOv8 加载到 GPU 显存...")
        self.yolo_model = YOLO(yolo_weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_samples = max_samples
        self.extractor = HSIPixelExtractor(self.yolo_model, rgb_bands, crop_height, crop_width)

    def run_training(self, input_path):
        # --- 阶段 1：扫描文件，按绝对数量采样 ---
        all_hdr_files = glob.glob(os.path.join(input_path, '*.hdr')) if os.path.isdir(input_path) else [input_path]
        
        sample_size = min(self.max_samples, len(all_hdr_files))
        hdr_files = random.sample(all_hdr_files, sample_size)
        print(f"\n[阶段 1/3] 开始提取一阶导数... (抽取 {sample_size}/{len(all_hdr_files)} 个文件)")
        
        global_pixels = []
        pbar = tqdm(hdr_files, desc="提取进度", unit="文件", ncols=100)
        for hdr_path in pbar:
            spe_path = hdr_path.rsplit('.', 1)[0] + '.spe'
            if os.path.exists(spe_path):
                pixels = self.extractor.process_and_extract(hdr_path, spe_path)
                if pixels.shape[0] > 0:
                    global_pixels.append(pixels)

        if not global_pixels:
            print("灾难：没有提取到任何有效矿石像素！")
            return

        X_raw = np.vstack(global_pixels)
        print(f"\n✅ 提取完成！本次抠出 {X_raw.shape[0]} 个纯净的 203维 像素点。")

        # --- 阶段 2：工业级 1D-CNN 自编码器训练 (DataLoader 版) ---
        print("\n[阶段 2/3] 启动 1D-CNN Autoencoder 强化训练...")
        
        # 🚨 数据集构建，解决爆显存问题
        X_tensor = torch.tensor(X_raw, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

        ae_model = Conv1DSpectralAutoencoder(input_dim=X_raw.shape[1], latent_dim=5).to(self.device)
        optimizer = torch.optim.AdamW(ae_model.parameters(), lr=0.001)
        
        # 🚨 注入 1400nm 先验 (假设索引为 100-110，你需要根据实际波长表修改)
        target_indices = list(range(121, 128))
        criterion = WeightedMSALoss(target_indices=target_indices, weight=50.0).to(self.device)
        # criterion = nn.MSELoss().to(self.device)

        ae_model.train()
        epochs = 30 # 使用 DataLoader 后，30轮足够收敛
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                latent, reconstructed = ae_model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            if (epoch+1) % 5 == 0:
                print(f"  -> Epoch [{epoch+1}/{epochs}], 加权 MSE Loss: {running_loss/len(dataloader):.6f}")

        # --- 阶段 3：GMM 聚类 (定性预演) ---
        print("\n[阶段 3/3] 提取潜变量进行无监督 GMM 聚类测试...")
        ae_model.eval()
        
        # 为了防爆显存，分批提取潜变量
        latent_list = []
        with torch.no_grad():
            for batch_x, _ in DataLoader(dataset, batch_size=2048):
                latent_features, _ = ae_model(batch_x.to(self.device))
                latent_list.append(latent_features.cpu().numpy())
        latent_np = np.vstack(latent_list)

        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(latent_np)

        torch.save(ae_model.state_dict(), "industrial_1d_ae.pth")
        joblib.dump(gmm, "industrial_gmm.pkl")
        print("\n🎉 大一统训练成功！1D-CNN 模型与 GMM 已保存！")

# ==========================================
# 4. 一键启动
# ==========================================
if __name__ == "__main__":
    YOLO_WEIGHTS = r"F:\weiyutao\work\ai\hsi\runs\segment\runs\hsi_seg\train_v1-4\weights\best.pt"
    INPUT_DIR = r"C:\Users\Administrator\Documents\SortingExpert\Files" 

    # 极大优化：使用明确数量代替百分比
    pipeline = HSIUnsupervisedPipeline(
        yolo_weights=YOLO_WEIGHTS,
        crop_height=480, 
        crop_width=640,
        max_samples=50   # <-- 抽取 20 张图像用于提取
    )
    
    pipeline.run_training(INPUT_DIR)