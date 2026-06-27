#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/25 18:00
@Author  : weiyutao
@File    : hsi_train.py
"""

import os
import glob
import math
import numpy as np
import cv2
import spectral.io.envi as envi
from ultralytics import YOLO
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import joblib
from tqdm import tqdm
import random

# ==========================================
# 1. 深度学习模块：一维自编码器
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
# 2. 核心层：单图像素提取器 (完美复用之前的滑窗逻辑)
# ==========================================
class HSIPixelExtractor:
    """
    负责单张高光谱的 滑窗 -> YOLO推理 -> 坐标映射 -> 光谱提取
    """
    def __init__(self, yolo_model, rgb_bands=(150, 100, 50), crop_height=480, crop_width=640):
        self.yolo = yolo_model
        self.rgb_bands = list(rgb_bands)
        self.crop_height = crop_height
        self.crop_width = crop_width

    def _stretch_to_8bit(self, img_array):
        """复用之前的强力防黑图拉伸算法"""
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
        """核心处理：带严格脏数据拦截的工业级提取"""
        try:
            img = envi.open(hdr_path, spe_path)
            data_cube = np.asarray(img.load())
            h, w, c = data_cube.shape

            # ==========================================
            # 🛡️ 工业级防爆盾：过滤非标数据与损坏数据
            # ==========================================
            if c != expected_bands:
                # 使用 tqdm.write 防止打乱进度条，动态提示被拦截的异形文件
                tqdm.write(f"⚠️ [脏数据拦截] 跳过 {os.path.basename(hdr_path)}: 发现 {c} 个波段 (系统要求 {expected_bands} 波段)。")
                return np.empty((0, expected_bands))
                
            if max(self.rgb_bands) >= c:
                tqdm.write(f"⚠️ [脏数据拦截] 跳过 {os.path.basename(hdr_path)}: 波段不足以合成设定的 RGB 伪彩图。")
                return np.empty((0, expected_bands))
            # ==========================================

            # 伪彩映射
            rgb_img = data_cube[:, :, self.rgb_bands]
            rgb_img_8bit = self._stretch_to_8bit(rgb_img)
            bgr_img = cv2.cvtColor(rgb_img_8bit, cv2.COLOR_RGB2BGR)

            # --- 完美对齐与 Padding 逻辑 ---
            if h < self.crop_height:
                pad_h = self.crop_height - h
                bgr_img = cv2.copyMakeBorder(bgr_img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                data_cube = np.pad(data_cube, ((0, pad_h), (0, 0), (0, 0)), mode='constant')
                h = self.crop_height

            if w > self.crop_width:
                bgr_img = bgr_img[:, :self.crop_width]
            elif w < self.crop_width:
                pad_w = self.crop_width - w
                bgr_img = cv2.copyMakeBorder(bgr_img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

            # --- 均匀滑窗逻辑 ---
            num_crops = math.ceil(h / self.crop_height)
            start_y_list = [0] if num_crops == 1 else [int(round(i * (h - self.crop_height) / (num_crops - 1))) for i in range(num_crops)]

            all_pure_pixels = []

            for start_y in start_y_list:
                end_y = start_y + self.crop_height
                crop = bgr_img[start_y:end_y, :]

                if crop.shape[0] != self.crop_height or crop.shape[1] != self.crop_width:
                    crop = cv2.resize(crop, (self.crop_width, self.crop_height))

                # YOLO 推理
                results = self.yolo(crop, conf=0.6, verbose=False)
                result = results[0]

                if result.masks is None:
                    continue

                # 合并 Mask
                slice_mask_combined = np.zeros((self.crop_height, self.crop_width), dtype=bool)
                for mask in result.masks.data:
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (self.crop_width, self.crop_height)) > 0.5
                    slice_mask_combined = slice_mask_combined | mask_resized

                # 坐标映射回全局
                local_y, local_x = np.where(slice_mask_combined)
                global_y = local_y + start_y
                global_x = local_x

                # 提取 204 波段
                pure_spectra = data_cube[global_y, global_x, :]
                all_pure_pixels.append(pure_spectra)

            if len(all_pure_pixels) == 0:
                return np.empty((0, expected_bands))

            return np.vstack(all_pure_pixels)

        except Exception as e:
            # 文件损坏等其他异常，安全返回空数组
            tqdm.write(f"❌ [读取异常] {os.path.basename(hdr_path)}: {e}")
            return np.empty((0, expected_bands))


# ==========================================
# 3. 调度层：全目录大一统训练管线
# ==========================================
class HSIUnsupervisedPipeline:
    """
    负责扫目录、进度展示、特征聚合，以及最终的模型训练。
    """
    def __init__(self, yolo_weights, rgb_bands=(150, 100, 50), crop_height=480, crop_width=640, file_sample_ratio=1.0):
        print("正在将 YOLOv8 加载到 GPU 显存...")
        self.yolo_model = YOLO(yolo_weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 挂载底层提取器
        self.extractor = HSIPixelExtractor(
            yolo_model=self.yolo_model,
            rgb_bands=rgb_bands,
            crop_height=crop_height,
            crop_width=crop_width
        )
        
        # 新增：文件采样比例 (0.0 到 1.0 之间)
        self.file_sample_ratio = file_sample_ratio

    def _get_spe_path(self, hdr_path):
        spe_path = hdr_path.rsplit('.', 1)[0] + '.spe'
        return spe_path if os.path.exists(spe_path) else None

    def run_training(self, input_path):
        # --- 阶段 1：遍历文件，聚拢数据 ---
        hdr_files = []
        if os.path.isfile(input_path) and input_path.endswith('.hdr'):
            hdr_files = [input_path]
        elif os.path.isdir(input_path):
            all_hdr_files = glob.glob(os.path.join(input_path, '*.hdr'))
            
            # ==========================================
            # 🎲 工业级重构：随机采样抽帧逻辑
            # ==========================================
            if 0 < self.file_sample_ratio < 1.0:
                # 计算需要抽取的数量，保底至少抽 1 个
                sample_size = max(1, int(len(all_hdr_files) * self.file_sample_ratio))
                # 随机打乱并抽取
                hdr_files = random.sample(all_hdr_files, sample_size)
                print(f"\n[阶段 1/3] 扫描文件 -> 开启 {self.file_sample_ratio*100:.0f}% 随机采样！")
                print(f" -> 原始总数: {len(all_hdr_files)} 个 | 实际抽取: {sample_size} 个。开始提取...")
            else:
                hdr_files = all_hdr_files
                print(f"\n[阶段 1/3] 扫描文件 -> 不采样，全量跑！共 {len(hdr_files)} 个文件。开始提取...")
            # ==========================================
        else:
            raise ValueError("提供的路径无效！")
        
        global_pixels = []
        pbar = tqdm(hdr_files, desc="提取进度", unit="文件", ncols=100)
        
        for hdr_path in pbar:
            spe_path = self._get_spe_path(hdr_path)
            if spe_path:
                base_name = os.path.basename(hdr_path).rsplit('.', 1)[0]
                pbar.set_postfix_str(f"处理: {base_name[:12]}...")
                
                pixels = self.extractor.process_and_extract(hdr_path, spe_path)
                if pixels.shape[0] > 0:
                    global_pixels.append(pixels)
            else:
                tqdm.write(f"[跳过] 无对应的 spe 文件: {hdr_path}")

        if not global_pixels:
            print("灾难：没有提取到任何有效矿石像素！请检查 YOLO 模型或文件。")
            return

        # 拼成终极大矩阵
        X_raw = np.vstack(global_pixels)
        print(f"\n✅ 提取完成！本次采样共抠出 {X_raw.shape[0]} 个纯净矿石像素！")

        # --- 阶段 2：自编码器训练 ---
        print("\n[阶段 2/3] 启动 Autoencoder 非线性降维训练...")
        X_norm = X_raw / (np.max(X_raw, axis=1, keepdims=True) + 1e-8)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        ae_model = SpectralAutoencoder(input_dim=X_raw.shape[1], latent_dim=5).to(self.device)
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        ae_model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            latent, reconstructed = ae_model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"  -> Epoch [{epoch+1}/50], 重建损失 MSE: {loss.item():.6f}")

        print("\n[阶段 3/3] 提取潜变量并进行 GMM 聚类...")
        ae_model.eval()
        with torch.no_grad():
            latent_features, _ = ae_model(X_tensor)
        latent_np = latent_features.cpu().numpy()

        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(latent_np)

        torch.save(ae_model.state_dict(), "industrial_ae_model.pth")
        joblib.dump(gmm, "industrial_gmm_model.pkl")
        print("\n🎉 无监督训练圆满成功！模型已保存！")


# ==========================================
# 4. 一键启动
# ==========================================
if __name__ == "__main__":
    
    # 你的 YOLO 权重路径
    YOLO_WEIGHTS = r"F:\weiyutao\work\ai\hsi\runs\segment\runs\hsi_seg\train_v1\weights\best.pt"
    
    # 【改回整个文件夹路径！】
    INPUT_DIR = r"C:\Users\Administrator\Documents\SortingExpert\Files" 

    ## 实例化大一统管线，开启 20% 随机采样！
    pipeline = HSIUnsupervisedPipeline(
        yolo_weights=YOLO_WEIGHTS,
        crop_height=480, 
        crop_width=640,
        file_sample_ratio=0.30  # <-- 改变在这里！目前设置为 20%
    )
    
    pipeline.run_training(INPUT_DIR)