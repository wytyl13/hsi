#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/05/14 10:09
@Author  : weiyutao
@File    : find_beat_band.py
"""

import numpy as np
import spectral.io.envi as envi
from itertools import combinations
import time

def find_best_rgb_bands(hdr_path):
    print("🔍 正在加载图像并计算全局光谱协方差 (离线标定)...")
    start_time = time.time()
    
    img = envi.open(hdr_path, hdr_path.replace('.hdr', '.spe'))
    data = np.asarray(img.load())
    h, w, c = data.shape

    # 1. 展平数据，随机抽样加速计算 (抽 20000 个像素点足够代表全局特征了)
    pixels = data.reshape(-1, c)
    if pixels.shape[0] > 20000:
        idx = np.random.choice(pixels.shape[0], 20000, replace=False)
        pixels = pixels[idx, :]

    # 2. 计算每个波段的标准差 (Std)
    stds = np.std(pixels, axis=0)

    # 3. 计算波段间的相关系数矩阵 (Correlation Matrix)
    corr_matrix = np.corrcoef(pixels.T)

    # 4. 暴力搜索 OIF (为了极速，只在前 30 个高方差波段里找组合)
    top_30_bands = np.argsort(stds)[-30:]
    
    best_oif = 0
    best_bands = (0, 0, 0)

    for combo in combinations(top_30_bands, 3):
        i, j, k = combo
        # OIF = 三个波段的标准差之和 / 相关系数绝对值之和
        std_sum = stds[i] + stds[j] + stds[k]
        corr_sum = abs(corr_matrix[i, j]) + abs(corr_matrix[i, k]) + abs(corr_matrix[j, k])
        
        oif = std_sum / (corr_sum + 1e-6)
        if oif > best_oif:
            best_oif = oif
            best_bands = combo

    # 按波长顺序排个序，对应 RGB
    best_bands = tuple(sorted(best_bands, reverse=True))
    
    print(f"✅ 分析完成！耗时: {time.time()-start_time:.2f} 秒")
    print(f"🥇 矿石与背景分离度最高的 3 个波段是: {best_bands}")
    print("👉 请将这个组合填入在线推理脚本的 rgb_bands 中！")

if __name__ == "__main__":
    # 填入你本地的一张测试图的路径
    find_best_rgb_bands(r"F:\500块采样\0_1-5.hdr")