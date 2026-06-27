#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/27 17:27
@Author  : weiyutao
@File    : find_index.py
"""

import spectral.io.envi as envi
import numpy as np

def find_target_wavelength_index(hdr_path, target_nm=1400.0):
    # 🚨 核心修复：手动拼接出 .spe 文件的路径
    spe_path = hdr_path.rsplit('.', 1)[0] + '.spe'
    
    # 1. 打开 hdr 文件，显式传入 spe_path
    try:
        img = envi.open(hdr_path, spe_path)
    except Exception as e:
        print(f"❌ 打开文件失败，请检查文件是否被占用: {e}")
        return None
    
    # 2. 提取波长列表并转为 numpy 数组 (float类型)
    try:
        wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
    except KeyError:
        print("❌ 错误：HDR 文件中没有找到 'wavelength' 字段！")
        return None
    
    # 3. 计算绝对值差，找到距离 target_nm 最近的那个索引
    differences = np.abs(wavelengths - target_nm)
    closest_idx = int(np.argmin(differences))
    actual_nm = wavelengths[closest_idx]
    
    print(f"🎯 目标波长: {target_nm} nm")
    print(f"✅ 找到最接近的物理波长: {actual_nm:.2f} nm")
    print(f"👉 对应的 Python 数组索引 (Index) 是: [{closest_idx}]")
    
    return closest_idx

if __name__ == "__main__":
    # 测试运行
    hdr_file = r"C:\Users\Administrator\Documents\SortingExpert\Files\甘总b料-20260326.hdr"
    idx = find_target_wavelength_index(hdr_file, 1400.0)