#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/24 14:45
@Author  : weiyutao
@File    : hsi_data_build.py
"""

import os
import glob
import math
import numpy as np
import cv2
import spectral.io.envi as envi


from tqdm import tqdm  # <-- 别忘了在文件开头导入 tqdm

# ==========================================
# 核心层：单张高光谱图像处理器
# ==========================================
class HSISingleProcessor:
    def __init__(self, rgb_bands=(76, 51, 25), crop_height=480, crop_width=640, save_full=True, verbose=True):
        self.rgb_bands = list(rgb_bands)
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.save_full = save_full
        self.verbose = verbose

    def _stretch_to_8bit(self, img_array, ignore_margin=150):
        """同步：支持指定边缘忽略区的智能拉伸，暴力压黑背景"""
        img_array = img_array.astype(np.float32)
        res = np.zeros_like(img_array)
        w = img_array.shape[1]
        
        for i in range(img_array.shape[2]):
            channel = img_array[:, :, i]
            
            # 避开边缘噪声区提取干净的 ROI 用于计算统计量
            if w > ignore_margin * 2:
                clean_roi = channel[:, ignore_margin : w - ignore_margin]
            else:
                clean_roi = channel
                
            # 使用 40~99.9 分位数，强压背景，提亮矿石
            p_low, p_high = np.percentile(clean_roi, (40, 99.9))
            
            if p_high - p_low > 1e-6:
                channel = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
            else:
                c_min, c_max = np.min(clean_roi), np.max(clean_roi)
                if c_max > c_min:
                    channel = (channel - c_min) / (c_max - c_min)
                else:
                    channel = np.zeros_like(channel)
                    
            res[:, :, i] = channel
            
        return (res * 255).astype(np.uint8)

    def process(self, hdr_path, spe_path, output_dir, base_name=None):
        if base_name is None:
            base_name = os.path.basename(hdr_path).rsplit('.', 1)[0]

        try:
            img = envi.open(hdr_path, spe_path)
            h, w, c = img.shape
            
            if self.verbose:
                print(f"图像尺寸: 高={h}, 宽={w}, 波段={c}")

            if max(self.rgb_bands) >= c:
                print(f"\n[警告] 波段超出范围 (请求:{max(self.rgb_bands)}, 实际:{c})，跳过 {base_name}")
                return 0

            data_cube = np.asarray(img.load())
            
            # ==========================================
            # 🌟 同步：视觉替身生成与边缘涂黑逻辑
            # ==========================================
            margin = 150 
            rgb_img = data_cube[:, :, self.rgb_bands]
            
            # 注意：如果在推理端 hsi_edge_client.py 做了 np.flip(rgb_img, axis=1)
            # 为了保证训练集和推理时的数据特征完全一致，这里也建议做翻转。
            # 如果不需要翻转，请将下面这行注释掉，直接使用 rgb_img_8bit = self._stretch_to_8bit(rgb_img, ignore_margin=margin)
            rgb_flipped = np.flip(rgb_img, axis=1) 
            
            # 使用带 margin 的高级拉伸
            rgb_img_8bit = self._stretch_to_8bit(rgb_flipped, ignore_margin=margin)
            bgr_img = cv2.cvtColor(rgb_img_8bit, cv2.COLOR_RGB2BGR)

            # 替身涂黑：防止 YOLO 学到边缘的硬件条纹噪声
            if w > margin * 2:
                bgr_img[:, :margin] = 0        
                bgr_img[:, w-margin:] = 0      

            # --- 以下保存和切片逻辑保持不变 ---
            if self.save_full:
                os.makedirs(output_dir, exist_ok=True)
                full_save_path = os.path.join(output_dir, f"{base_name}_full.png")
                is_success, buffer = cv2.imencode('.png', bgr_img)
                if is_success:
                    buffer.tofile(full_save_path)
                    if self.verbose:
                        print(f"保存完整图像至: {full_save_path}")

            saved_count = 0
            if h < self.crop_height:
                pad_h = self.crop_height - h
                bgr_img = cv2.copyMakeBorder(bgr_img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                h = self.crop_height

            if w > self.crop_width:
                bgr_img = bgr_img[:, :self.crop_width]
            elif w < self.crop_width:
                pad_w = self.crop_width - w
                bgr_img = cv2.copyMakeBorder(bgr_img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

            num_crops = math.ceil(h / self.crop_height)
            start_y_list = [0] if num_crops == 1 else [int(round(i * (h - self.crop_height) / (num_crops - 1))) for i in range(num_crops)]

            for start_y in start_y_list:
                end_y = start_y + self.crop_height
                crop = bgr_img[start_y:end_y, :]

                if crop.shape[0] != self.crop_height or crop.shape[1] != self.crop_width:
                    crop = cv2.resize(crop, (self.crop_width, self.crop_height))

                save_name = f"{base_name}_crop_{saved_count:03d}.png"
                save_path = os.path.join(output_dir, save_name)
                
                is_crop_success, crop_buffer = cv2.imencode('.png', crop)
                if is_crop_success:
                    crop_buffer.tofile(save_path)
                    if self.verbose: 
                        print(f"  -> 已保存切片: {save_name}")
                    saved_count += 1

            return saved_count

        except Exception as e:
            print(f"\n[处理异常] {base_name}: {e}")
            return 0


# ==========================================
# 调度层：数据集构建管线 
# ==========================================
class HSIDatasetPipeline:
    def __init__(self, output_dir, rgb_bands=(76, 51, 25), crop_height=480, crop_width=640, save_full=True):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.processor = HSISingleProcessor(
            rgb_bands=rgb_bands, 
            crop_height=crop_height, 
            crop_width=crop_width,
            save_full=save_full
        )

    def _get_spe_path(self, hdr_path):
        spe_path = hdr_path.rsplit('.', 1)[0] + '.spe'
        return spe_path if os.path.exists(spe_path) else None

    def run(self, input_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入路径不存在: {input_path}")

        total_saved = 0

        # 情景 1: 处理单图 (保持日志全开，方便调试)
        if os.path.isfile(input_path):
            if not input_path.endswith('.hdr'):
                raise ValueError("请提供 .hdr 文件的路径！")
                
            spe_path = self._get_spe_path(input_path)
            if not spe_path:
                print(f"[跳过] 找不到匹配的 .spe 文件: {input_path}")
                return
                
            print(f"模式：单图处理 -> {input_path}")
            self.processor.verbose = True  # 单图模式开启啰嗦日志
            total_saved += self.processor.process(input_path, spe_path, self.output_dir)

        # 情景 2: 处理目录 (关闭啰嗦日志，开启 tqdm 进度条)
        elif os.path.isdir(input_path):
            hdr_files = glob.glob(os.path.join(input_path, '*.hdr'))
            print(f"模式：批量处理目录 -> 共找到 {len(hdr_files)} 个高光谱文件\n")
            
            # 关闭单图处理器的日志，防止冲刷进度条
            self.processor.verbose = False 
            
            # 使用 tqdm 包装文件列表
            pbar = tqdm(hdr_files, desc="总进度", unit="文件", ncols=100)
            
            for hdr_path in pbar:
                spe_path = self._get_spe_path(hdr_path)
                if spe_path:
                    base_name = os.path.basename(hdr_path).rsplit('.', 1)[0]
                    # 动态更新进度条右侧的提示文字
                    pbar.set_postfix_str(f"正在切片: {base_name[:15]}...") 
                    
                    count = self.processor.process(hdr_path, spe_path, self.output_dir, base_name)
                    total_saved += count
                else:
                    # 如果有孤立的 hdr，使用 tqdm.write 防止打乱进度条
                    tqdm.write(f"[警告] 孤立的 HDR 文件跳过: {hdr_path}")

        print("\n" + "=" * 50)
        print(f"执行完毕！共生成 {total_saved} 张切片图。")
        print(f"保存路径: {self.output_dir}")


# ==========================================
# 使用示范
# ==========================================
if __name__ == "__main__":
    
    # OUTPUT_DIR = r"F:\weiyutao\work\data\hsi\yolo_dataset\images"
    OUTPUT_DIR = r"F:\weiyutao\work\data\hsi\yolo_dataset\平陆一诺铝土矿500num20260522"
    
    # 实例化管线，可以根据需要开关 save_full 参数
    pipeline = HSIDatasetPipeline(
        output_dir=OUTPUT_DIR, 
        crop_height=480,  
        crop_width=640,
        save_full=False    # <-- 这里设为 False，就不会再保存全尺寸的长图了
    )
    
    # 传入单个字符串文件测试
    # SINGLE_FILE_PATH = r"C:\Users\Administrator\Documents\SortingExpert\Files\一诺-曹川-20260423.hdr"
    # INPUT_DIR = r"C:\Users\Administrator\Documents\SortingExpert\Files"
    INPUT_DIR = r"F:\500块采样"
    pipeline.run(INPUT_DIR)