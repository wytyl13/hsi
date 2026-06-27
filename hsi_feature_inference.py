#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/27 16:43
@Author  : weiyutao
@File    : hsi_feature_inference.py
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
from model import Conv1DSpectralAutoencoder


class HSIInferenceEngine:
    def __init__(self, yolo_weights, ae_weights, gmm_weights, rgb_bands=(150, 100, 50), expected_bands=204):
        print("🚀 正在组装工业级 AI 推理引擎 (含高级 UI 渲染)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expected_bands = expected_bands
        self.rgb_bands = list(rgb_bands)
        
        # 1. 加载 YOLO
        self.yolo = YOLO(yolo_weights)
        
        # 2. 加载 AE 
        self.ae_model = Conv1DSpectralAutoencoder(input_dim=expected_bands - 1, latent_dim=5).to(self.device)
        self.ae_model.load_state_dict(torch.load(ae_weights, map_location=self.device, weights_only=True))
        self.ae_model.eval()
        
        # 3. 加载 GMM
        self.gmm = joblib.load(gmm_weights)
        
        # 视觉配置: Class 0 (红色), Class 1 (绿色)
        self.colors = {0: (50, 50, 255), 1: (50, 255, 50)} 
        print(f"✅ 引擎组装完毕！当前算力节点: {self.device}。")

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

        # 1. 制作全局底图画板
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
        drawn_text_rects = [] # 用于记录已绘制的文字背景框，防止重叠！

        for start_y in start_y_list:
            end_y = start_y + crop_height
            crop_bgr = global_canvas[start_y:end_y, :crop_width].copy()

            # 2. YOLO 空间推断 
            results = self.yolo(crop_bgr, conf=0.4, verbose=False, retina_masks=True)[0]
            if results.masks is None:
                continue

            # 3. 颗粒级物理提取与量化推理
            for mask_data in results.masks.data:
                mask_np = mask_data.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (crop_width, crop_height)) > 0.5
                
                local_y, local_x = np.where(mask_resized)
                if len(local_y) < 50: 
                    continue
                    
                global_y = local_y + start_y
                global_x = local_x

                # 提取用于物理量化的“原始光谱”
                particle_spectra_raw = data_cube[global_y, global_x, :]
                # 必须做归一化，消除石头亮暗和距离的影响
                X_norm_raw = particle_spectra_raw / (np.max(particle_spectra_raw, axis=1, keepdims=True) + 1e-8)

                # ==========================================
                # 🛡️ 工业级物理规则：波段深度 (Band Depth) 提取
                # ==========================================
                # 1400nm 坑底大约在 124，我们取左右肩和坑底的区间均值抗噪
                left_shoulder  = np.mean(X_norm_raw[:, 118:121], axis=1) # 坑左侧的平缓区
                right_shoulder = np.mean(X_norm_raw[:, 127:130], axis=1) # 坑右侧的平缓区
                valley_bottom  = np.mean(X_norm_raw[:, 123:126], axis=1) # 坑底

                # 计算理论深度
                band_depths = ((left_shoulder + right_shoulder) / 2.0) - valley_bottom
                # 获取该石头所有像素的平均坑深
                mean_1400_depth = np.mean(band_depths)
                
                # ==========================================
                # 💥 架构大升级：从“离散分类”到“连续量化”
                # ==========================================
                # 设定物理经验极值 (根据你之前打印的数据，这里设置了稳健的上下限)
                MIN_DEPTH = 0.01  # 极浅的坑 -> 对应极高品位纯铝
                MAX_DEPTH = 0.05  # 极深的坑 -> 对应极低品位脉石/硅
                MAX_RATIO = 10.0  # 最高预估铝硅比
                MIN_RATIO = 1.0   # 最低预估铝硅比

                # 1. 限制数值范围，防止极端噪点引发越界
                d_clipped = np.clip(mean_1400_depth, MIN_DEPTH, MAX_DEPTH)
                
                # 2. 线性逆映射：计算预估铝硅比 (坑越深，比例越低)
                est_ratio = MAX_RATIO - ((d_clipped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)) * (MAX_RATIO - MIN_RATIO)
                
                print(f"[{particle_count}号石头] 坑深: {mean_1400_depth:.4f} -> 预估铝硅比: {est_ratio:.1f}")

                # 3. 动态热力图颜色引擎 (红: 高品位铝, 绿: 低品位硅)
                # 将比值归一化到 0~1 之间用于计算颜色比例
                ratio_norm = (est_ratio - MIN_RATIO) / (MAX_RATIO - MIN_RATIO)
                
                # OpenCV 颜色格式是 (B, G, R)
                r_val = int(255 * ratio_norm)
                g_val = int(255 * (1.0 - ratio_norm))
                dynamic_color = (0, g_val, r_val) 

                # ==========================================
                # 4. 高级 UI 渲染：动态颜色与防遮挡画图逻辑
                # ==========================================
                contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt[:, 0, 1] += start_y 
                    # 绘制带物理意义动态颜色的轮廓
                    cv2.drawContours(global_canvas, [cnt], -1, dynamic_color, thickness=2)
                    
                    # 获取外接矩形定位文字
                    x, y, w, box_h = cv2.boundingRect(cnt)
                    
                    # 极简量化标签
                    label_text = f"A/S:{est_ratio:.1f} (D:{mean_1400_depth:.3f})"
                    font_scale = 0.45
                    font_thickness = 1
                    
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    text_x = x
                    text_y = y - 6
                    
                    if text_y - text_h < 0:
                        text_y = y + box_h + text_h + 6
                        
                    bg_x1 = max(0, text_x - 3)
                    bg_y1 = text_y - text_h - 4
                    bg_x2 = text_x + text_w + 3
                    bg_y2 = text_y + baseline + 2
                    
                    # 💥 防碰撞躲避逻辑
                    collision = True
                    max_attempts = 5 
                    attempts = 0
                    while collision and attempts < max_attempts:
                        collision = False
                        for dr in drawn_text_rects:
                            dr_x1, dr_y1, dr_x2, dr_y2 = dr
                            if not (bg_x2 < dr_x1 or bg_x1 > dr_x2 or bg_y2 < dr_y1 or bg_y1 > dr_y2):
                                collision = True
                                shift_y = text_h + 8
                                bg_y1 -= shift_y
                                bg_y2 -= shift_y
                                text_y -= shift_y
                                break
                        attempts += 1
                        
                    drawn_text_rects.append([bg_x1, bg_y1, bg_x2, bg_y2])

                    # 绘制带边框的深灰底色标签牌
                    cv2.rectangle(global_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), (30, 30, 30), -1) 
                    cv2.rectangle(global_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), dynamic_color, 1)         
                    
                    # 绘制纯白文字
                    cv2.putText(global_canvas, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (250, 250, 250), font_thickness)

                particle_count += 1

        # 裁剪掉 padding 部分并保存
        final_result = global_canvas[:img.shape[0], :]
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{base_name}_quantified.png")
        cv2.imencode('.png', final_result)[1].tofile(save_path)
        print(f"🎉 量化推理完成！共分析 {particle_count} 个矿石。成分热力图已保存至: {save_path}")


    def infer_and_draw_bake(self, hdr_path, output_dir, crop_height=480, crop_width=640):
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

        # 1. 制作全局底图画板
        rgb_img = data_cube[:, :, self.rgb_bands]
        rgb_img_8bit = self._stretch_to_8bit(rgb_img)
        global_canvas = cv2.cvtColor(rgb_img_8bit, cv2.COLOR_RGB2BGR)

        # 2. 计算全局一阶导数 
        deriv_cube = np.diff(data_cube, axis=2).astype(np.float32)

        # 补齐逻辑
        pad_h = max(0, crop_height - h)
        if pad_h > 0:
            global_canvas = cv2.copyMakeBorder(global_canvas, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            deriv_cube = np.pad(deriv_cube, ((0, pad_h), (0, 0), (0, 0)), mode='constant')
            h = crop_height

        num_crops = math.ceil(h / crop_height)
        start_y_list = [0] if num_crops == 1 else [int(round(i * (h - crop_height) / (num_crops - 1))) for i in range(num_crops)]

        particle_count = 0
        drawn_text_rects = [] # 用于记录已绘制的文字背景框，防止重叠！

        for start_y in start_y_list:
            end_y = start_y + crop_height
            crop_bgr = global_canvas[start_y:end_y, :crop_width].copy()

            # 3. YOLO 空间推断 
            results = self.yolo(crop_bgr, conf=0.4, verbose=False, retina_masks=True)[0]
            if results.masks is None:
                continue

            # 4. 颗粒级物理提取与化学推理
            for mask_data in results.masks.data:
                mask_np = mask_data.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (crop_width, crop_height)) > 0.5
                
                local_y, local_x = np.where(mask_resized)
                if len(local_y) < 50: 
                    continue
                    
                global_y = local_y + start_y
                global_x = local_x


                # 1. 提取用于 1D-CNN 的一阶导数光谱
                particle_spectra_deriv = deriv_cube[global_y, global_x, :]
                X_tensor = torch.tensor(particle_spectra_deriv, dtype=torch.float32).to(self.device)

                # 💥 2. 提取用于物理量化的“原始光谱”
                particle_spectra_raw = data_cube[global_y, global_x, :]
                # 必须做归一化，消除石头亮暗和距离的影响
                X_norm_raw = particle_spectra_raw / (np.max(particle_spectra_raw, axis=1, keepdims=True) + 1e-8)

                # ==========================================
                # 🛡️ 工业级物理规则：波段深度 (Band Depth) 拦截
                # ==========================================
                # 1400nm 坑底大约在 124，我们取左右肩和坑底的区间均值抗噪
                left_shoulder  = np.mean(X_norm_raw[:, 118:121], axis=1) # 坑左侧的平缓区
                right_shoulder = np.mean(X_norm_raw[:, 127:130], axis=1) # 坑右侧的平缓区
                valley_bottom  = np.mean(X_norm_raw[:, 123:126], axis=1) # 坑底

                # 计算理论深度：假设没有坑，两肩连线中点的值 减去 实际坑底的值
                band_depths = ((left_shoulder + right_shoulder) / 2.0) - valley_bottom
                
                # 获取该石头所有像素的平均坑深
                mean_1400_depth = np.mean(band_depths)
                
                print(f"[{particle_count}号石头] 真实波段深度: {mean_1400_depth:.4f}")

                # 🚨 观察打印出的真实波段深度，重新设定这个阈值！
                # 通常真实的深坑会大于 0.05 或 0.1，平滑的石头在 0 附近甚至为负数
                HARD_THRESHOLD = 0.04  

                if mean_1400_depth > HARD_THRESHOLD:
                    # 触发物理规则：深坑矿石 (比如 T1 硅)
                    final_class = 1 
                    confidence = 99.9 
                else:
                    # ==========================================
                    # 如果坑很浅/没坑，交给 1D-CNN + GMM 判断微小差异
                    # ==========================================
                    with torch.no_grad():
                        latent_features, _ = self.ae_model(X_tensor)
                    latent_np = latent_features.cpu().numpy()
                    
                    # 此时 GMM 不需要 6 维了，因为深坑已经被提前拦截掉了！
                    # 我们回退到 5 维 GMM 进行分类 (确保你用的是最初那个 5 维的 pkl 模型)
                    probs = self.gmm.predict_proba(latent_np)
                    mean_prob = np.mean(probs, axis=0)
                    final_class = int(np.argmax(mean_prob))
                    confidence = mean_prob[final_class] * 100


                # AE 降维与 GMM 概率聚合
                # with torch.no_grad():
                #     latent_features, _ = self.ae_model(X_tensor)
                # latent_np = latent_features.cpu().numpy()
                
                # probs = self.gmm.predict_proba(latent_np)
                # mean_prob = np.mean(probs, axis=0)
                # final_class = int(np.argmax(mean_prob))
                # confidence = mean_prob[final_class] * 100

                # ==========================================
                # 5. 高级 UI 渲染：防遮挡画图逻辑
                # ==========================================
                color = self.colors.get(final_class, (255, 255, 255))
                
                # 画出稍微透明的掩膜轮廓 (细线)
                contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt[:, 0, 1] += start_y 
                    cv2.drawContours(global_canvas, [cnt], -1, color, thickness=1)
                    
                    # 获取该石头的外接矩形，用于定位文字
                    x, y, w, box_h = cv2.boundingRect(cnt)
                    
                    # 极简标签
                    label_text = f"T{final_class}|{confidence:.1f}%"
                    font_scale = 0.45
                    font_thickness = 1
                    
                    # 计算文字需要的空间
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # 初始定位：放在石头左上角上方
                    text_x = x
                    text_y = y - 6
                    
                    # 边界保护：如果顶到屏幕最上面了，就画在石头下面
                    if text_y - text_h < 0:
                        text_y = y + box_h + text_h + 6
                        
                    # 计算背景色块的矩形坐标
                    bg_x1 = max(0, text_x - 3)
                    bg_y1 = text_y - text_h - 4
                    bg_x2 = text_x + text_w + 3
                    bg_y2 = text_y + baseline + 2
                    
                    # 💥 防碰撞躲避逻辑：检查是否与之前的标签打架了
                    collision = True
                    max_attempts = 5 # 最多向上躲避 5 次
                    attempts = 0
                    while collision and attempts < max_attempts:
                        collision = False
                        for dr in drawn_text_rects:
                            dr_x1, dr_y1, dr_x2, dr_y2 = dr
                            # 如果两个矩形相交
                            if not (bg_x2 < dr_x1 or bg_x1 > dr_x2 or bg_y2 < dr_y1 or bg_y1 > dr_y2):
                                collision = True
                                # 发生碰撞，把当前标签往上推移
                                shift_y = text_h + 8
                                bg_y1 -= shift_y
                                bg_y2 -= shift_y
                                text_y -= shift_y
                                break
                        attempts += 1
                        
                    # 记录成功安置的色块坐标
                    drawn_text_rects.append([bg_x1, bg_y1, bg_x2, bg_y2])

                    # 绘制带边框的黑底标签牌
                    cv2.rectangle(global_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), (25, 25, 25), -1) # 深灰底色
                    cv2.rectangle(global_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)         # 类别边框
                    
                    # 绘制纯白文字
                    cv2.putText(global_canvas, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), font_thickness)

                particle_count += 1

        # 裁剪掉 padding 部分并保存
        final_result = global_canvas[:img.shape[0], :]
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{base_name}_inference.png")
        cv2.imencode('.png', final_result)[1].tofile(save_path)
        print(f"🎉 推理完成！共精准渲染 {particle_count} 个矿石。漂亮的可视化结果已保存至: {save_path}")

# ==========================================
# 3. 执行测试
# ==========================================
if __name__ == "__main__":
    # 配置模型路径 (确保这些 pth 和 pkl 文件在你的运行目录下)
    YOLO_MODEL = r"F:\weiyutao\work\ai\hsi\runs\segment\runs\hsi_seg\train_v1\weights\best.pt"
    AE_MODEL = r"industrial_1d_ae.pth"  # <- 注意名字改成了 1D 版的
    GMM_MODEL = r"industrial_gmm.pkl"
    
    # 测试图像
    TEST_HDR = r"C:\Users\Administrator\Documents\SortingExpert\Files\甘总b料-20260326.hdr"
    OUTPUT_DIR = r"F:\weiyutao\work\data\hsi\inference_results"

    # 启动推理
    engine = HSIInferenceEngine(YOLO_MODEL, AE_MODEL, GMM_MODEL)
    engine.infer_and_draw(TEST_HDR, OUTPUT_DIR)