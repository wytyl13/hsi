#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/28 15:33
@Author  : weiyutao
@File    : hsi_inference_test.py
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/28
@Author  : Gemini
@File    : hsi_edge_client.py
@Desc    : 工控机端高光谱图像处理与自动上报脚本 (零落盘、全内存流)
"""

import os
import io
import math
import glob
import json
import time
import cv2
import requests
import numpy as np
import spectral.io.envi as envi
from ultralytics import YOLO

class HSIEdgeProcessor:
    def __init__(self, yolo_weights, api_url, crop_height=480, crop_width=640, rgb_bands=(76, 51, 25), expected_bands=104):
        # 101, 92, 77
        # 150, 100, 50
        print("🤖 [初始化] 正在加载边缘计算 YOLOv8 模型...")
        self.yolo = YOLO(yolo_weights)
        self.api_url = api_url
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.rgb_bands = list(rgb_bands)
        self.expected_bands = expected_bands
        print(f"🔗 [初始化] 目标服务器接口: {self.api_url}")


    def _stretch_to_8bit(self, img_array, ignore_margin=150):
        """将高维光谱数据拉伸为可见的 RGB 图像，支持指定边缘忽略区"""
        img_array = img_array.astype(np.float32)
        res = np.zeros_like(img_array)
        w = img_array.shape[1]
        
        for i in range(img_array.shape[2]):
            channel = img_array[:, :, i]
            
            if w > ignore_margin * 2:
                clean_roi = channel[:, ignore_margin : w - ignore_margin]
            else:
                clean_roi = channel
                
            # ==========================================
            # 🌟 核心破局点：修改百分位，暴力压黑背景！
            # p_low = 40: 把最暗的 40% (即皮带) 强行压成纯黑
            # p_high = 99.9: 矿石占比极小，必须逼近极值才能防止皮带过曝
            # ==========================================
            p_low, p_high = np.percentile(clean_roi, (40, 99.9))
            
            if p_high - p_low > 1e-6:
                # np.clip 会把低于 p_low 的值直接变成 0，高于 p_high 的直接变成 1
                channel = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
            else:
                c_min, c_max = np.min(clean_roi), np.max(clean_roi)
                if c_max > c_min:
                    channel = (channel - c_min) / (c_max - c_min)
                else:
                    channel = np.zeros_like(channel)
                    
            res[:, :, i] = channel
            
        return (res * 255).astype(np.uint8)



    def _stretch_to_8bit_bak(self, img_array):
        """将高维光谱数据拉伸为可见的 RGB 图像"""
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
    

    def process_and_upload(self, hdr_path, batch_name):
        """核心流：读取 -> 推理 -> 分割提取 -> 内存打包 -> API 发送"""
        spe_path = hdr_path.rsplit('.', 1)[0] + '.spe'
        base_name = os.path.basename(hdr_path).replace('.hdr', '')
        
        print(f"\n---> [开始处理] 图像: {base_name}")
        start_time = time.time()
        
        try:
            # ==========================================
            # 1. 数据加载 (坚守原则：绝不动原始像素值！)
            # ==========================================
            img = envi.open(hdr_path, spe_path)
            raw_data_cube = np.asarray(img.load())
            h, w_raw, c = raw_data_cube.shape

            if c != self.expected_bands:
                print(f"⚠️ [跳过] {base_name}: 波段数 {c} 异常。")
                return False

            # 尺寸对齐：只裁剪多余视场或补齐黑边，不碰波段像素
            if w_raw > self.crop_width:
                working_data_cube = raw_data_cube[:, :self.crop_width, :]
            elif w_raw < self.crop_width:
                pad_w = self.crop_width - w_raw
                working_data_cube = np.pad(raw_data_cube, ((0, 0), (0, pad_w), (0, 0)), mode='constant')
            else:
                working_data_cube = raw_data_cube
            w = self.crop_width # 此时统一为 640

            # ==========================================
            # 🌟 2. 制作 YOLO 视觉替身 (物理隔离)
            # ==========================================
            margin = 150 
            rgb_raw = working_data_cube[:, :, self.rgb_bands]
            
            # 替身翻转：只把发给 YOLO 的替身做镜像，使其对齐物理气阀坐标
            rgb_flipped = np.flip(rgb_raw, axis=1) 
            bgr_img = cv2.cvtColor(self._stretch_to_8bit(rgb_flipped, ignore_margin=margin), cv2.COLOR_RGB2BGR)

            # 替身涂黑：防止 YOLO 被边缘条纹干扰 (替身已翻转，涂两边效果一样)
            if w > margin * 2:
                bgr_img[:, :margin] = 0        
                bgr_img[:, w-margin:] = 0      

            # ==========================================
            # 3. YOLO 滑窗推理
            # ==========================================
            global_binary_mask = np.zeros((h, w), dtype=bool)
            num_crops = math.ceil(h / self.crop_height)
            start_y_list = [0] if num_crops == 1 else [int(round(i * (h - self.crop_height) / (num_crops - 1))) for i in range(num_crops)]

            for start_y in start_y_list:
                end_y = start_y + self.crop_height
                crop_bgr = bgr_img[start_y:end_y, :]
                
                results = self.yolo(crop_bgr, conf=0.4, verbose=False, retina_masks=True)
                if results[0].masks is not None:
                    for mask_tensor in results[0].masks.data:
                        local_mask = cv2.resize(mask_tensor.cpu().numpy(), (self.crop_width, self.crop_height)) > 0.5
                        global_binary_mask[start_y:end_y, :] |= local_mask

            # ==========================================
            # 4. 连通域分离与特征提取 (双向闭环)
            # ==========================================
            mask_uint8 = (global_binary_mask.astype(np.uint8) * 255)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
            
            global_mask_visual = np.zeros((h, w), dtype=np.uint8)
            particles_info = []
            particles_file_payloads = [] 

            # 🌟 神来之笔：把标签图“逆向翻转”回去，以完美对齐原始纯净的高光谱数据！
            original_direction_labels = np.flip(labels, axis=1)

            particle_idx = 0
            for label in range(1, num_labels):
                x, y, w_box, h_box, area = stats[label]
                cx, cy = centroids[label]

                if area < 30:  
                    continue
                
                # 🌟 用逆向翻转后的 Mask 去纯净的数据源里提数据，毫无污染！
                single_mask_original = (original_direction_labels == label)
                particle_pixels = working_data_cube[single_mask_original]  
                
                # 🛡️ 绝杀防爆：过滤边缘截断残次品，彻底干掉云端数据库的 NaN 崩溃！
                if particle_pixels.shape[0] < 10:
                    continue
                
                # 用于后台展示的 global mask 依然保持对齐物理气阀的正向视角
                global_mask_visual[labels == label] = 255
                filename = f"particle_{particle_idx}.npy"
                
                npy_stream = io.BytesIO()
                np.save(npy_stream, particle_pixels)
                npy_bytes = npy_stream.getvalue()
                
                particles_info.append({
                    "idx": particle_idx,
                    "filename": filename,
                    "pixel_count": int(area),
                    "is_ore": False,
                    "centroid_x": float(cx), # 发给后端的 cx 就是工厂气阀的绝对坐标！
                    "centroid_y": float(cy),
                    "bbox_x": int(x),        
                    "bbox_y": int(y),
                    "bbox_w": int(w_box),
                    "bbox_h": int(h_box)
                })
                
                particles_file_payloads.append(
                    ('npy_files', (filename, npy_bytes, 'application/octet-stream'))
                )
                particle_idx += 1

            # ==========================================
            # 5. 图像编码与多部分上传
            # ==========================================
            _, bgr_encoded = cv2.imencode('.jpg', bgr_img)
            _, mask_encoded = cv2.imencode('.png', global_mask_visual)

            meta_data = {
                "batch_name": batch_name,
                "image_name": base_name,
                "width": w,
                "height": h,
                "particles_info": particles_info
            }

            files = [
                ('rgb_file', (f"{base_name}_rgb.jpg", bgr_encoded.tobytes(), 'image/jpeg')),
                ('mask_file', (f"{base_name}_mask.png", mask_encoded.tobytes(), 'image/png'))
            ]
            files.extend(particles_file_payloads)

            data = {'meta_data_str': json.dumps(meta_data)}

            print(f"📡 [上传中] {base_name} 包含 {len(particles_info)} 个颗粒...")
            response = requests.post(self.api_url, data=data, files=files, timeout=60)
            
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("success"):
                    print(f"✅ [成功] {base_name} 成功入库。耗时: {time.time() - start_time:.2f}秒")
                    return True
                else:
                    print(f"❌ [业务失败] 服务器返回: {res_json.get('message')}")
                    return False
            else:
                print(f"❌ [网络错误] HTTP状态码: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ [客户端奔溃] 处理 {base_name} 发生严重错误: {str(e)}")
            return False



    def process_and_upload_bak(self, hdr_path, batch_name):
        """核心流：读取 -> 推理 -> 分割提取 -> 内存打包 -> API 发送"""
        spe_path = hdr_path.rsplit('.', 1)[0] + '.spe'
        base_name = os.path.basename(hdr_path).replace('.hdr', '')
        
        print(f"\n---> [开始处理] 图像: {base_name}")
        start_time = time.time()
        
        try:
            # ==========================================
            # 1. 数据加载与预处理
            # ==========================================
            img = envi.open(hdr_path, spe_path)
            data_cube = np.asarray(img.load())
            data_cube = np.flip(data_cube, axis=1)
            h, w, c = data_cube.shape

            if c != self.expected_bands:
                print(f"⚠️ [跳过] {base_name}: 波段数 {c} 异常。")
                return False

            margin = 150 
            rgb_img = data_cube[:, :, self.rgb_bands]
            
            bgr_img = cv2.cvtColor(self._stretch_to_8bit(rgb_img, ignore_margin=margin), cv2.COLOR_RGB2BGR)

            if w > margin * 2:
                bgr_img[:, :margin] = 0        # 左边 150 涂黑
                bgr_img[:, w-margin:] = 0      # 右边 150 涂黑
                

            # 宽度对齐
            if w > self.crop_width:
                bgr_img = bgr_img[:, :self.crop_width]
                data_cube = data_cube[:, :self.crop_width, :]
                w = self.crop_width
            elif w < self.crop_width:
                pad_w = self.crop_width - w
                bgr_img = cv2.copyMakeBorder(bgr_img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                data_cube = np.pad(data_cube, ((0, 0), (0, pad_w), (0, 0)), mode='constant')
                w = self.crop_width

            # ==========================================
            # 2. YOLO 滑窗推理与全局掩码合并
            # ==========================================
            global_binary_mask = np.zeros((h, w), dtype=bool)
            num_crops = math.ceil(h / self.crop_height)
            start_y_list = [0] if num_crops == 1 else [int(round(i * (h - self.crop_height) / (num_crops - 1))) for i in range(num_crops)]

            for start_y in start_y_list:
                end_y = start_y + self.crop_height
                crop_bgr = bgr_img[start_y:end_y, :]
                
                results = self.yolo(crop_bgr, conf=0.4, verbose=False, retina_masks=True)
                if results[0].masks is not None:
                    for mask_tensor in results[0].masks.data:
                        local_mask = cv2.resize(mask_tensor.cpu().numpy(), (self.crop_width, self.crop_height)) > 0.5
                        global_binary_mask[start_y:end_y, :] |= local_mask

            # ==========================================
            # 3. 连通域颗粒分离与特征提取 (写入内存流)
            # ==========================================
            mask_uint8 = (global_binary_mask.astype(np.uint8) * 255)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
            
            global_mask_visual = np.zeros((h, w), dtype=np.uint8)
            particles_info = []
            particles_file_payloads = []  # 存放 npy 的表单文件结构

            particle_idx = 0
            # 注意: label 0 是背景，从 1 开始遍历
            for label in range(1, num_labels):
                # 解析当前连通域的几何统计信息
                x, y, w_box, h_box, area = stats[label]
                cx, cy = centroids[label]

                if area < 30:  # 用 area 过滤极小噪点，比 particle_pixels.shape[0] 更快
                    continue
                
                single_mask = (labels == label)
                particle_pixels = data_cube[single_mask]  
                
                global_mask_visual[single_mask] = 255
                filename = f"particle_{particle_idx}.npy"
                
                # 💥 核心：不落盘，直接写入 BytesIO
                npy_stream = io.BytesIO()
                np.save(npy_stream, particle_pixels)
                npy_bytes = npy_stream.getvalue()
                
                particles_info.append({
                    "idx": particle_idx,
                    "filename": filename,
                    "pixel_count": int(area),
                    "is_ore": False,
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                    "bbox_x": int(x),
                    "bbox_y": int(y),
                    "bbox_w": int(w_box),
                    "bbox_h": int(h_box)
                })
                
                # 组装 NPY 的 request files 格式
                particles_file_payloads.append(
                    ('npy_files', (filename, npy_bytes, 'application/octet-stream'))
                )
                particle_idx += 1

            # ==========================================
            # 4. 图像编码 (同样写入内存)
            # ==========================================
            _, bgr_encoded = cv2.imencode('.jpg', bgr_img)
            _, mask_encoded = cv2.imencode('.png', global_mask_visual)

            # ==========================================
            # 5. 组装并发送 Multipart POST 请求
            # ==========================================
            meta_data = {
                "batch_name": batch_name,
                "image_name": base_name,
                "width": w,
                "height": h,
                "particles_info": particles_info
            }

            # 构建完整的 files 列表
            files = [
                ('rgb_file', (f"{base_name}_rgb.jpg", bgr_encoded.tobytes(), 'image/jpeg')),
                ('mask_file', (f"{base_name}_mask.png", mask_encoded.tobytes(), 'image/png'))
            ]
            files.extend(particles_file_payloads)

            data = {
                'meta_data_str': json.dumps(meta_data)
            }

            print(f"📡 [上传中] {base_name} 包含 {len(particles_info)} 个颗粒，请求大小约 {sum([len(f[1][1]) for f in files])/1024/1024:.2f} MB")
            
            # 发送请求，设置合适的超时时间
            response = requests.post(self.api_url, data=data, files=files, timeout=60)
            
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("success"):
                    cost_time = time.time() - start_time
                    print(f"✅ [成功] {base_name} 成功入库。耗时: {cost_time:.2f}秒")
                    return True
                else:
                    print(f"❌ [业务失败] 服务器返回: {res_json.get('message')}")
                    return False
            else:
                print(f"❌ [网络错误] HTTP 状态码: {response.status_code}, 内容: {response.text}")
                return False

        except Exception as e:
            print(f"❌ [客户端奔溃] 处理 {base_name} 发生严重错误: {str(e)}")
            return False


    def run_directory_watcher(self, watch_dir, batch_name):
        """扫描目录并处理文件"""
        print(f"📂 [开始任务] 监控目录: {watch_dir} | 批次名: {batch_name}")
        
        # 获取所有 hdr 文件
        hdr_files = glob.glob(os.path.join(watch_dir, '*.hdr'))
        if not hdr_files:
            print("⚠️ 警告：目录中未找到任何 .hdr 文件。")
            return
            
        success_count = 0
        for hdr_path in hdr_files:
            # 调用单图处理上传逻辑
            is_success = self.process_and_upload(hdr_path, batch_name)
            
            if is_success:
                success_count += 1
                
                # 可选：处理成功后删除本地原图以释放工控机磁盘空间
                # os.remove(hdr_path)
                # os.remove(hdr_path.replace('.hdr', '.spe'))
                
            # 缓冲一下，防止把服务器打满
            time.sleep(0.5)

        print(f"\n🎉 [批次结束] {batch_name} 处理完毕。成功: {success_count}/{len(hdr_files)}")


# ==========================================
# 工控机启动入口
# ==========================================
if __name__ == "__main__":
    # --- 配置区 (根据现场工控机环境修改) ---
    # YOLO_WEIGHTS = r"F:\weiyutao\work\ai\hsi\runs\segment\runs\alsi_seg_20260519_v1\train_v1_20260519\weights\best.pt"
    YOLO_WEIGHTS = r"F:\weiyutao\work\ai\hsi\runs\segment\runs\alsi_seg_20260523_v2\alsi_seg_20260523_v2\weights\best.pt"
    
    # 指向你的 FastAPI 服务器地址
    # 假设你的 FastAPI 跑在 192.168.1.100 的 8000 端口
    API_URL = "http://10.0.26.209:9039/api/storage/upload/hsi_processed_image" 
    
    # 本地高光谱数据目录
    # INPUT_DIR = r"F:\500块采样"
    INPUT_DIR = r"F:\weiyutao\work\data\hsi\data_0612_new"
    
    # 批次名称
    # BATCH_NAME = "平陆一诺铝土矿500num20260527"
    BATCH_NAME = "data_0612_new"

    # 初始化处理引擎
    processor = HSIEdgeProcessor(
        yolo_weights=YOLO_WEIGHTS,
        api_url=API_URL,
        crop_height=480,
        crop_width=640
    )
    
    # 启动任务
    processor.run_directory_watcher(INPUT_DIR, BATCH_NAME)