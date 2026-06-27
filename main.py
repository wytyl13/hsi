

import spectral
import numpy as np
import cv2
import pandas as pd
import spectral.io.envi as envi
import os



# ==========================================
# 0. 配置路径和参数
# ==========================================
hdr_path = r'C:\Users\Administrator\Documents\SortingExpert\Files\一诺-曹川-20260423.hdr'
spe_path = r'C:\Users\Administrator\Documents\SortingExpert\Files\一诺-曹川-20260423.spe'
output_dir = 'F:\\weiyutao\\work\\ai\\hsi\\results' # 保存文件夹
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. 读取高光谱数据
# ==========================================
print("正在读取高光谱数据 (约 2700x640x204)...")
img = envi.open(hdr_path, spe_path)
full_data = np.asarray(img.load())
print(f"✅ 成功加载！维度: {full_data.shape}")

# ==========================================
# 2. 抠图核心流程 (简单提取 + 进阶清理)
# ==========================================
print("\n正在从传送带上抠出纯颗粒...")

# a. 挑选一个近红外波段 (通常是第 100 个通道)
gray_band = full_data[:, :, 100]

# b. 设定阈值 (可以手动调节以获得理想效果)
# threshold = np.mean(gray_band) + 0.1 # 如果噪声多，可以把均值加上一个系数
threshold = 0.15 # 手动调节的最佳数值

# c. 生成【初始(带噪声)】掩膜 (第一张不干净 Mask 的来源)
raw_mask_bool = gray_band > threshold
# 为了可视化，转换为 0/255 的 8位图像
raw_mask_uint8 = (raw_mask_bool * 255).astype(np.uint8)

# d. 【最关键一步】进阶清理：应用形态学开运算 (先腐蚀后膨胀)
# 这一步是把小的、条纹状噪声抹去，只留下圆润的颗粒 Mask
kernel = np.ones((7,7), np.uint8) # 卷积核越大，清理越猛，但也可能漏掉小矿石
clean_mask_uint8 = cv2.morphologyEx(raw_mask_uint8, cv2.MORPH_OPEN, kernel)
# 转换为布尔型索引用于提取数据
clean_mask_bool = clean_mask_uint8.astype(bool)


# ==========================================
# 3. 提取纯颗粒像素并保存 CSV
# ==========================================
# 完美提取所有被标记为 True 的矿石像素，形状为 (样本数, 204)
ore_pixels = full_data[clean_mask_bool]
print(f"✅ 剔除背景和噪声后，成功提取到 {ore_pixels.shape[0]} 个纯矿石像素点！")

# # 保存为无标签的 CSV (准备喂给自编码器)
# csv_name = os.path.join(output_dir, "ore_cleaned_dataset.csv")
# print(f"正在保存纯颗粒数据集至 {csv_name} (请稍候)...")
# df = pd.DataFrame(ore_pixels)
# df.to_csv(csv_name, index=False)
# print(">>> CSV 保存完成！")


# ==========================================
# 4. 保存可视化对比图 (校验用)
# ==========================================
print("\n正在保存可视化对比图 (校验 Mask 质量)...")

# 1. 保存带噪声的初始 Mask (也就是你提到的那个不干净图)
cv2.imwrite(os.path.join(output_dir, "mask_0_noisy_binary.png"), raw_mask_uint8)

# 2. 保存清理后的干净二值 Mask (纯矿石轮廓)
cv2.imwrite(os.path.join(output_dir, "mask_1_cleaned_binary.png"), clean_mask_uint8)

# 3. 保存干净的高光对比图 (对应你提到的第二个理想图)
highlight_img = np.zeros_like(gray_band)
highlight_img[clean_mask_bool] = gray_band[clean_mask_bool]
# 归一化到 0-255 以保存
highlight_visual = (highlight_img / np.max(highlight_img) * 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, "mask_2_cleaned_highlight.png"), highlight_visual)

print(f"📸 校验图保存完成，请查看 {output_dir} 文件夹下的三张图片对比效果！")