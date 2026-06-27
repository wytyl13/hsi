import os
from ultralytics import YOLO
import torch

def check_gpu():
    """训练前的防呆检查：确保 5060 处于激活状态"""
    if torch.cuda.is_available():
        print(f"✅ GPU 已就绪: {torch.cuda.get_device_name(0)}")
        return "0"
    else:
        print("❌ 警告：未检测到 GPU，将使用 CPU 训练（极其缓慢）！请检查环境。")
        return "cpu"

def main():
    # 1. 环境检查
    device = check_gpu()

    # 2. 初始化模型
    # 我们做的是像素级抠图，必须使用带有 '-seg' 后缀的模型！
    # yolov8n-seg.pt (Nano版，速度极快)， yolov8s-seg.pt (Small版，精度和速度的平衡点，推荐 5060 使用)
    print("正在加载预训练权重...")
    model = YOLO(r"C:\Users\Administrator\Desktop\yolov8s-seg.pt") 

    # 3. 核心训练参数配置
    print("🚀 开始启动工业级训练流水线...")
    
    # 你的 data.yaml 文件的绝对路径 (千万别写错)
    DATA_YAML_PATH = r"F:\weiyutao\work\ai\hsi\data.yaml"

    results = model.train(
        data=DATA_YAML_PATH,     # 数据集配置文件路径
        epochs=100,              # 训练轮数（建议至少 100 轮起步，看收敛情况）
        imgsz=640,               # 图像输入尺寸（必须和你切割的尺寸一致）
        rect=True,
        batch=16,                 # 批次大小。5060 通常是 8GB 显存，设为 8 或 16 最佳
        device=device,           # 强制指定使用 GPU
        workers=4,               # DataLoader 的多线程读取数量（Windows 建议设为 4 或 8）
        
        # --- 进阶/防御性参数 ---
        patience=30,             # 早停机制：如果 30 轮 mAP 都没有提升，自动停止训练防止过拟合
        save=True,               # 保存最佳和最后的模型权重 (best.pt, last.pt)
        project="runs/alsi_seg_202606111647_v5",  # 训练结果保存的主目录
        name="alsi_seg_202606111647_v5",         # 本次训练的实验名称
        
        # --- 数据增强参数 (极度关键！这是解决你之前域漂移、认不出石头的救星) ---
        hsv_h=0.015,             # 色调增强 (轻微改变石头颜色，防止模型死记硬背颜色)
        hsv_s=0.7,               # 饱和度增强
        hsv_v=0.4,               # 亮度增强 (极其重要！让模型适应车间里忽明忽暗的光照)
        degrees=180.0,           # 随机旋转 180 度 (石头掉在皮带上的角度是随机的)
        flipud=0.5,              # 50% 概率上下翻转
        fliplr=0.5,              # 50% 概率左右翻转
    )
    
    print("\n🎉 训练任务执行完毕！")
    print("👉 请前往 runs/hsi_seg/train_v1/weights/ 目录下寻找你的 best.pt")

# Windows 环境下多线程训练必须写在 __main__ 保护块里
if __name__ == '__main__':
    main()