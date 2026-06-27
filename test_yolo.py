from ultralytics import YOLO

# 1. 加载你的模型
model = YOLO(r"C:\Users\Administrator\Desktop\gold_ore_images_3_train_20260402.pt")

# 2. 随便找一张你之前切好的、肉眼能看到石头的 PNG 图片路径
TEST_IMG = r"F:\weiyutao\work\data\hsi\yolo_dataset\images\1224_3采样_crop_005.png"

# 3. 让 YOLO 直接测这张图，并开启保存功能 (save=True)
print("开始单独测试 YOLO...")
results = model.predict(source=TEST_IMG, save=True, conf=0.05)

# 4. 打印结果
if results[0].masks is not None:
    print(f"✅ 成功！检测到 {len(results[0].masks)} 个矿石！")
    print(f"请去 runs/segment/predict 文件夹下查看画好框的图片。")
else:
    print("❌ 完蛋！检测到 0 个矿石。说明模型本身没训练好！")