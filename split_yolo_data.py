import os
import shutil
import random

def split_yolo_dataset(source_dir, dest_dir, split_ratio=0.8):
    """
    将 Label Studio 导出的 YOLO 数据集拆分为训练集和验证集。
    
    :param source_dir: Label Studio 导出的原始文件夹路径 (包含 images, labels, classes.txt)
    :param dest_dir: 拆分后生成的新数据集存放路径
    :param split_ratio: 训练集所占比例，默认 0.8 (80% 训练，20% 验证)
    """
    
    # 原始路径配置
    src_images_dir = os.path.join(source_dir, 'images')
    src_labels_dir = os.path.join(source_dir, 'labels')
    src_classes_file = os.path.join(source_dir, 'classes.txt')
    
    # 目标路径配置
    train_images_dir = os.path.join(dest_dir, 'images', 'train')
    val_images_dir = os.path.join(dest_dir, 'images', 'val')
    train_labels_dir = os.path.join(dest_dir, 'labels', 'train')
    val_labels_dir = os.path.join(dest_dir, 'labels', 'val')
    dest_classes_file = os.path.join(dest_dir, 'classes.txt')
    
    # 1. 创建目标文件夹 (如果存在则清理后重建，防止数据污染)
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(d, exist_ok=True)
        
    # 2. 获取所有图片文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    all_images = [f for f in os.listdir(src_images_dir) if f.lower().endswith(supported_formats)]
    
    if not all_images:
        print("❌ 错误：在源目录中没有找到图片文件！请检查路径。")
        return

    # 3. 随机打乱数据，保证训练集和验证集分布均匀
    random.shuffle(all_images)
    
    # 4. 计算切分索引
    split_index = int(len(all_images) * split_ratio)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]
    
    print(f"🔄 开始处理数据... (总数: {len(all_images)} | 训练集: {len(train_images)} | 验证集: {len(val_images)})")

    # 5. 核心拷贝函数 (匹配图片和标签)
    def copy_data(image_list, target_img_dir, target_lbl_dir):
        copied_count = 0
        missing_label_count = 0
        
        for img_name in image_list:
            # 图片源路径与目标路径
            src_img_path = os.path.join(src_images_dir, img_name)
            dst_img_path = os.path.join(target_img_dir, img_name)
            
            # 推导对应的标签文件名 (将后缀替换为 .txt)
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + '.txt'
            src_lbl_path = os.path.join(src_labels_dir, lbl_name)
            dst_lbl_path = os.path.join(target_lbl_dir, lbl_name)
            
            # 拷贝图片
            shutil.copy2(src_img_path, dst_img_path)
            
            # 拷贝标签 (如果存在的话)
            if os.path.exists(src_lbl_path):
                shutil.copy2(src_lbl_path, dst_lbl_path)
                copied_count += 1
            else:
                # 针对 Label Studio 导出时，没有标注的负样本可能没有 txt 文件的情况
                # YOLO 可以接受没有 txt 文件的背景图片，但我们做个统计
                missing_label_count += 1
                
        return copied_count, missing_label_count

    # 6. 执行拷贝
    train_copied, train_missing = copy_data(train_images, train_images_dir, train_labels_dir)
    val_copied, val_missing = copy_data(val_images, val_images_dir, val_labels_dir)
    
    # 7. 拷贝 classes.txt
    if os.path.exists(src_classes_file):
        shutil.copy2(src_classes_file, dest_classes_file)
    else:
        print("⚠️ 警告：未找到 classes.txt 文件。")

    # 8. 打印报告
    print("\n" + "="*40)
    print("✅ 数据集划分完成！")
    print("="*40)
    print(f"📂 输出目录: {dest_dir}")
    print(f"📊 训练集 (Train): {len(train_images)} 张图片, {train_copied} 个有效标签文件")
    print(f"📊 验证集 (Val)  : {len(val_images)} 张图片, {val_copied} 个有效标签文件")
    if train_missing > 0 or val_missing > 0:
        print(f"⚠️ 提示: 共有 {train_missing + val_missing} 张图片没有对应的 txt 标签文件。")
        print(f"   (这些通常是你在 Label Studio 中未添加任何框的图片，YOLO 将它们视为纯背景/负样本，这是正常的)。")
    print("========================================")


if __name__ == '__main__':
    # ================= 配置区 =================
    # 1. 替换为你的 Label Studio 解压后的文件夹名称（根据你的截图填写的）
    SOURCE_DIRECTORY = 'project-13-at-2026-06-11-08-45-5d6e60b0'
    
    # 2. 想要保存的最终 YOLO 数据集名称
    DESTINATION_DIRECTORY = 'alsi_seg_202606111647_v5'
    
    # 3. 划分比例 (0.8 表示 80% 训练集，20% 验证集)
    SPLIT_RATIO = 0.8
    # ==========================================
    
    split_yolo_dataset(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, SPLIT_RATIO)