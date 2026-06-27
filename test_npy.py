import numpy as np

# 1. 加载 .npy 文件 (将 'your_file.npy' 替换为您的实际文件名)
data = np.load(r'C:\Users\Administrator\Desktop\npy\particle_3 (1).npy')

# 2. 打印数据的维度 (shape)
print("数据的维度是:", data.shape)

# 如果您想知道总共有多少个维度（例如：1维、2维、3维）
print("数据的维度数 (ndim) 是:", data.ndim)