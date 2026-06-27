# the his data industrial_vision pipeline
## train pipeline
```
step1: extract three dimension data from the raw data what the dimension is 204.
step2: extract mask matrix from the three dimension RGB data by yolov8-seg what is the deep learning model.
step3: extrat the feature data from the raw data using the autoencoder.
step4: unsupervised deep clustering using the GMM algorithm what is the machine learnig model.
step5: constructing the particle-scale aluminum-silicon ratio.
```
raw data -> seg -> 自编码器特征提取 -> 100, 5(GMM) / 回归分析 -> 100, 1  1(3.5)

# 线性变换

100， 204 -> 100, 5 -> 100, 204
            1400nm




无监督 -> 铝硅比  0-1

50-100
40 * 100 = 4000
训练数据   label
100，5 ->  3.5
2 - 6
0.1
40


推理
seg

index1  100, 204  -> 100, 5 -> 3.5

我现在已经完成了c++代码的底层项目，我现在希望可以对接yolov8 tensorrt模型去实现对图像的分割，然后将分割的结果渲染成为
  彩图输出，输出的方式已经确定了，你要懂，你只需要把调用tensorrt那部分代码写好就行，调用的模型我已经定义好了









   ---
  根因确认：SG 边界处理方式不同，特征偏差高达 19%

  这就是在线和离线结果差异大的直接原因。

  ┌──────────────┬────────────────────────────────────────┬───────────────────────────────────────┐
  │              │          Python 离线（训练）           │           C++ 在线（推理）            │
  ├──────────────┼────────────────────────────────────────┼───────────────────────────────────────┤
  │ SG 边界      │ scipy 默认 mode='interp'（多项式插值） │ 手写 mirror（镜像反射）               │
  ├──────────────┼────────────────────────────────────────┼───────────────────────────────────────┤
  │ 均值特征 MAE │ —                                      │ 19% 的系统性偏差                      │
  ├──────────────┼────────────────────────────────────────┼───────────────────────────────────────┤
  │ 最大误差波段 │ —                                      │ band 86，差了 37%（-0.171 vs -0.231） │
  └──────────────┴────────────────────────────────────────┴───────────────────────────────────────┘

  这 19% 的特征偏差是在进入 XGBoost 之前的，树模型没见过这个分布，预测结果当然大幅偏移。

  修复方式：让 C++ 和 Python 用同一种边界处理。有两个方向：

  方案A（推荐）：修改 C++ 改用 interp 边界 — 逻辑上是重新实现 scipy 的插值边界，比较复杂。

  方案B（更简单）：修改 Python 训练，改用 mirror 边界 — 在训练代码里把 savgol_filter 改成 mode='mirror'，然后重新训练模型，这样以后 C++ 的 mirror
  就是对的了。

  # 把这行
  smoothed = savgol_filter(spec, window, poly, axis=1)
  # 改成
  smoothed = savgol_filter(spec, window, poly, axis=1, mode='mirror')

  但这需要重新训练模型。如果你不想重训，就要改 C++ 端。

  我帮你改 C++ 的实现，把边界模式从 mirror 改成和 scipy interp 等价的多项式插值？还是你倾向于改 Python 端重训？