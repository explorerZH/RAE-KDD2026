import matplotlib.pyplot as plt
import numpy as np

# 手工输入数据
weight_decay = [0,1e-10,1e-9,5e-9,1e-8,3e-8,5e-8,7e-8,
                9e-8,1e-7,1.5e-7,2e-7,3e-7,5e-7,7e-7,1e-6,2e-6]

top1_cos = [78.34,78.39,78.56,79.41,80.17,82.59,84.54,85.6,
            86.38,86.67,87.59,87.67,87.18,85.27,83.51,80.76,73.66]

top5_cos = [80.88,80.9,81.04,81.71,82.49,84.84,86.36,87.52,88.2,
            88.42,89.18,89.32,88.93,87.42,85.8,83.53,78]

top10_cos = [82.07,82.09,82.23,82.91,83.69,85.92,87.45,88.42,
             89.09,89.34,90,90.14,89.88,88.58,87.18,85.27,80.05]

top50_cos = [84.20,84.21,84.35,84.93,85.59,87.64,89.02,89.93,
             90.56,90.79,91.46,91.68,91.61,90.73,89.73,88.28,84.15]

# 分离0值和非0值
wd_nonzero = weight_decay[1:]
top1_nonzero = top1_cos[1:]
top5_nonzero = top5_cos[1:]
top10_nonzero = top10_cos[1:]
top50_nonzero = top50_cos[1:]

# # 画图
# plt.figure(figsize=(10, 6))
# plt.semilogx(wd_nonzero, top1_nonzero, 'o-', label='Top-1', markersize=6)
# plt.semilogx(wd_nonzero, top5_nonzero, 's-', label='Top-5', markersize=6)
# plt.semilogx(wd_nonzero, top10_nonzero, '^-', label='Top-10', markersize=6)
# plt.semilogx(wd_nonzero, top50_nonzero, 'd-', label='Top-50', markersize=6)

# plt.axvline(x=weight_decay[0] if weight_decay[0] > 0 else 1e-10, 
#            color='gray', linestyle='--', alpha=0.5, label=f'No WD (Top-5: {top5_cos[0]}%)')

# plt.xlabel('Weight Decay (log scale)', fontsize=12)
# plt.ylabel('k-NN Accuracy (%)', fontsize=12)
# plt.title('Weight Decay vs k-NN Accuracy(cosine)', fontsize=14)
# plt.grid(True, which="both", alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

# 画图
plt.figure(figsize=(10, 6))
plt.semilogx(weight_decay, top1_cos, 'o-', label='Top-1', markersize=6)
plt.semilogx(weight_decay, top5_cos, 's-', label='Top-5', markersize=6)
plt.semilogx(weight_decay, top10_cos, '^-', label='Top-10', markersize=6)
plt.semilogx(weight_decay, top50_cos, 'd-', label='Top-50', markersize=6)

# 添加No WD基准线（weight_decay=0时的值）
no_wd_value = 80.88  # weight_decay=0时的Top-1准确率
plt.axhline(y=no_wd_value, color='gray', linestyle='--', alpha=0.5, 
            label=f'No WD (Top-5: {no_wd_value}%)')

plt.xlabel('Weight Decay (log scale)', fontsize=12)
plt.ylabel('k-NN Accuracy (%)', fontsize=12)
plt.title('Weight Decay vs k-NN Accuracy(cosine)', fontsize=14)
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()