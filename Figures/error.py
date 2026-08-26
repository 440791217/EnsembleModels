import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 创建画布
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# 1. 硬件层框图
hw_box = patches.FancyBboxPatch((0.5, 2), 2, 2, boxstyle="round,pad=0.3", ec="black", fc="#f0f0f0", lw=1.5)
ax.add_patch(hw_box)
ax.text(1.5, 3.3, "Hardware Layer", fontsize=11, fontweight='bold', ha='center')
ax.text(1.5, 2.7, "SRAM / Register\n(Bit-Flip / SEU)", fontsize=9, ha='center', color='red')

# 箭头 1
ax.annotate('', xy=(3.2, 3), xytext=(2.6, 3),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# 2. 中间层传播框图
net_box = patches.FancyBboxPatch((3.5, 1), 4.5, 4, boxstyle="round,pad=0.3", ec="black", fc="#e6f2ff", lw=1.5)
ax.add_patch(net_box)
ax.text(5.75, 4.3, "DNN Forward Propagation", fontsize=11, fontweight='bold', ha='center')
ax.text(5.75, 3.5, "Feature Map [i]\n(Injected Error)", fontsize=9, ha='center', color='darkorange')
ax.text(5.75, 2.5, "Propagation & Accumulation\n(ReLU / Conv layers)", fontsize=9, ha='center')

# 箭头 2
ax.annotate('', xy=(8.4, 3), xytext=(8.0, 3),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# 3. 输出层框图
out_box = patches.FancyBboxPatch((8.7, 2), 2.8, 2, boxstyle="round,pad=0.3", ec="black", fc="#fff0f0", lw=1.5)
ax.add_patch(out_box)
ax.text(10.1, 3.3, "Output Layer", fontsize=11, fontweight='bold', ha='center')
ax.text(10.1, 2.6, "Probability Shift\n-> CSDC / SDC", fontsize=9, ha='center', color='darkred')

# 全局大标题
plt.title("Illustration of Soft Error Propagation and Failure Mechanism in DNNs", fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("dnn_soft_error_mechanism.pdf", bbox_inches='tight') # 可直接导出 PDF 用于论文
plt.show()