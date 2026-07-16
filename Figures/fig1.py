import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# 1. SCI 论文规范配置（字体、字号、线条、抗锯齿）
# =========================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================================
# 2. 从你上传的表格中提取的真实实验数据
# =========================================================================
# (1) 你的创新通道：ResNet AVG (18, 34, 50)
avg_coverage = np.array([100.0, 92.19, 90.97, 89.78, 87.89])
avg_risk     = np.array([4.22,  1.49,  1.23,  1.00,  0.77])

# (2) 对照组 1：大模型单体 ResNet101
r101_coverage = np.array([100.0, 96.84, 96.23, 95.38, 93.53])
r101_risk     = np.array([4.93,  3.36,  3.21,  2.88,  2.28])

# (3) 对照组 2：单体 ResNet50
r50_coverage = np.array([100.0, 96.52, 95.80, 94.83, 93.00])
r50_risk     = np.array([5.18,  3.43,  3.15,  2.81,  2.29])

# (4) 对照组 3：单体 ResNet34
r34_coverage = np.array([100.0, 96.57, 95.78, 94.79, 93.21])
r34_risk     = np.array([5.29,  3.54,  3.22,  2.87,  2.38 ])

# (5) 对照组 4：单体 ResNet18
r18_coverage = np.array([100.0, 95.70, 94.86, 93.72, 91.44])
r18_risk     = np.array([5.51,  3.45,  3.14,  2.80,  2.15])

# (6) 【全新加入】对照组 5：传统多数投票系统 (Traditional Majority Voting) - 无拒绝
traditional_vt_coverage = 100.0
traditional_vt_risk     = 4.40

# (7) 【全新加入】对照组 6：带选择性分类的多数投票变体 (ResNet VT + SC) - 消融对照散点
vt_sc_coverage = 92.57
vt_sc_risk     = 1.71

# (8) 【全新加入】对照组 6：带选择性分类的多数投票变体 (ResNet VT + SC) - 消融对照散点
vt_sc_coverage1 = 99.48
vt_sc_risk1     = 4.09

# =========================================================================
# 3. 画图核心逻辑
# =========================================================================
# 【调整】将图片大小从 (4.5, 3.0) 稍微增大到 (6.0, 4.5)，避免图例过多撑开爆框
fig, ax = plt.subplots(figsize=(4, 3), dpi=300) 

# 3.1 绘制各模型的 Risk-Coverage 动态曲线
# 创新集成方法：醒目的深红色粗实线突出，圆形标记
ax.plot(avg_coverage, avg_risk, 
        label='ResNet AVG(18,34,50)', 
        color='#D62728', linestyle='-', marker='o', markersize=4.5, linewidth=1, zorder=5)

# 对照方法曲线
ax.plot(r101_coverage, r101_risk, label='ResNet101', color='#D62728', linestyle='--', marker='s', markersize=4.5, linewidth=1, zorder=4)
ax.plot(r50_coverage, r50_risk, label='ResNet50', color='#1F77B4', linestyle='-.', marker='^', markersize=4.5, linewidth=1, zorder=3)
ax.plot(r34_coverage, r34_risk, label='ResNet34', color='#2CA02C', linestyle=':', marker='v', markersize=4.5, linewidth=1, zorder=2)
ax.plot(r18_coverage, r18_risk, label='ResNet18', color='#FF7F0E', linestyle='--', marker='d', markersize=4.5, linewidth=1, zorder=1)

# 3.2 绘制全新加入的多数投票基准点与消融对照散点
# 基准 1：传统多数投票（大粗叉号 X）
ax.scatter(traditional_vt_coverage, traditional_vt_risk, 
           label='ResNet VT(18,34,50)', 
           color='#9467BD', marker='X', s=30, linewidths=1.0, edgecolors='black', zorder=6)

# 基准 2：带选择性分类的多数投票（使用大号中空钻石 ◊ 或特制符号，这里用五边形 'p' 形成鲜明对比）
ax.scatter(vt_sc_coverage1, vt_sc_risk1, 
           label='ResNet VT_2(18,34,50)', 
           color='#8C564B', marker='X', s=30, linewidths=1, edgecolors='black', zorder=7)

# 基准 3：带选择性分类的多数投票（使用大号中空钻石 ◊ 或特制符号，这里用五边形 'p' 形成鲜明对比）
ax.scatter(vt_sc_coverage, vt_sc_risk, 
           label='ResNet VT_3(18,34,50)', 
           color='#7F7F7F', marker='X', s=30, linewidths=1, edgecolors='black', zorder=8)


# =========================================================================
# 4. 图表美化与高级细节雕刻（对标一区顶刊）
# =========================================================================
# 坐标轴标签 (使用 LaTeX 语法增强专业感)
ax.set_xlabel('Coverage (%)', fontsize=7, fontweight='bold', labelpad=8)
ax.set_ylabel('Risk (%)', fontsize=7, fontweight='bold', labelpad=8)

# 坐标轴显示区间优化（根据你的实际数据范围：X轴聚焦高覆盖率，Y轴聚焦0~6%的错误率）
ax.set_xlim(85.0, 101.5) 
ax.set_ylim(0.0, 6.0)

# 科学细网格线
ax.grid(True, linestyle=':', alpha=0.5, color='gray')

# 图例设计：置于左上角，取消花哨的效果，硬朗的学术风
# 为了适应更多的图例项，将 fontsize 稍微缩小到 9.0
ax.legend(loc='upper left', fontsize=7, frameon=True, edgecolor='black', fancybox=False, shadow=False)

# 移除上方和右方边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 刻度字号调大，显得扎实
ax.tick_params(axis='both', which='major', labelsize=7)

# 强行紧凑布局（防止小尺寸图片切掉标签）
plt.tight_layout()

# =========================================================================
# 5. 保存图像
# =========================================================================
plt.savefig('Fault_Risk_Coverage_Curve.png', dpi=300, bbox_inches='tight')
plt.savefig('Fault_Risk_Coverage_Curve.pdf', format='pdf', bbox_inches='tight')

print("SUCCESS: 包含您自定义名称数据的真实RC曲线图已生成完毕！")
plt.show()