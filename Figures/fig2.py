import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# 1. SCI 论文规范配置（字体、样式、抗锯齿）
# =========================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================================
# 2. 通用用户配置区（在此处替换你的数据、模型名称和图例配置）
# =========================================================================

# ---- 2.1 动态曲线数据配置 (支持任意数量的对比模型) ----
# 格式: { '模型或方案名称': (Coverage数组, Risk数组) }
CURVE_MODELS = {
    'ResNet AVG(18,34,50)+SC': (np.array([100.0, 92.19, 90.97, 89.78, 87.89]), np.array([4.22, 1.49, 1.23, 1.00, 0.77])),
    'ResNet101+SC': (np.array([100.0, 96.84, 96.23, 95.38, 93.53]), np.array([4.93, 3.36, 3.21, 2.88, 2.28])),
    'ResNet50+SC': (np.array([100.0, 96.52, 95.80, 94.83, 93.00]), np.array([5.18, 3.43, 3.15, 2.81, 2.29])),
    'ResNet34+SC': (np.array([100.0, 96.57, 95.78, 94.79, 93.21]), np.array([5.29, 3.54, 3.22, 2.87, 2.38])),
    'ResNet18+SC': (np.array([100.0, 95.70, 94.86, 93.72, 91.44]), np.array([5.51, 3.45, 3.14, 2.80, 2.15]))
}

CURVE_MODELS = {
    # 你的创新方法：对应表格第 1 段（低故障率，高鲁棒性，最低 Risk 达 0.69%）
    '1E-7': (np.array([100.0, 91.7, 90.53, 89.2, 87.3]), np.array([5.03, 1.64, 1.33, 1.01, 0.69])),
    # 对照组 1：对应表格第 2 段（中度故障环境，单体中型模型性能开始轻微震荡）
    '1E-6':           (np.array([100.0, 90.9, 89.73, 88.4, 86.5]), np.array([8.63, 1.65, 1.34, 1.02, 0.69])),
    # 对照组 2：对应表格第 3 段（高故障环境，无防御下基础 Risk 暴增至 35.27%，但 SC 拦截后降到 0.77%）
    '1E-5':           (np.array([100.0, 81.8, 80.73, 79.63, 77.93]), np.array([35.27, 1.71, 1.4, 1.17, 0.77])),
    # 对照组 3：对应表格第 4 段（极端恶劣故障，无防御 Risk 逼近 89.23% 彻底失效，SC 机制牺牲大量 Coverage 强行止损）
    '1E-4':           (np.array([100.0, 26.9, 26.47, 26.1, 25.5]), np.array([89.23, 1.98, 1.64, 1.15, 0.92]))
}



# 动态曲线的样式配置池（自动循环应用，保证对比组样式统一且精细）
# 颜色：深红(突出创新点), 折线蓝, 森林绿, 活力橙, 优雅紫
COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD']
LINE_STYLES = ['-', '--', '-.', ':', '--']
MARKERS = ['o', 's', '^', 'v', 'd']

# ---- 2.2 离散基准点数据配置 (如传统的消融对照散点，若无则留空 {} ) ----
# 格式: { '基准点名称': { 'x': Coverage值, 'y': Risk值, 'marker': 符号, 'color': 颜色 } }
if 0:
        SCATTER_POINTS = {
        'ResNet VT(18,34,50)': {'x': 100.0, 'y': 4.40, 'marker': 'X', 'color': '#7F7F7F'},
        'ResNet VT(18,34,50)+Reject':     {'x': 92.57, 'y': 1.71, 'marker': 'p', 'color': '#FFFF00'}
        }
else:
     SCATTER_POINTS={}

# ---- 2.3 图表物理边界与字号控制 ----
PLOT_CONFIG = {
    'fig_size': (4, 3),          # 单栏排版微型黄金比例
    'xlim': (0, 101.5),       # X轴范围
    'ylim': (0.0, 101.0),          # Y轴范围
    'font_size_labels': 8,       # 轴标签字号
    'font_size_ticks': 7,        # 刻度数字字号
    'font_size_legend': 7,       # 图例字号
    'legend_loc': 'upper left'   # 图例位置
}

# =========================================================================
# 3. 自动化绘图引擎（逻辑完全解耦，无需修改）
# =========================================================================
fig, ax = plt.subplots(figsize=PLOT_CONFIG['fig_size'], dpi=300)

# 3.1 渲染动态曲线
for idx, (model_name, (cov, risk)) in enumerate(CURVE_MODELS.items()):
    # 首个模型默认为您的创新方法，赋予更高的zorder、加粗线宽和独立标记
    is_proposed = (idx == 0)
    
    ax.plot(
        cov, risk,
        label=model_name,
        color=COLORS[idx % len(COLORS)],
        linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
        marker=MARKERS[idx % len(MARKERS)],
        linewidth=2.0 if is_proposed else 1.0,
        markersize=5.0 if is_proposed else 3.5,
        zorder=10 if is_proposed else (5 - idx)
    )

# 3.2 渲染离散基准点
for point_name, props in SCATTER_POINTS.items():
    ax.scatter(
        props['x'], props['y'],
        label=point_name,
        color=props['color'],
        marker=props['marker'],
        s=45,
        linewidths=1.0,
        edgecolors='black',
        zorder=12
    )

# =========================================================================
# 4. 视觉规范细节打磨
# =========================================================================
# 坐标轴名与显示范围
ax.set_xlabel('Coverage (%)', fontsize=PLOT_CONFIG['font_size_labels'], fontweight='bold', labelpad=6)
ax.set_ylabel('Risk (%)', fontsize=PLOT_CONFIG['font_size_labels'], fontweight='bold', labelpad=6)
ax.set_xlim(PLOT_CONFIG['xlim'])
ax.set_ylim(PLOT_CONFIG['ylim'])

# 刻度与网格线
ax.tick_params(axis='both', which='major', labelsize=7)
ax.grid(True, linestyle=':', alpha=0.5, color='gray')

# 硬朗学术风图例
ax.legend(
    loc=PLOT_CONFIG['legend_loc'], 
    fontsize=PLOT_CONFIG['font_size_legend'], 
    frameon=True, 
    edgecolor='black', 
    fancybox=False, 
    shadow=False
)

# 移除多余边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 紧凑型输出防止内容截断
plt.tight_layout()

# =========================================================================
# 5. 图像多格式持久化保存
# =========================================================================
plt.savefig('Academic_Risk_Coverage_Curve.png', dpi=300, bbox_inches='tight')
plt.savefig('Academic_Risk_Coverage_Curve.pdf', format='pdf', bbox_inches='tight')

print("SUCCESS: 顶刊规范通用RC折线图已成功生成！")
plt.show()