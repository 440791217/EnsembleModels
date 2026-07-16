import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter

# =========================================================================
# 1. SCI 论文规范配置（字体、样式、抗锯齿）
# =========================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================================
# 2. 核心用户配置区
# =========================================================================
CURVE_MODELS = {
    'SC (0)': (np.array([100.0, 92.19, 90.97, 89.78, 87.89]), np.array([4.22, 1.49, 1.23, 1.00, 0.77])),
    'SC (1E-6)': (np.array([100.0, 88.1, 86.93, 85.7, 83.83]), np.array([8.63, 1.7, 1.38, 1.05, 0.72])),
    'SC (1E-5)': (np.array([100.0, 58.87, 58.0, 57.2, 55.9]), np.array([35.27, 1.81, 1.38, 1.05, 0.72])),
    'SC (1E-1)': (np.array([100.0, 0, 0, 0, 0]), np.array([90.03, 0.1, 0.1, 0.1, 0.1])),
    'SC+ED (1E-6)': (np.array([100.0, 92.07, 90.87, 89.47, 87.63]), np.array([4.8, 1.63, 1.32, 0.93, 0.68])),
    'SC+ED (1E-5)': (np.array([100.0, 92.1, 91.0, 89.83, 87.97]), np.array([4.97, 1.56, 1.32, 1.0, 0.8])),
    'SC+ED (1E-1)': (np.array([100.0, 85.23, 83.57, 81.57, 77.73]), np.array([5.17, 1.53, 1.32, 1.06, 0.77])),
}

COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#9467BD', '#1F77B4', '#2CA02C', '#9467BD']
# 补全线型数组，使其长度与曲线数量对齐，防止潜在越界
LINE_STYLES = ['--', '--', '--', '--', '-', '-', '-'] 
MARKERS = ['o', 's', 's', 's', '^', '^', '^']  

PLOT_CONFIG = {
    'fig_size': (4, 3),          
    'xlim': (-2.0, 105.0),       
    'ylim': (0.08, 150.0),       # <-- 【核心修改】放宽至 0.08，确保 0.1 标签不被边缘切割       
    'font_size_labels': 8,       
    'font_size_ticks': 7,        
    'font_size_legend': 6.5,  
    'legend_loc': 'upper left'   
}

# =========================================================================
# 3. 自动化绘图引擎
# =========================================================================
fig, ax = plt.subplots(figsize=PLOT_CONFIG['fig_size'], dpi=200)

for idx, (model_name, (cov, risk)) in enumerate(CURVE_MODELS.items()):
    ax.plot(
        cov, risk,
        label=model_name,
        color=COLORS[idx % len(COLORS)],
        linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
        marker=MARKERS[idx % len(MARKERS)],
        linewidth=0.5,
        markersize=2.5,
        zorder=10 - idx  
    )

# =========================================================================
# 4. 视觉规范细节打磨
# =========================================================================
ax.set_xlabel('Coverage (%)', fontsize=PLOT_CONFIG['font_size_labels'], fontweight='bold', labelpad=3)
ax.set_ylabel('Risk (%)', fontsize=PLOT_CONFIG['font_size_labels'], fontweight='bold', labelpad=3)

# 4.1 线性 X 轴刻度
ax.set_xlim(PLOT_CONFIG['xlim'])
ax.set_xticks([0, 20, 40, 60, 80, 100])  

# 4.2 Y 轴对数坐标
ax.set_yscale('log')
ax.set_ylim(PLOT_CONFIG['ylim'])

# 4.3 ✨【核心修改：将 Y 轴最小值调整为 0.1】
y_ticks = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
ax.set_yticks(y_ticks)

# 0.1 使用 LaTeX 渲染为标准的 10^-1 科学计数法
y_labels = [r'$10^{-1}$', '1.0', '5.0', '10.0', '50.0', '100.0']
ax.get_yaxis().set_major_formatter(FixedFormatter(y_labels))

# 4.4 配置刻度字号与全网格
ax.tick_params(axis='both', which='both', labelsize=PLOT_CONFIG['font_size_ticks'])
ax.grid(True, which="both", linestyle=':', alpha=0.5, color='gray')

# 4.5 硬朗学术风图例
ax.legend(
    loc=PLOT_CONFIG['legend_loc'], 
    fontsize=PLOT_CONFIG['font_size_legend'], 
    frameon=True, 
    edgecolor='black', 
    fancybox=False, 
    shadow=False,
    labelspacing=0.2  
)

# 4.6 移除上方和右方边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# =========================================================================
# 5. 保存
# =========================================================================
plt.savefig('YOLO_Fault_Risk_Coverage_Min0.1.png', dpi=300, bbox_inches='tight')
print("SUCCESS: Y 轴下限已调整为 0.1 ($10^{-1}$)，图表生成成功！")
plt.show()