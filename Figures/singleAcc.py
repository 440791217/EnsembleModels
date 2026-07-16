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
    'Base (0)': (np.array([100.0, 95.71, 94.87, 93.73, 91.45]), np.array([5.5194, 3.4584, 3.1517, 2.8059, 2.1651])),
    'SC (1E-5)': (np.array([100.0000, 97.1667, 96.6333, 95.7667, 94.3333]), (np.array([36.3545, 35.7804, 35.7710, 35.5726, 35.5830]))),
    'SC (1E-3)': (np.array([100.00, 100.00, 100.00, 100.00, 100.00]), np.array([91.0696, 91.0696, 91.0696, 91.0696, 91.0696])),
    'SC (1E-1)': (np.array([100., 100., 100., 100., 100.]), np.array([90.0367, 90.0367, 90.0367, 90.0367, 90.0367])),
    'SC+ED (1E-5)': (np.array([64.0000, 61.1333, 60.6000, 59.7333, 58.3000]), np.array([6.3542, 4.0349, 3.7404, 2.9576, 2.1727])),
    'SC+ED (1E-3)': (np.array([0.1667, 0.1667, 0.1667, 0.1667, 0.1667]), np.array([40.0000, 40.0000, 40.0000, 40.0000, 40.0000])),
    'SC+ED (1E-1)': (np.array([0.1, 0.1, 0.1, 0.1, 0.1]), np.array([100., 100., 100., 100., 100.])),
    'SC+ED+DMR (1E-5)': (np.array([100.0000, 97.1343, 96.6011, 95.7348, 94.3019]), np.array([4.0653, 2.5386, 2.3456, 1.8448, 1.3428])),
}

COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#9467BD', '#1F77B4', '#2CA02C', '#9467BD']
LINE_STYLES = ['--', '--', '--', '--', '-', '-', '-'] 
MARKERS = ['o', 's', 's', 's', '^', '^', '^']  

PLOT_CONFIG = {
    'fig_size': (4, 3.5),          
    'xlim': (-2.0, 105.0), 
    'ylim': (0.08, 150.0),       
    'font_size_labels': 8,       
    'font_size_ticks': 7,        
    'font_size_legend': 6.5,  
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

# 4.3 Y 轴标签
y_ticks = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
ax.set_yticks(y_ticks)
y_labels = [r'$10^{-1}$', '1.0', '5.0', '10.0', '50.0', '100.0']
ax.get_yaxis().set_major_formatter(FixedFormatter(y_labels))

# 4.4 配置刻度字号与全网格
ax.tick_params(axis='both', which='both', labelsize=PLOT_CONFIG['font_size_ticks'])
ax.grid(True, which="both", linestyle=':', alpha=0.5, color='gray')

# =========================================================================
# 4.5 ✨【完美终极版：严格匹配 CURVE_MODELS 的行优先横铺算法】
# =========================================================================
raw_handles, raw_labels = ax.get_legend_handles_labels()
num_columns = 4  # 4 列横铺

# 计算行数
import math
num_items = len(raw_labels)
num_rows = math.ceil(num_items / num_columns)

# 初始化一个空矩阵用于位置映射
matrix = [[None for _ in range(num_columns)] for _ in range(num_rows)]

# 1. 把原始按 CURVE_MODELS 顺序读出的句柄，按“行优先（从左到右）”填入矩阵
idx = 0
for r in range(num_rows):
    for c in range(num_columns):
        if idx < num_items:
            matrix[r][c] = (raw_handles[idx], raw_labels[idx])
            idx += 1

# 2. 按 Matplotlib 的“列优先”读取方式，逆向重排句柄顺序
reordered_handles = []
reordered_labels = []
for c in range(num_columns):
    for r in range(num_rows):
        if matrix[r][c] is not None:
            handle, label = matrix[r][c]
            reordered_handles.append(handle)
            reordered_labels.append(label)

# 3. 将重排后的句柄送入 legend，此时图表上渲染出来的视觉顺序将完美等同于行优先
ax.legend(
    reordered_handles, reordered_labels,
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.18),  
    ncol=num_columns,              
    fontsize=PLOT_CONFIG['font_size_legend'], 
    frameon=True, 
    edgecolor='black', 
    fancybox=False, 
    shadow=False,
    labelspacing=0.2,
    columnspacing=0.5              # 缩短列距，名字更紧凑
)

# 4.6 移除上方和右方边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# =========================================================================
# 5. 保存
# =========================================================================
plt.savefig('YOLO_Fault_Risk_Coverage_RowMajor_Legend.png', dpi=300, bbox_inches='tight')
print("SUCCESS: 图例名字已修改为‘从左到右’横向紧靠排列！")
plt.show()