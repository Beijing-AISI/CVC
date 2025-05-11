# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('TkAgg')  # 修复后端问题
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置图形风格为白色背景 + 淡色网格
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'axes.facecolor': '#FFFFFF',        # 改为纯白
    'figure.facecolor': '#FFFFFF',      # 改为纯白
    'axes.edgecolor': '#DDDDDD',
    'grid.color': '#EEEEEE',
    'axes.labelcolor': '#555555',
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'text.color': '#333333'
})

# 模型文件映射
model_mapping = {
    'DeepSeek-R1.csv': 'DeepSeek-R1',
    'DeepSeek-V3.csv': 'DeepSeek-V3',
    'Doubao-1.5-pro-32k.csv': 'Doubao-1.5-32k',
    'Doubao-1.5-pro-256k.csv': 'Doubao-1.5-256k',
    'Qwen2.5-7B-Instruct.csv': 'Qwen2.5-7B',
    'Qwen2.5-32B-Instruct.csv': 'Qwen2.5-32B',
    'Qwen2.5-72B-Instruct.csv': 'Qwen2.5-72B',
    'yi-34b-chat-0205.csv': 'yi-34b-chat',
    'GLM-4-32B-0414.csv': 'GLM-4-32B',
    'gpt-4o.csv': 'GPT-4o',
    'o1.csv': 'O1',
    'gpt-3.5-turbo-1106.csv': 'GPT-3.5-Turbo',
    'claude-3-7-sonnet-20250219.csv': 'Claude-3-Sonnet',
    'gemini-1.5-pro.csv': 'Gemini-1.5-Pro',
    'aihubmix-Llama-3-1-70B-Instruct.csv': 'Llama-3-70B',
    'aihubmix-Llama-3-1-405B-Instruct.csv': 'Llama-3-405B',
    'codestral-latest.csv': 'Codestral'
}

# 统计每个选项的比例
result_stats = {'A': [], 'B': [], 'C': []}
for file_name, model_name in model_mapping.items():
    try:
        df = pd.read_csv(f'result/llm_result/{file_name}')
        if 'result' not in df.columns:
            print(f"Warning: '{file_name}' 中不包含 'result' 列，跳过。")
            continue
        for option in ['A', 'B', 'C']:
            result_stats[option].append((df['result'] == option).mean())
    except Exception as e:
        print(f"读取文件 '{file_name}' 时出错：{e}")

# 绘图准备
x = np.arange(len(model_mapping))
bar_width = 0.8
model_labels = list(model_mapping.values())

# 淡色调色板
blue_palette = ['#6caed6', '#c8dcf0', '#F6F6F6']

# 创建图像
plt.figure(figsize=(14, 7), dpi=100)

# 堆叠柱状图
bar_a = plt.bar(x, result_stats['A'], width=bar_width,
                color=blue_palette[0], edgecolor='#CCCCCC', linewidth=0.8, label='Option A')

bottom_b = np.array(result_stats['A'])
bar_b = plt.bar(x, result_stats['B'], width=bar_width, bottom=bottom_b,
                color=blue_palette[1], edgecolor='#CCCCCC', linewidth=0.8, label='Option B')

bottom_c = bottom_b + np.array(result_stats['B'])
bar_c = plt.bar(x, result_stats['C'], width=bar_width, bottom=bottom_c,
                color=blue_palette[2], edgecolor='#CCCCCC', linewidth=0.8, label='Option C')

# 添加百分比标签
for i in x:
    plt.text(i, result_stats['A'][i] / 2, f"{result_stats['A'][i]:.1%}",
             ha='center', va='center', color='#333333', fontsize=9)

    plt.text(i, bottom_b[i] + result_stats['B'][i] / 2, f"{result_stats['B'][i]:.1%}",
             ha='center', va='center', color='#333333', fontsize=9)

    plt.text(i, bottom_c[i] + result_stats['C'][i] / 2, f"{result_stats['C'][i]:.1%}",
             ha='center', va='center', color='#333333', fontsize=9)

# 设置横轴刻度倾斜显示
plt.xticks(x, model_labels, rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Proportion', fontsize=12, fontweight='semibold', labelpad=10)

# 图例
plt.legend(loc='lower right', frameon=True, framealpha=0.3, edgecolor='#AAAAAA', fontsize=10, bbox_to_anchor=(1, -0.2))

# 布局与保存
plt.tight_layout()
output_filename = 'result/picture/option_preference.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
print(f"图表已保存为 {output_filename}")

# 显示图表
try:
    plt.show()
except Exception as e:
    print(f"显示图表时出错：{e}")
    print("图表已成功保存，但显示时出现问题。")
