import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 移除中文字体设置，使用系统默认字体
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# 定义模型列表（文件名与显示名称的映射）
models = [
    # 中国模型
    {'file': 'DeepSeek-R1.csv', 'name': 'DeepSeek-R1'},
    {'file': 'DeepSeek-V3.csv', 'name': 'DeepSeek-V3'},
    {'file': 'Doubao-1.5-pro-32k.csv', 'name': 'Doubao-1.5-32k'},
    {'file': 'Doubao-1.5-pro-256k.csv', 'name': 'Doubao-1.5-256k'},
    {'file': 'Qwen2.5-7B-Instruct.csv', 'name': 'Qwen2.5-7B'},
    {'file': 'Qwen2.5-32B-Instruct.csv', 'name': 'Qwen2.5-32B'},
    {'file': 'Qwen2.5-72B-Instruct.csv', 'name': 'Qwen2.5-72B'},
    {'file': 'yi-34b-chat-0205.csv', 'name': 'yi-34b-chat'},
    {'file': 'GLM-4-32B-0414.csv', 'name': 'GLM-4-32B'},
    # 美国模型
    {'file': 'gpt-4o.csv', 'name': 'GPT-4o'},
    {'file': 'o1.csv', 'name': 'O1'},
    {'file': 'gpt-3.5-turbo-1106.csv', 'name': 'GPT-3.5-Turbo'},
    {'file': 'claude-3-7-sonnet-20250219.csv', 'name': 'Claude-3-Sonnet'},
    {'file': 'gemini-1.5-pro.csv', 'name': 'Gemini-1.5-Pro'},
    # 欧洲模型
    {'file': 'aihubmix-Llama-3-1-70B-Instruct.csv', 'name': 'Llama-3-70B'},
    {'file': 'aihubmix-Llama-3-1-405B-Instruct.csv', 'name': 'Llama-3-405B'},
    {'file': 'codestral-latest.csv', 'name': 'Codestral'},

]

# 强制使用非交互式后端
matplotlib.use('Agg')

# 读取所有模型的答案数据
model_answers = []

for model in models:
    try:
        df = pd.read_csv(f'result/llm_result/{model["file"]}', encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(f'result/llm_result/{model["file"]}', encoding="gbk")

    answers = df['result']
    model_answers.append(answers)

# 确保所有模型的数据长度一致
assert len(set(len(a) for a in model_answers)) == 1, "模型数据长度不一致"

# 计算相似度矩阵
n = len(models)
matrix = np.eye(n)  # 初始化单位矩阵

for i in range(n):
    for j in range(i + 1, n):
        matches = sum(a == b for a, b in zip(model_answers[i], model_answers[j]))
        similarity = matches / len(model_answers[i])
        matrix[i][j] = matrix[j][i] = similarity

# 可视化设置
plt.figure(figsize=(10, 8))
sns.set(style="white", font_scale=1.2)

# 使用mask隐藏对角线
mask = np.eye(n, dtype=bool)
np.fill_diagonal(mask, False)

ax = sns.heatmap(
    matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    linewidths=0.5,
    square=True,
    annot_kws={"size": 8},
    xticklabels=[m['name'] for m in models],
    yticklabels=[m['name'] for m in models],
    cbar_kws={"shrink": 0.8}
)

# 调整标签位置
ax.set_xticks(np.arange(n) + 0.5)
ax.set_yticks(np.arange(n) + 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# ax.set_title('Model response similarity matrix', pad=20)

plt.tight_layout()
plt.savefig('result/picture/similarity.png', dpi=300, bbox_inches='tight')
print("可视化结果已保存为 result/picture/similarity.png")
