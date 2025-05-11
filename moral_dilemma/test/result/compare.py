# -*- coding: utf-8 -*-
import pandas as pd

models = [
    {'file': 'aihubmix-Llama-3-1-70B-Instruct.csv', 'name': 'Llama-3-70B'},
    {'file': 'aihubmix-Llama-3-1-405B-Instruct.csv', 'name': 'Llama-3-405B'},
    {'file': 'claude-3-7-sonnet-20250219.csv', 'name': 'Claude-3-Sonnet'},
    {'file': 'codestral-latest.csv', 'name': 'Codestral'},
    {'file': 'DeepSeek-R1.csv', 'name': 'DeepSeek-R1'},
    {'file': 'DeepSeek-V3.csv', 'name': 'DeepSeek-V3'},
    {'file': 'Doubao-1.5-pro-32k.csv', 'name': 'Doubao-1.5-32k'},
    {'file': 'Doubao-1.5-pro-256k.csv', 'name': 'Doubao-1.5-256k'},
    {'file': 'gemini-1.5-pro.csv', 'name': 'Gemini-1.5-Pro'},
    {'file': 'GLM-4-32B-0414.csv', 'name': 'GLM-4-32B'},
    {'file': 'gpt-3.5-turbo-1106.csv', 'name': 'GPT-3.5-Turbo'},
    {'file': 'gpt-4o.csv', 'name': 'GPT-4o'},
    {'file': 'o1.csv', 'name': 'O1'},
    {'file': 'Qwen2.5-7B-Instruct.csv', 'name': 'Qwen2.5-7B'},
    {'file': 'Qwen2.5-32B-Instruct.csv', 'name': 'Qwen2.5-32B'},
    {'file': 'Qwen2.5-72B-Instruct.csv', 'name': 'Qwen2.5-72B'},
    {'file': 'yi-34b-chat-0205.csv', 'name': 'yi-34b-chat'}
]

# 读取所有模型答案
model_answers = []
model_names = []
dataframes = []

for model in models:
    try:
        df = pd.read_csv(f'llm_result/{model["file"]}', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(f'llm_result/{model["file"]}', encoding='gbk')
    model_answers.append(df['result'].tolist())
    model_names.append(model['name'])
    dataframes.append(df)

# 构建模型回答矩阵
df_all_answers = pd.DataFrame(model_answers).T
df_all_answers.columns = model_names  # 列名为模型名

# 用第一个模型的完整DataFrame作为基础
base_df = dataframes[0].copy()

# 删除 base_df 中的 result 列
if 'result' in base_df.columns:
    base_df.drop(columns=['result'], inplace=True)

# 拼接模型回答
df_combined = pd.concat([base_df.reset_index(drop=True), df_all_answers.reset_index(drop=True)], axis=1)


# 筛选出模型回答不一致的问题
def is_inconsistent(row):
    answers = [row[m] for m in model_names]
    return len(set(answers)) > 1


df_inconsistent = df_combined[df_combined.apply(is_inconsistent, axis=1)]

# 保存结果
output_path = "inconsistent_responses.csv"
df_inconsistent.to_csv(output_path, index=False, encoding="utf-8")
print(f"已保存带原始信息的不一致回答到：{output_path}")
