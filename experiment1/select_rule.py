# -*- coding: utf-8 -*-
"""从12类核心价值中随机抽取5条规则"""
import pandas as pd
import os
import glob

# 运行该文件前，需要先运行data_control中的value_classify.py文件，得到最后的核心价值分类文件夹
data_dir = '../data_control/核心价值'

value_names = [
    '富强', '民主', '文明', '和谐',
    '自由', '平等', '公正', '法治',
    '爱国', '敬业', '诚信', '友善'
]

sampled_cols = []

seed = 123

for val in value_names:
    pattern = os.path.join(data_dir, f'*{val}*.csv')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"找不到匹配 '{pattern}' 的文件，请检查命名。")
    file_path = matches[0]

    try:
        df = pd.read_csv(file_path, usecols=['rot'], encoding="gbk")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, usecols=['rot'], encoding="utf-8")

    series = df['rot'].sample(n=5, random_state=seed).reset_index(drop=True)
    series.name = val
    sampled_cols.append(series)

result_df = pd.concat(sampled_cols, axis=1)

out_file = 'rules.csv'
result_df.to_csv(out_file, index=False, encoding='utf-8')

print(f"抽样结果已保存到: {out_file}")
