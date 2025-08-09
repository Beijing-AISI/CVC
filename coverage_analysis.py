# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from text2vec import SentenceModel

print("读取数据中...")
human_df = pd.read_csv("human_remain.csv")
llm_df = pd.read_csv("llm_remain.csv")
print(f"人类规则数: {len(human_df)}, LLM规则数: {len(llm_df)}")


def select_text(row):
    return str(row['重写']) if row['refine'] == 1 else str(row['rot'])


human_texts = human_df.apply(select_text, axis=1).tolist()
llm_texts = llm_df.apply(select_text, axis=1).tolist()

print("加载中文 embedding 模型...")
model = SentenceModel("./text2vec-base-multilingual")

print("正在编码人类规则文本...")
human_embeds = model.encode(human_texts, normalize_embeddings=True)
print("人类编码完成 shape:", human_embeds.shape)

print("正在编码LLM规则文本...")
llm_embeds = model.encode(llm_texts, normalize_embeddings=True)
print("LLM编码完成 shape:", llm_embeds.shape)

# 构建 FAISS 索引
d = llm_embeds.shape[1]
print(f"正在构建 FAISS IndexFlatIP 索引 (维度: {d})...")
index = faiss.IndexFlatIP(d)
index.add(llm_embeds.astype("float32"))
print(f"索引构建完成，添加向量数: {index.ntotal}")

threshold = 0.80
matched_indices = set()

print("开始相似度搜索...")
batch_size = 512
total_batches = (len(human_embeds) + batch_size - 1) // batch_size

for batch_id in tqdm(range(0, len(human_embeds), batch_size), desc="匹配中"):
    batch = human_embeds[batch_id:batch_id + batch_size].astype("float32")
    D, I = index.search(batch, 100)

    for j in range(len(D)):
        for score, idx in zip(D[j], I[j]):
            if score > threshold:
                matched_indices.add(idx)

    if (batch_id // batch_size + 1) % 10 == 0:  # 每10个batch输出一次进度统计
        print(f"已处理 {batch_id + batch_size} / {len(human_embeds)} 条人类规则，当前匹配数: {len(matched_indices)}")

# 统计结果
C = len(matched_indices)
total = len(llm_embeds)
ratio = C / total

print("\n匹配完成！")
print(f"相似度阈值 > {threshold} 的 LLM 规则覆盖数：{C}")
print(f"占全部 LLM 规则比例：{ratio:.4%}")
