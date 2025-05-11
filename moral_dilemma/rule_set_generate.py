# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from threading import Thread

OUTPUT_FILE = "dataset/1_rule_set/rule_set.csv"
BATCH_SIZE = 1000


# def load_data():
#     """加载数据"""
#     in_path = "../../data_control/label_rot_data/human/人工最终保留数据.csv"
#     file_list = [os.path.join(in_path, f) for f in os.listdir(in_path) if f.endswith('.csv')]
#     temp_list = []
#
#     for file in file_list:
#         try:
#             try:
#                 df = pd.read_csv(file, encoding='gbk')
#             except UnicodeDecodeError:
#                 df = pd.read_csv(file, encoding='utf-8')
#
#             for _, row in df.iterrows():
#                 temp_list.append({
#                     'rot': row['rot'],
#                     'characters': row.get('characters', ''),
#                     'level': row.get('Level', ''),
#                     'core_values': row.get('Core Values', ''),
#                     'derived_values': row.get('Derived Values', ''),
#                     'action': row.get('action', '')
#                 })
#         except Exception as e:
#             print(f"加载 {file} 失败: {str(e)}")
#     return temp_list


def save_data(conflicting_pairs):
    df = pd.DataFrame(conflicting_pairs, columns=["rot1", "rot2", "translate1", "level1", "core_values_1",
                                                  "derived_values_1", "translate2", "level2", "core_values_2",
                                                  "derived_values_2"])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8", mode="a", header=False)


def save_data_async(conflicting_pairs):
    """异步存储数据"""
    thread = Thread(target=save_data, args=(conflicting_pairs,))
    thread.start()


def filter_similar_rules(rules):
    """
    通过 SentenceTransformer 计算规则之间的相似性
    :param rules: 规则文本列表
    :return: 规则文本的相似度矩阵
    """
    embeddings = sentence_model.encode(rules, convert_to_tensor=True)

    # [b, n] x [n, b] = [b, b]
    # x[i, j] = xxx
    # embeddings是张量，可以直接使用矩阵乘法计算两两之间的相似度，其效率大于1对1计算
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    # print(similarity_matrix)

    # similar_indices = torch.where(
    #     torch.triu(similarity_matrix, diagonal=1) >= similarity_threshold
    # )

    # for i, j in zip(similar_indices[0].tolist(), similar_indices[1].tolist()):
    #     similarity = similarity_matrix[i, j].item()
    #     similar_rule_pairs.append((rules[i], rules[j], similarity))

    # 最好不要使用for循环和index索引，因为效率低，耗时长
    # for i in range(len(rules)):
    #     for j in range(i + 1, len(rules)):
    #         similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
    #         if similarity >= similarity_threshold:
    #             similar_rule_pairs.append((rules[i], rules[j], similarity))

    return similarity_matrix


def is_conflict_batch(rules_batch1, rules_batch2, contradict_threshold, batch_texts1, batch_texts2):
    """
    判断两个规则是否存在冲突
    :param rules_batch1: 规则集合1
    :param rules_batch2: 规则集合2
    :param contradict_threshold: 矛盾概率阈值
    :return: 如果任一方向的矛盾概率 >= threshold，则返回 True
    """
    inputs = tokenizer(rules_batch1, rules_batch2, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[:, 0].tolist()

    # 反过来：以 rule2 为前提、rule1 为假设
    inputs_reversed = tokenizer(rules_batch2, rules_batch1, padding=True, truncation=True, return_tensors="pt").to(
        device)
    with torch.no_grad():
        outputs_reversed = model(**inputs_reversed)
    probs_reversed = torch.softmax(outputs_reversed.logits, dim=1)[:, 0].tolist()

    results = []
    for rule1, rule2, contradiction1, contradiction2 in zip(rules_batch1, rules_batch2, probs, probs_reversed):
        max_contradiction = max(contradiction1, contradiction2)

        if max_contradiction >= contradict_threshold:
            results.append((
                rule1,
                rule2,
                *batch_texts1[rule1],  # 展开列表
                *batch_texts2[rule2]
            ))
    return results


def find_conflicting_rules(rules, rule_text, similarity_threshold, contradict_threshold):
    similarity_matrix = filter_similar_rules(rules)

    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=["rot1", "rot2", "translate1", "level1", "core_values_1", "derived_values_1", "translate2",
                              "level2", "core_values_2", "derived_values_2"]).to_csv(OUTPUT_FILE, index=False)

    # 创建一个布尔掩码 (mask)，标记出 similarity_matrix 中哪些元素大于等于 similarity_threshold，但仅限于上三角部分（不包含主对角线）。
    mask = torch.triu(similarity_matrix >= similarity_threshold, diagonal=1)
    # 提取 mask 中所有 True 元素的索引，并以张量形式返回。
    i_indices, j_indices = torch.nonzero(mask, as_tuple=True)
    print(i_indices.size(0), j_indices.size(0))

    batch_rules1, batch_rules2 = [], []
    batch_texts1, batch_texts2 = {}, {}

    for i, j in tqdm(zip(i_indices.tolist(), j_indices.tolist()), desc="Finding conflicting rules",
                     total=i_indices.size(0)):
        rule1, rule2 = rules[i], rules[j]

        batch_rules1.append(rule1)
        batch_rules2.append(rule2)
        batch_texts1[rule1] = rule_text.iloc[i].tolist()
        batch_texts2[rule2] = rule_text.iloc[j].tolist()

        if len(batch_rules1) >= BATCH_SIZE:
            conflicting_pairs = is_conflict_batch(batch_rules1, batch_rules2, contradict_threshold, batch_texts1,
                                                  batch_texts2)
            # print(conflicting_pairs)
            save_data_async(conflicting_pairs)
            batch_rules1, batch_rules2 = [], []
            batch_texts1, batch_texts2 = {}, {}

    if batch_rules1:
        save_data_async(is_conflict_batch(batch_rules1, batch_rules2, contradict_threshold, batch_texts1, batch_texts2))


if __name__ == '__main__':
    # 读取数据
    in_file = "../../data_control/final.csv"
    try:
        data = pd.read_csv(in_file, encoding='gbk')
    except UnicodeDecodeError:
        data = pd.read_csv(in_file, encoding='utf-8')

    rule_list = data['translate'].tolist()
    rule_text = data[['rot', 'level', 'core values', 'derived values']]

    # 设置本地模型路径
    # 英文模型
    transformer_model_path = "./roberta-large-mnli"
    sentence_transformer_model_path = "./all-mpnet-base-v2"

    # 中文模型
    # transformer_model_path = "./chinese-roberta-wwm-ext-large"
    # sentence_transformer_model_path = "./text2vec-base-chinese"

    # 选择运行设备（自动检测是否有 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 RoBERTa MNLI 预训练模型
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_model_path).to(device)

    # 加载 SentenceTransformer
    sentence_model = SentenceTransformer(sentence_transformer_model_path, device=device)

    # 寻找冲突规则集
    find_conflicting_rules(rule_list, rule_text, similarity_threshold=0.5, contradict_threshold=0.8)

    print("冲突规则集生成完成——————————")
