# -*- coding: utf-8 -*-
"""有无规则指导下的场景生成"""
from openai import OpenAI
import pandas as pd
import concurrent.futures
import os
import re

client = OpenAI(
    api_key="",
    base_url="",
)


in_file = "rules.csv"
try:
    data = pd.read_csv(in_file, encoding='gbk')
except UnicodeDecodeError:
    data = pd.read_csv(in_file, encoding='utf-8')

with_rule_systemPrompt = (
        '''给定1个主题和5个规则，请基于这5个规则生成场景，要求如下：'''
        +
        '''1.每个规则需要生成20个不同的场景，并确保场景的多样性。'''
        +
        '''2.生成的场景需要围绕主题。'''
        +
        '''3.请生成中文场景。'''
        +
        '''
        4.返回格式：
        1：xxxx
        2：xxxx
        3：xxxx
        4：xxxx
        5：xxxx
        ... ...
        100：xxxx
        '''
        +
        '''
        5.例子
        主题：友善
        规则：
        1. 帮助他人是正确的。  
        2. 照顾身边的家人是好的。
        ... ...
        回答：
        1: 你看到一位老人摔倒在街上。  
        2: 同事请求你帮助完成紧急报表。  
        3：你的同学向你请教一道数学题。
        ... ...
        '''
        +
        '''注意：请按照格式返回生成的场景，不要返回任何其他内容。'''
)

without_rule_systemPrompt = (
        '''给定1个主题，请基于这个主题生成场景，要求如下：'''
        +
        '''1.每个主题需要生成100个不同的场景，并确保场景的多样性。'''
        +
        '''2.生成的场景需要围绕主题。'''
        +
        '''3.请生成中文场景。'''
        +
        '''
        4.返回格式：
        1：xxxx
        2：xxxx
        3：xxxx
        4：xxxx
        5：xxxx
        ... ...
        100：xxxx
        '''
        +
        '''
        5.例子
        主题：友善
        回答：
        1: 你看到一位老人摔倒在街上。  
        2: 同事请求你帮助完成紧急报表。  
        3：你的同学向你请教一道数学题。
        ... ...
        '''
        +
        '''注意：请按照格式返回生成的场景，不要返回任何其他内容。'''
)


def LLM_response(index, systemPrompt, userPrompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": systemPrompt
                 },
                {"role": "user",
                 "content": userPrompt
                 }
            ],
            model="Qwen/Qwen2.5-72B-Instruct"
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Request {index} failed:{e}")
        content = ""
    return content


def task(index, theme, rules, systemPrompt, userPrompt):
    result = LLM_response(index, systemPrompt, userPrompt)
    return [(index, theme, rules, result)]


def save_data(results, out_file):
    generated_data = pd.DataFrame(results, columns=['index', 'theme', 'rules', 'scene'])
    generated_data = generated_data.sort_values(by="index")
    if os.path.exists(out_file):
        generated_data.to_csv(out_file, mode='a',
                              header=False, index=False)
    else:
        generated_data.to_csv(out_file, mode='a',
                              header=True, index=False)


def with_rule():
    """以规则为指导生成场景"""
    queue = []
    task_maxNum = 2
    results = []

    out_file = "100/with_rule_scene_100.csv"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for index, theme in enumerate(data.columns):
            rules = data[theme].dropna().tolist()
            rule_list = "\n".join([f"{i + 1}. {r}" for i, r in enumerate(rules)])
            user_content = f"主题: {theme}\n规则：{rule_list}"
            queue.append(executor.submit(task, index, theme, rules, with_rule_systemPrompt, user_content))
            if len(queue) >= task_maxNum or index >= 11:
                print("begin->" + str(index) + ":")

                new_result = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
                results.extend(new_result)

                if len(results) >= task_maxNum or index >= 11:
                    save_data(results, out_file)
                    results = []

                queue = []

    print(f"基于规则生成场景完成，初步结果存储在{out_file}下。")


def without_rule():
    """没有规则指导直接生成场景"""
    out_file = "100/without_rule_scene_100.csv"
    results = []
    for index, theme in enumerate(data.columns):
        user_content = f"主题: {theme}"
        result = LLM_response(index, without_rule_systemPrompt, user_content)
        result = [(index, theme, "", result)]
        results.extend(result)
        print(f"{theme}主题生成完成。")
        save_data(results, out_file)
        results = []
    print(f"直接生成场景完成，初步结果存储在{out_file}下。")


def process_data(file):
    """格式化数据"""
    try:
        df = pd.read_csv(file, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='utf-8')
    expanded_rows = []

    for _, row in df.iterrows():
        index, theme, rules, scenes_blob = row['index'], row['theme'], row['rules'], row['scene']

        parts = re.split(r'\n?\d+:\s+', scenes_blob.strip())[1:]

        for scene_text in parts:
            scene_text = scene_text.strip()
            if scene_text:
                expanded_rows.append({
                    'index': index,
                    'theme': theme,
                    'rules': rules,
                    'scene': scene_text
                })

    new_df = pd.DataFrame(expanded_rows)

    new_df.to_csv(file, index=False, encoding='utf-8')


if __name__ == '__main__':
    with_rule()
    without_rule()
    # process_data("100/with_rule_scene_100.csv")
    # process_data("100/without_rule_scene_100.csv")
