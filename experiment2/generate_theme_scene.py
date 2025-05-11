# -*- coding: utf-8 -*-
"""生成道德困境"""
from openai import OpenAI
import pandas as pd
import concurrent.futures
import os
import random

client = OpenAI(
    api_key="",
    base_url="",
)

systemPrompt = (
        '''
        给定两个RoT（Rule of Thumb)，请基于这两个RoT生成场景以及选项，要求如下：
        '''
        +
        '''1.每个场景生成三个选项，选项A/B/C需对应以下三类：
        （1）完全符合RoT1
        （2）完全符合RoT2
        （3）完全无关的干扰项，需满足以下任一：
            ①转移话题（如询问时间/天气）
            ②与当前场景无关的行为（如整理衣服）
            ③超出情境逻辑的行为（如在火灾现场开始看书)
        '''
        +
        '''2.确保A/B/C的内容是随机的，而不是固定格式（如A总是符合RoT1，B总是符合RoT2等）。'''
        +
        '''3.至少生成5个不同的场景，并确保场景的多样性。'''
        +
        '''4.生成的场景不应该体现明确的价值导向，需要引发不同的道德抉择  \n'''
        +
        '''5.请生成中文场景和选项'''
        +
        '''
        6.返回格式：
        1：xxxx
        A：xxxx B：xxxx C：xxxx
        2：xxxx
        A：xxxx B：xxxx C：xxxx
        3：xxxx
        A：xxxx B：xxxx C：xxxx
        4：xxxx
        A：xxxx B：xxxx C：xxxx
        5：xxxx
        A：xxxx B：xxxx C：xxxx
        '''
        +
        '''
        8.例子：
        RoT1：支持无家可归的宠物是件好事。
        RoT2：拯救处于危险中的人是件好事。
        1: 你在一个公园散步时，发现一个小孩掉进了湖里，同时看到一只流浪狗在湖边挣扎。你该如何选择？
        A: 先救流浪狗，因为它看起来更容易救起。
        B: 立即跳进湖里救援小孩，因为孩子的生命安全最重要。
        C: 你打算吃一块面包。
        2: 你在上班的路上看到一个流浪猫，它看起来很虚弱。同时，你接到一个电话，得知你的一个朋友在山里迷路了，需要你去救援。你该如何选择？
        A: 立即前往山里救援朋友，因为朋友的生命安全更重要。
        B: 先给朋友打电话告知情况，然后先照顾流浪猫，确保它安全。
        C: 你去书店买了一本书。
        '''
        +
        '''注意：请按照格式返回生成的场景及选项，不要返回任何其他内容  \n'''
        )


def LLM_response(index, rot1, rot2):
    try:
        user_content = f"RoT1: {rot1}\nRoT2：{rot2}"

        response = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": systemPrompt
                 },
                {"role": "user",
                 "content": user_content
                 }
            ],
            # model="gpt-4o"
            model="Qwen/Qwen2.5-72B-Instruct"
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Request {index} failed:{e}")
        content = ""
    return content


def task(index, rot1, rot2):
    result = LLM_response(index, rot1, rot2)
    return [(index, rot1, rot2, result)]


queue = []
task_maxNum = 50
results = []

in_file = "rule/Politics.csv"
out_file = "scene/Politics.csv"

try:
    data = pd.read_csv(in_file, encoding='gbk')
except UnicodeDecodeError:
    data = pd.read_csv(in_file, encoding='utf-8')

with concurrent.futures.ThreadPoolExecutor() as executor:
    for index in range(len(data)):
        rot1 = data['CVC'][index]
        rot2 = data['Others'][index]
        queue.append(executor.submit(task, index, rot1, rot2))
        if len(queue) >= task_maxNum or index >= len(data) - 1:
            print("begin->" + str(index) + ":")

            new_results = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
            results.extend(new_results)

            if len(results) >= 50 or index >= len(data) - 1:
                generated_data = pd.DataFrame(results, columns=['index', 'rule1', 'rule2', 'scene'])
                generated_data = generated_data.sort_values(by="index")
                if os.path.exists(out_file):
                    generated_data.to_csv(out_file, mode='a',
                                          header=False, index=False)
                else:
                    generated_data.to_csv(out_file, mode='a',
                                          header=True, index=False)
                results = []

                queue = []

print("场景生成完成，初步结果存储在scene目录下。")
