# -*- coding: utf-8 -*-
from openai import OpenAI
import pandas as pd
import os
import concurrent.futures


client = OpenAI(
    api_key="",
    base_url="",
)


def LLM_response(scene, rot):
    systemPrompt = (
        '''给定一个场景和其对应的拇指规则（RoT），请完成以下任务'''
        +
        '''
        1.RoT要求
        （1）RoT应该尽量简短，不能包含过多的场景信息
        （2）RoT只应该包含一个（行为，判断）对。
        '''
        +
        '''
        2.任务
        （1）如果RoT满足上述要求，则不做任何处理直接返回该RoT。
        （2）如果RoT不满足，请修改RoT，并返回修改后的RoT。
        '''
        +
        '''
        3.输出格式：
        xxx
        '''
        +
        '''
        4.示例：
        场景_1:由于连降大雨，一条线路的人字杆由于基础下陷发生倾倒，他带领全班人员，迅速投入到抢修工作中。天渐渐黑了下来，抢修工作仍在继续，直到第二天清晨人字杆立起来了，导线通了。王孝生一夜没合眼，圆满完成了抢修任务。
        RoT_1:在面临突发紧急状况时，人们应该不辞辛劳地进行抢修工作，以保障公共设施的正常运行。
        回答:突发紧急状况时，应全力抢修确保公共设施正常运行。
        场景_2:肇事司机邓被抓获时酒气熏天，民警随即对邓进行酒精呼气检测，当时测得指标为242mg/100ml，涉嫌醉酒驾驶，随后对邓进行抽血检验，检验结果为225.9mg/100ml。
        RoT_2:醉酒驾驶是违反法律和危害公共安全的行为，应该遵守交通法规以确保自身和他人的安全。
        回答:醉酒驾驶是违反法律和危害公共安全的行为。
        场景_3:无偿捐献器官是梁中建的妻子和5个哥哥1个姐姐在8月18日商量决定的。
        RoT_3:在亲人去世后，无偿捐献器官是一种高尚的行为，体现了对生命的尊重和对他人的关爱。
        回答:无偿捐献器官是一种高尚的行为。
        '''
        +
        '''
        注意：只能返回RoT，不能返回任何其他内容。
        '''
    )
    user_content = f"explicit_scene_result: {scene}\nRoT: {rot}"
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": systemPrompt
                 },
                {"role": "user",
                 "content": user_content
                 }
            ],
            model="Qwen/Qwen2.5-72B-Instruct"
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Request {index} failed:{e}")
        content = ""
    return content


def task(index, scene, rot):
    result = LLM_response(scene, rot)
    return [(index, scene, result)]


queue = []
task_maxNum = 100
results = []

in_path = "2_formatted"
out_path = "3_processed"
file_list = [f for f in os.listdir(in_path) if f.endswith('.csv')]

for file_name in file_list:
    input_file_path = os.path.join(in_path, file_name)
    try:
        data = pd.read_csv(input_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(input_file_path, encoding='gbk')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for index in range(len(data)):
            scene = data["explicit_scene_result"][index]
            rot = data["rot"][index]
            queue.append(executor.submit(task, index, scene, rot))
            if len(queue) >= task_maxNum or index >= len(data)-1:
                print("begin->" + str(index) + ":")

                new_results = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
                results.extend(new_results)

                if (index - 100 + 1) % 100 == 0 or index >= len(data)-1:
                    generated_data = pd.DataFrame(results, columns=['序号', 'explicit_scene_result', 'rot'])
                    generated_data = generated_data.sort_values(by="序号")
                    output_file_path = os.path.join(out_path, file_name)
                    if os.path.exists(output_file_path):
                        generated_data.to_csv(output_file_path, mode='a',
                                              header=False, index=False)
                    else:
                        generated_data.to_csv(output_file_path, mode='a',
                                              header=True, index=False)
                    results = []

                queue = []

print("处理完成！")
