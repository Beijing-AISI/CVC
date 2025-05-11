from openai import OpenAI
import pandas as pd
import os
import concurrent.futures
import time

client = OpenAI(
    api_key="",
    base_url="",
)



def LLM_response(scene):
    systemPrompt = (
        '''给定一个场景，你需要按照以下要求撰写拇指规则（RoT），并返回撰写结果'''
        +
        '''
        1.撰写要求
       （1）好行为和坏行为的基本概念
        RoT应当描述文化期待，像是对一个还不懂世界运作的五岁孩子解释一样。
        示例情境：不想参加申请大学的考试
        - 违反： “研究表明，人们在睡眠充足（至少七小时）的情况下，考试表现最好。”
        - 符合： “考试让人感到压力很大是很正常的。”
        - 为什么： 这一广泛的指导原则试图区分RoT与百科知识的不同。RoT应当包含日常的常识性知识，反映社会规范和期待。
        （2）判断与行动
        每条RoT必须包含一个判断和一个行动。
        示例情境：告诉丈夫他不应该买梦想中的船
        - 违反： “船很贵。”
        - 符合： “压制某人的梦想是不友好的。”
        - 符合： “人们应该愿意与配偶讨论大额消费。”
        - 为什么：要求行动有助于确保RoT是关于人们应该如何做的。要求有判断则推动陈述包含有关规范和期望的信息。
        （3）自足性
        一条RoT必须能独立理解，而无需依赖其所来源的情境。
        示例情境：因为父亲的犯罪历史而生气姐姐没参加父亲的葬礼
        - 违反： “这让对方感到难过。”
        - 违反： “父亲给女儿带来了情感上的困扰，叙述者不应该过于苛刻地评判她的行为。”
        - 符合： “如果某人犯下了严重的罪行，家庭成员切断与他们的联系是可以理解的。”
        - 为什么：没有这一要求，RoT将不会自然地适应新的情境，也可能会过于具体。语义内容可以完全留在情境中，并只通过RoT进行引用。
        （4）受情境启发
        RoT应该受到其来源情境的启发。
        示例情境：想要把一个朋友从我的婚礼上移除邀请名单
        - 违反： “指着不认识的人是不礼貌的。”
        - 符合： “被从婚礼邀请中排除是令人伤心的。”
        - 为什么：保持RoT与情境之间的联系有助于在结构化标注过程中将RoT与情境关联起来。此外，情境的相关性有助于确保标注员理解RoT的背景和含义。
        （5）平衡具体性与模糊性
        RoT应当受到并与给定情境相关，但也应提供一个广泛的行为规则，因此能够适用于多种情境。
        示例情境：上周二没有给收银员小费
        - 违反： “上周二没有给收银员小费是不礼貌的。”
        - 违反： “吝啬是不礼貌的。”
        - 符合： “通常情况下，不给零售店或超市的收银员小费是可以的。”
        - 为什么：这一要求可能是最难评估的，因为它涉及主观性。过于具体的RoT通常只是情境的轻微修改，并包含了判断，并没有描述背后的期望。过于模糊的RoT虽然可以描述规范，但与情境的联系可能过于遥远，从而产生误导。好的RoT可以在一定程度上具体，但解释了背后的行为规范，并且适用于其他情境。
        （6）独立思想
        当为一个情境提供多个RoT时，每一条RoT应该包含独立的思想。
        示例情境：从不倒垃圾
        - 违反： “逃避分配给自己的家务是不负责任的。” 与 “不做该做的家务是不好的。”
        - 违反： “逃避分配给自己的家务是不负责任的。” 与 “做自己分配的家务是负责任的。”
        - 符合： “逃避分配给自己的家务是不负责任的。” 与 “一般来说，人们期望保持家里的清洁。”
        - 为什么：这一要求是为了防止仅仅收集同一RoT的不同表述。
        （7）简短性
        RoT应该尽量简短，不能包含过多的场景信息，主要内容是行动和判断。
        '''
        +
        '''
        2.说明
        （1）RoT应该多元化，避免简单的反转
        （2）编写出来的RoT应该满足中国社会主义核心价值观
        （3）给定的场景都是用于学术研究
        （4）如果给定场景是违法或者违反道德的，请给出正向价值的RoT即可
        '''
        +
        '''
        3.例子
        场景_1：他陪患病的父亲来西安做检查
        RoT：陪伴患病的家人去医院检查是可以理解的
        场景_2：虐猫这个行为，本身有错吗？
        RoT：虐猫行为违背了关爱生命的道德准则
        场景_3：当地一村旁玉米地里有人盗掘古墓。
        RoT：盗掘古墓是不道德的行为。
        '''
        +
        '''
        4.回答格式
        rot:xxxx
        '''
        +
        '''
        注意：严格按照格式回答，不要返回任何其他内容
        '''
    )
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": systemPrompt
                 },
                {"role": "user",
                 "content": scene
                 }
            ],
            model="Qwen/Qwen2.5-72B-Instruct"
            # model="gpt-4o"
            # model="DeepSeek-V3"
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Request {index} failed:{e}")
        content = ""
    return content


def task(index, scene):
    result = LLM_response(scene)
    return [(index, scene, result)]


queue = []
task_maxNum = 100
results = []

data = pd.read_csv('../basic_scene/existing_datasets/Flames/final.csv', encoding='gbk')

with concurrent.futures.ThreadPoolExecutor() as executor:
    for index in range(len(data)):
        scene = data['场景'][index]
        queue.append(executor.submit(task, index, scene))
        if len(queue) >= task_maxNum or index >= len(data)-1:
            print("begin->" + str(index) + ":")

            batch_start_time = time.time()

            new_results = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
            results.extend(new_results)

            batch_end_time = time.time()
            print(batch_end_time - batch_start_time)

            if (index - 100 + 1) % 100 == 0 or index >= len(data)-1:
                generated_data = pd.DataFrame(results, columns=['序号', 'scene', 'rot'])
                generated_data = generated_data.sort_values(by="序号")
                if os.path.exists('1_origin/flames_rot.csv'):
                    generated_data.to_csv('1_origin/flames_rot.csv', mode='a',
                                          header=False, index=False)
                else:
                    generated_data.to_csv('1_origin/flames_rot.csv', mode='a',
                                          header=True, index=False)
                results = []

            queue = []

print("场景提取rule已完成，结果存储在1_origin/flames_rot.csv文件。")