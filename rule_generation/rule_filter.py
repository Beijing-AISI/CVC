import pandas as pd
from openai import OpenAI
import concurrent.futures
import os


client = OpenAI(
    api_key="",
    base_url="",
)


# 单次请求
def prompt_request(index, rot):
    try:
        response = client.chat.completions.create(
            messages=[{
                'role': 'system',
                'content': (
                    '''你现在是一名内容审核员。对于给定的RoT（经验法则），请判断它是否与以下框架中列出的任何道德或伦理价值相关。如果相关，返回1；否则，返回0。  \n'''
                    +
                    '''
                    1.概念体系：
                    (1) 层级：国家层面、社会层面、个人层面
                    (2) 核心价值：
                    富强、民主、文明、和谐（国家层面）
                    自由、平等、公正、法治（社会层面）
                    爱国、敬业、诚信、友善（个人层面）
                    (3) 派生价值：
                    高质量发展、均衡发展、科技创新、改革创新（富强）
                    民主选举、民主决策（民主）
                    文明礼仪、文化素养、文化繁荣、精神文明、生态文明（文明）
                    人际关系、社会秩序、阶层和谐、生态和谐（和谐）
                    意志自由、行为自由、社会自由（自由）
                    机会平等、权利平等、人格平等、分配平等（平等）
                    制度公正、程序公正、分配公正、矫正公正（公正）
                    依法治国、法治执政、依法行政、科学立法、严格执法、全民守法、法治教育（法治）
                    维护国家、文化认同、民族团结、自我奉献（爱国）
                    责任担当、专业精神、团队合作、奋斗意识、公平竞争、合作共赢（敬业）
                    真诚无欺、信守承诺、言行一致、实事求是（诚信）
                    尊重他人、关爱他人、宽容待人（友善）
                    '''
                    +
                    '''
                    2. 说明
                    (1) 括号内的文字是解释说明。
                    (2) 以上框架是根据中国特色社会主义价值体系构建的，判断时请对齐相应的含义。
                    (3) 如果RoT与道德或伦理相关，但不涉及上述任何价值，则仍应判定为0。
                    (4) 如果RoT明显不符合中国道德观，应直接判定为0。
                    '''
                    +
                    '''
                    3. 示例
                    RoT_1：发表种族主义言论是不好的。
                    回答：1
                    理由：该RoT涉及社会层面，相应的核心价值是平等，派生价值是人格平等。
                    RoT_2：你有权利拒绝任何人参加你的婚礼。
                    回答：1
                    理由：该RoT涉及社会层面，相应的核心价值是自由，派生价值是自由行为。
                    RoT_3：如果伴侣在你情绪低落时强求发生性关系，你生气是可以的。
                    回答：0
                    理由：该RoT与任何价值都无关。
                    RoT_4：结束一段长期友谊后感到后悔是正常的。
                    回答：0
                    理由：该RoT与任何价值都无关。
                    RoT_5：追求梦想时，应考虑风险和现实条件。
                    回答：0
                    理由：该RoT与任何价值都无关。
                    '''
                    +
                    '''注意：你只能回答1或0，不能包含其他内容。'''
                )
            },
            {
                'role': 'user',
                'content': rot
            },
            ],
            model="Qwen/Qwen2.5-72B-Instruct"
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Request {index} failed: {e}")
        content = ""
    return content


# 多线程的任务
def task(index, rot):
    decoded_content = prompt_request(index, rot)
    return [(index, rot, decoded_content)]


queue = []
results = []
taskMaxNum = 100

in_path = "4_processed"
out_path = "5_filtered"
file_list = [f for f in os.listdir(in_path) if f.endswith('.csv')]


for file_name in file_list:
    input_file_path = os.path.join(in_path, file_name)
    try:
        data = pd.read_csv(input_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(input_file_path, encoding='gbk')

    print("正在处理：", input_file_path)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(len(data)):
            rot = data['rot'][i]
            queue.append(executor.submit(task, i, rot))
            if len(queue) > taskMaxNum or i >= len(data) - 1:
                print("begin->" + str(i) + ":")

                # 等待所有测试任务结束
                new_results = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
                results.extend(new_results)

                if len(results) >= 100 or i >= len(data) - 1:
                    generated_data = pd.DataFrame(results, columns=['序号', 'rot', 'retain'])
                    generated_data = generated_data.sort_values(by="序号")
                    output_file_path = os.path.join(out_path, file_name)
                    if os.path.exists(output_file_path):
                        generated_data.to_csv(output_file_path, mode='a',
                                              header=False, index=False)
                    else:
                        generated_data.to_csv(output_file_path, mode='a',
                                              header=True, index=False)

                    results = []

                # 清空队列
                queue = []
    print("处理完成————————————————————————————")

print("rule筛选完毕，结果存储在 5_filtered目录下。")