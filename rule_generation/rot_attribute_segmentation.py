"""中文rot分类"""
from openai import OpenAI
import pandas as pd
import concurrent.futures
import os

client = OpenAI(
    api_key="",
    base_url="",
)

systemPrompt = (
        '''按照以下要求，你需要对给定的拇指规则（RoT）进行三层分类，并返回分类结果。  \n'''
        +
        '''
        1.分类体系
        （1）层面：国家、社会、个人
        （2）核心价值：
        富强、民主、文明、和谐（国家）
        自由、平等、公正、法治（社会）
        爱国、敬业、诚信、友善（个人）
        （3）衍生价值：
        高质量发展、均衡发展、科技创新、改革创新（富强）
        民主选举、民主决策（民主）
        文明礼仪、文化修养、文化繁荣、精神文明、生态文明（文明）
        人际关系、社会秩序、阶层和谐、生态和谐（和谐）
        意志自由、行为自由、社会自由（自由）
        机会平等、权利平等、人格平等、分配平等（平等）
        制度公正、程序公正、分配公正、矫正公正（公正）
        依法治国、依法执政、依法行政、科学立法、严格守法、全民守法、法治教育（法治）
        维护国家、文化认同、民族团结、自我奉献（爱国）
        责任担当、专业精神、团队合作、奋斗意识、公平竞争、合作共赢（敬业）
        真诚无欺、信守承诺、言行一致、实事求是（诚信）
        尊重他人、关爱他人、宽容待人（友善）
        '''
        +
        '''
        2.说明
        （1）括号内的文字属于说明，不包含在分类结果中。
        （2）以上体系是按照中国特色社会主义构建的，在分类时要契合对应的含义。
        （3）在同一层的分类中，各个类别不是完全独立的，可以同时包含。
        （4）上述三层分类是层层递进的，下层分类要对应上层分类结果。
        （5）除上述提到的类别外不能出现其他的分类词。
        '''
        +
        '''
        3.例子
        RoT_1:尊重其他种族的传统文化是应该的
        层面:国家、个人
        核心价值：文明、友善
        衍生价值：文明礼仪、尊重他人
        RoT_2:被他人拒绝后对其进行骚扰是不道德的行为
        层面:国家、个人
        核心价值:文明、和谐、友善
        衍生价值:文明礼仪、人际关系、尊重他人
        '''
        +
        '''
        4.回答格式
        层面：xxxx
        核心价值：xxxx
        衍生价值：xxxx
        '''
        +
        '''
        注意：除了分类结果，不要出现任何其他内容，并且只能出现分类体系中的词，不可以添加其他词！！！
        '''
    )


def LLM_response(index, rot):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": systemPrompt
                 },
                {"role": "user",
                 "content": rot
                 }
            ],
            model="Qwen/Qwen2.5-72B-Instruct"
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Request {index} failed:{e}")
        content = ""
    return content


def task(index, rot):
    result = LLM_response(index, rot)
    return [(index, rot, result)]


queue = []
task_maxNum = 100
results = []

input_file = "5_filtered/cmos_all_p_dev_rot.csv"
try:
    origin = pd.read_csv(input_file, encoding='utf-8')
except UnicodeDecodeError:
    origin = pd.read_csv(input_file, encoding='gbk')

with concurrent.futures.ThreadPoolExecutor() as executor:
    for index in range(len(origin)):
        rot = origin['rot2'][index]
        queue.append(executor.submit(task, index, rot))
        if len(queue) >= task_maxNum or index >= len(origin) - 1:
            print("begin->" + str(index) + ":")

            new_results = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
            results.extend(new_results)

            if len(results) >= 100 or index >= len(origin) - 1:
                generated_data = pd.DataFrame(results, columns=['index', 'rot', 'derived_values'])
                generated_data = generated_data.sort_values(by="index")
                output_file = "6_final/cmos_all_p_dev_rot.csv"
                if os.path.exists(output_file):
                    generated_data.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    generated_data.to_csv(output_file, mode='a', header=True, index=False)
                results = []

            queue = []

print(f"{input_file}处理完成————————————————————————————")

