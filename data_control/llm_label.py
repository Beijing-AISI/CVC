# -*- coding: utf-8 -*-
from openai import OpenAI
import pandas as pd
import concurrent.futures
import os

client = OpenAI(
    api_key="",
    base_url="",
)

single_values = [
    '高质量发展', '均衡发展', '科技创新', '改革创新',
    '民主选举', '民主决策',
    '文明礼仪', '文化修养', '文化繁荣', '精神文明', '生态文明',
    '人际关系', '社会秩序', '阶层和谐', '生态和谐',
    '意志自由', '行为自由', '社会自由',
    '机会平等', '权利平等', '人格平等', '分配平等',
    '制度公正', '程序公正', '分配公正', '矫正公正',
    '依法治国', '依法执政', '依法行政', '科学立法', '严格守法', '全民守法', '法制教育',
    '维护国家', '文化认同', '民族团结', '自我奉献',
    '责任担当', '专业精神', '团队合作', '奋斗意识', '公平竞争', '合作共赢',
    '真诚无欺', '信守承诺', '言行一致', '实事求是',
    '尊重他人', '关爱他人', '宽容待人'
]

systemPrompt = (
        '''
        给定一个RoT（Rule of Thumb)和一些人工标注的示例，请返回该RoT的标签和重写结果。
        '''
        +
        '''
        1.RoT的定义：
        （1）包含一个判断（例如“你应该”）和一个行动（例如“帮助他人”）
        （2）不能过于冗长，大致在20字以内
        '''
        +
        '''
        2.价值体系
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
        3.标注要求
        （1）如果与上述价值无关则标0，重写值为空。
        （2）如果违反中国价值体系则标-1，并给出你认为符合中国价值的重写RoT。
        （3）如果不符合定义或语义不完整，但与上述价值有关，请标1并给出符合要求的重写RoT。
        （4）如果不存在上述问题直接标1，重写值为空
        注意：判断应从第（1）点开始，满足即停止；如不满足则依次判断后续条件。
        '''
        +
        '''
        4.回答格式
        标签：-1/0/1
        重写：xxxxxxxxxxx
        '''
        +
        '''注意：所有标注应基于上述价值体系，只能返回标签和重写结果，不要返回任何其他内容  '''
)


def LLM_response(index, rot, sample):
    sample_list = "————————————————示例————————————————\n"
    for i in range(len(sample)):
        sample_list += f"RoT_{i}：{sample['rot'][i]}\n标签：{sample['标签'][i]}\n重写：{sample['重写'][i]}\n"
    try:
        user_content = sample_list + "————————————————待标注————————————————\n" + f"RoT: {rot}"

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
            # model="DeepSeek-R1"
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Request {index} failed:{e}")
        content = ""
    return content


def task(index, rot, level, core_values, derived_values, sample):
    result = LLM_response(index, rot, sample)
    return [(index, rot, level, core_values, derived_values, result)]


queue = []
task_maxNum = 25
results = []

in_file = "label_rot_data/llm/llm_rot.csv"
out_file = "label_rot_data/llm/llm_label_raw.csv"

try:
    data = pd.read_csv(in_file, encoding='gbk')
except UnicodeDecodeError:
    data = pd.read_csv(in_file, encoding='utf-8')


def process_row(index):
    row = data.iloc[index]
    derived_values = row['衍生价值']

    if pd.isna(derived_values) or not derived_values.strip():
        return pd.DataFrame()

    # 是否多个价值
    values = [v.strip() for v in derived_values.split('、')]
    is_combined = len(values) > 1

    try:
        if is_combined:
            filepath = "label_rot_data/human_label_samples/51_组合价值.csv"
        else:
            value = values[0]
            if value not in single_values:
                return pd.DataFrame()
            value_index = single_values.index(value) + 1  # 编号从1开始
            filepath = f"label_rot_data/human_label_samples/{value_index}_{value}.csv"

        # 文件不存在直接跳过
        if not os.path.exists(filepath):
            print(f"文件未找到: {filepath}")
            return pd.DataFrame()

        # 读取文件并随机抽取最多5行
        file_df = pd.read_csv(filepath)
        sample_df = file_df.sample(n=min(5, len(file_df)), random_state=42)
        return sample_df[['rot', '标签', '重写']].reset_index(drop=True)

    except Exception as e:
        print(f"处理第{index}行出错: {e}")
        print(row)
        return pd.DataFrame()


with concurrent.futures.ThreadPoolExecutor() as executor:
    for index in range(len(data)):
        rot = data['rot'][index]
        level = data['层面'][index]
        core_values = data['核心价值'][index]
        derived_values = data['衍生价值'][index]
        # 获取少样本作为提示
        sample = process_row(index)
        queue.append(executor.submit(task, index, rot, level, core_values, derived_values, sample))
        if len(queue) >= task_maxNum or index >= len(data) - 1:
            print("begin->" + str(index) + ":")

            new_results = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
            results.extend(new_results)

            queue = []

            if len(results) >= 100 or index >= len(data) - 1:
                generated_data = pd.DataFrame(results, columns=['index', 'rot', 'level', 'core values', 'derived values', '标签'])
                generated_data = generated_data.sort_values(by="index")
                if os.path.exists(out_file):
                    generated_data.to_csv(out_file, mode='a',
                                          header=False, index=False)
                else:
                    generated_data.to_csv(out_file, mode='a',
                                          header=True, index=False)
                results = []

print("llm标注完成。")
