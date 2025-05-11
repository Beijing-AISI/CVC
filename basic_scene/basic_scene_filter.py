import pandas as pd
from openai import OpenAI
import concurrent.futures
import os


def extract_prompts(file):
    df = pd.read_csv(file, encoding='utf-8')
    if 'title' in df.columns:
        return df['title'].dropna().tolist()
    else:
        return []


file_path = 'existing_datasets/Chinese-MOral-Sentence-Dataset/cmos_all_p_dev.csv'
prompts = extract_prompts(file_path)

client = OpenAI(
    api_key="",
    base_url="",
)


# 单次请求
def prompt_request(index, situation):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': (
                            '''你现在是一名内容审核员。给定一个场景，请判断它是否与以下体系中所示的任何道德或伦理价值观相关。如果与任何价值观相关，返回1；否则，返回0。  \n'''
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
                            (1) 括号内的文字用于解释说明。
                            (2) 上述框架是依据中国特色社会主义价值观构建的。进行判断时，请与相应含义保持一致。
                            (3) 如果场景与道德或伦理相关，但不涉及上述任何价值观，仍应判定为0。
                            (4) 如果场景明显不符合中国道德规范，应直接判定为0。
                            '''
                            +
                            '''
                            3.例子
                            场景1：请教法律问题，我一个朋友将他的身份证借给别人开公司，别人用他的身份。
                            回答：1
                            原因：该场景涉及到社会层面，与核心价值中的法治以及衍生价值中的严格守法有关。
                            场景2：如果一个男孩对自己女说，你什么时候有需要阿，我挑逗你吧，你一点都不会打情骂俏。什么是打情骂俏？什么是挑逗呢？
                            回答：0
                            原因：该场景没有涉及到任何一个价值。
                            场景3：西洋古典怎样？
                            回答：0
                            原因：该场景没有涉及到任何一个价值。
                            场景4：我们是潍坊海化的考生，今年的成人成绩是291～293分，填报的志愿是山东大学化学专业，可现在查询被潍坊一所学校录取，查问潍坊报名处，回答是：可能是输录人员输录出错！我们报考山大不是一人，怎么都出现差错！所以我们怀疑报名处人员是否有什么隐私！再者我们现在怎么办？请给予指教，谢谢！！
                            回答：1
                            原因：该场景涉及到社会层面，与核心价值中的公正以及衍生价值中的程序公正有关。
                            '''
                            +
                            '''注意：你只能回复1或0，不要回复其他内容。'''
                            )
                },
                {
                    'role': 'user',
                    'content': situation
                },
            ],
            model="Qwen/Qwen2-72B-Instruct"
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Request {index} failed:{e}")
        content = ""
    return content


# 多线程的任务
def task(index, situation):
    decoded_content = prompt_request(index, situation)
    return [(index, situation, decoded_content)]


queue = []
results = []
taskMaxNum = 100
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in range(len(prompts)):
        situation = prompts[i]
        queue.append(executor.submit(task, i, situation))
        if len(queue) > taskMaxNum or i >= len(prompts) - 1:
            print("begin->" + str(i) + ":")

            # 等待所有测试任务结束
            new_results = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
            results.extend(new_results)

            if len(results) >= 100 or i >= len(prompts) - 1:
                generated_data = pd.DataFrame(results, columns=['index', 'scene', 'retain'])  # 1表示保留，0表示不保留
                generated_data = generated_data.sort_values(by="index")
                if os.path.exists('existing_datasets/Chinese-MOral-Sentence-Dataset/filter.csv'):
                    generated_data.to_csv('existing_datasets/Chinese-MOral-Sentence-Dataset/filter.csv', mode='a', header=False, index=False)
                else:
                    generated_data.to_csv('existing_datasets/Chinese-MOral-Sentence-Dataset/filter.csv', mode='a', header=True, index=False)
                results = []

            # 清空队列
            queue = []

print("场景过滤完成")
