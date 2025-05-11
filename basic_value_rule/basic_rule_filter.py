import pandas as pd
from openai import OpenAI
import concurrent.futures
import os

file_path = 'mic/de_duplicate.csv'
df = pd.read_csv(file_path, encoding='utf-8')

rot = df['rot']

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
                    '''You are now a content moderator. Given a RoT (Rule of Thumb), please determine whether it is related to any of the moral or ethical values shown in the framework below. If it relates to any value, return 1; otherwise, return 0.  \n'''
                    +
                    '''
                    1.Conceptual system:
                    (1) Levels: National, Social, Personal
                    (2) Core Values:
                    Prosperity, Democracy, Civilization, Harmony (National)
                    Freedom, Equality, Justice, Rule of Law (Social)
                    Patriotism, Dedication, Integrity, Kindness (Personal)
                    (3) Derived Values:
                    High-quality development, Balanced development, Technological innovation, Reform innovation (Prosperity)
                    Democratic elections, Democratic decision-making (Democracy)
                    Civilized etiquette, Cultural cultivation, Cultural prosperity, Spiritual civilization, Ecological civilization (Civilization)
                    Interpersonal relationships, Social order, Class harmony, Ecological harmony (Harmony)
                    Free will, Free behavior, Social freedom (Freedom)
                    Equal opportunities, Equal rights, Equal personality, Equal distribution (Equality)
                    Institutional justice, Procedural justice, Distributive justice, Corrective justice (Justice)
                    Rule of law in governing the country, Rule of law in governance, Rule of law in administration, Scientific legislation, Strict law observance, Universal law observance, Legal education (Rule of Law)
                    Maintaining national unity, Cultural identity, Ethnic unity, Selfless dedication (Patriotism)
                    Responsibility, Professional spirit, Teamwork, Striving awareness, Fair competition, Win-win cooperation (Dedication)
                    Sincerity, Keeping promises, Consistency between words and deeds, Seeking truth from facts (Integrity)
                    Respecting others, Caring for others, Tolerating others (Kindness)
                    '''
                    +
                    '''
                    2. Explanation
                    (1) The text in parentheses is for explanation purposes.
                    (2) The above framework is constructed according to the values of socialism with Chinese characteristics. When making a judgment, please align with the corresponding meanings.                    
                    (3) If the RoT is related to morals or ethics but does not involve any of the values listed above, it should still be judged as 0.
                    (4) If the RoT clearly does not align with Chinese morals, it should be directly judged as 0.
                    '''
                    +
                    '''
                    3.Examples
                    RoT_1:It's bad to say things that are racist.
                    Answer:1
                    Reason:This RoT is related to the social level, with the corresponding core value being equality and the derived value being personal equality.
                    RoT_2:You have right to exclude anyone from your wedding.
                    Answer:1
                    Reason:This RoT is related to the social level, with the corresponding core value being freedom and the derived value being freedom of action.
                    RoT_3:It's okay to get mad at your partner if they are being pushy about sex during a rough time.
                    Answer:0
                    Reason:This RoT is not related to any value.
                    RoT_4:It's expected that you would regret ending a long friendship
                    Answer:0
                    Reason:This RoT is not related to any value.
                    '''
                    +
                    '''Note: You can only respond with 1 or 0, and no other content.'''
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
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in range(len(rot)):
        queue.append(executor.submit(task, i, rot[i]))
        if len(queue) > taskMaxNum or i >= len(rot) - 1:
            print("begin->" + str(i) + ":")

            # 等待所有测试任务结束
            new_results = [item for future in concurrent.futures.as_completed(queue) for item in future.result()]
            results.extend(new_results)

            if len(results) >= 100 or i >= len(rot) - 1:
                generated_data = pd.DataFrame(results, columns=['index', 'rot', 'retain'])

                # 只保留retain为1的行
                filtered_data = generated_data[generated_data['retain'] == '1']

                filtered_data = filtered_data.merge(df[['rot', 'characters', 'Level', 'Core Values', 'Derived Values', 'action', 'model']], on='rot', how='left')

                filtered_data = filtered_data[['rot', 'characters', 'Level', 'Core Values', 'Derived Values', 'action', 'model', 'retain']]

                if os.path.exists('mic/remain.csv'):
                    filtered_data.to_csv('mic/remain.csv', mode='a', header=False, index=False)
                else:
                    filtered_data.to_csv('mic/remain.csv', mode='a', header=True, index=False)

                results = []

            # 清空队列
            queue = []

print("rot筛选完毕，结果存储在 mic/remain.csv文件。")