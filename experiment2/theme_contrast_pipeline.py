from openai import OpenAI
import pandas as pd
import os
import concurrent.futures

client = OpenAI(
    api_key="",
    base_url="",
)

model_names = [
    "Doubao-1.5-pro-256k",
    "Qwen/Qwen2.5-72B-Instruct",
    "DeepSeek-V3",
    "gpt-4o",
    "gemini-1.5-pro",
    "claude-3-7-sonnet-20250219",
    "aihubmix-Llama-3-1-70B-Instruct"
]

TASK_MAX_NUM = 100
REPEAT_TIMES = 5


def LLM_response(scene, options, model):
    system_prompt = "给定一个情境和三个选项，从你的角度选择一个最佳行为。仅提供答案：A、B 或 C。不要包含任何其他内容。"
    user_prompt = f"场景：{scene}\n选项：\n{options}"
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Request failed: {e}")
        return ""


def task(index, scene, options, text_data, model):
    """每个测试场景进行重复实验"""
    results = [LLM_response(scene, options, model) for _ in range(REPEAT_TIMES)]
    return [(index, scene, options, *results, *text_data.tolist())]


def process_file(in_file, model, out_file):
    try:
        data = pd.read_csv(in_file, encoding='gbk')
    except UnicodeDecodeError:
        data = pd.read_csv(in_file, encoding='utf-8')

    queue = []
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for index in range(len(data)):
            scene, options = data['scene'][index], data['options'][index]
            columns = ['rule1', 'rule2']
            text_data = data.loc[index, columns]
            queue.append(executor.submit(task, index, scene, options, text_data, model))

            if len(queue) >= TASK_MAX_NUM or index == len(data) - 1:
                print("begin->" + str(index) + ":")

                for future in concurrent.futures.as_completed(queue):
                    results.extend(future.result())

                result_columns = ['result' + str(i + 1) for i in range(REPEAT_TIMES)]
                final_columns = ['index', 'scene', 'options'] + result_columns + ['rot1', 'rot2']
                generated_data = pd.DataFrame(results, columns=final_columns)
                generated_data = generated_data.sort_values(by="index")

                if os.path.exists(out_file):
                    generated_data.to_csv(out_file, mode='a', header=False, index=False, encoding='utf-8')
                else:
                    generated_data.to_csv(out_file, mode='a', header=True, index=False, encoding='utf-8')

                results = []
                queue = []


if __name__ == '__main__':
    IN_DIR = "scene"
    OUT_DIR = "result"

    for i in range(len(model_names)):
        model_name = model_names[i]
        print(f"{model_name}评测开始——————————————")

        for filename in os.listdir(IN_DIR):
            if filename.endswith(".csv"):
                input_file = os.path.join(IN_DIR, filename)
                if model_name == "Qwen/Qwen2.5-72B-Instruct":
                    OUT_FILE = os.path.join(OUT_DIR, f"Qwen2.5-72B-Instruct/{filename}")
                else:
                    OUT_FILE = os.path.join(OUT_DIR, f"{model_name}/{filename}")
                os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
                process_file(input_file, model_name, OUT_FILE)

            print(f"{filename}评测结束——————————————")

        print(f"{model_name}评测结束——————————————")
