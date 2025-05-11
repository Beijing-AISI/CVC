import pandas as pd
import re
import os


def clean_rot(rot_text):
    if pd.isna(rot_text):
        return None

    rot_lines = re.findall(r'rot:(.*?)$', rot_text, re.MULTILINE)

    if not rot_lines:
        return None

    return "\n".join(line.strip() for line in rot_lines)


in_path = "1_origin"
out_path = "2_formatted"

if not os.path.exists(out_path):
    os.makedirs(out_path)

file_list = [f for f in os.listdir(in_path) if f.endswith('.csv')]

for file_name in file_list:
    input_file_path = os.path.join(in_path, file_name)
    try:
        data = pd.read_csv(input_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(input_file_path, encoding='gbk')
    data["rot"] = data["rot"].apply(clean_rot)
    data = data.dropna(subset=["rot"])
    output_file_path = os.path.join(out_path, file_name)
    data.to_csv(output_file_path, index=False)

print("处理完成！")
