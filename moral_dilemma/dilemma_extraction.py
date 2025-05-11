import pandas as pd
import os


def process_scene_phase1(scene_str):
    """第一阶段：处理原始场景文本，分离场景内容和选项"""
    if pd.isna(scene_str):
        return "", ""

    scene_lines = []
    options = []
    for part in scene_str.split("\n"):
        part = part.strip()
        if not part:
            continue
        if part[0].isdigit():
            scene_lines.append(part.split(":", 1)[-1].strip())
        elif part[0] in ("A", "B", "C"):
            options.append(part)
    return "\n".join(scene_lines), "\n".join(options)


def process_row_phase2(row, extra_fields):
    """第二阶段：将场景内容和选项拆分为多行记录"""

    def safe_str(value):
        return str(value) if pd.notnull(value) else ""

    scene_str = safe_str(row["scene"])
    options_str = safe_str(row["options"])

    scene_parts = scene_str.split("\n") if scene_str else []
    options_parts = options_str.split("\n") if options_str else []

    processed_rows = []
    for scene_counter, scene_part in enumerate(scene_parts):
        scene_text = scene_part.split(":", 1)[-1].strip()
        option_start = scene_counter * 3
        option_end = option_start + 3
        current_options = options_parts[option_start:option_end]

        new_row = {
            "index": row["index"],
            "scene": scene_text,
            "options": "\n".join(current_options),
        }

        # 加入其余字段
        for field in extra_fields:
            new_row[field] = safe_str(row.get(field, ""))

        processed_rows.append(new_row)

    return processed_rows


def main():
    input_file = "dataset/2_dilemma/1_origin/all_dilemma_1.csv"
    output_file = "dataset/2_dilemma/2_processed/all_dilemma_1.csv"
    try:
        df = pd.read_csv(input_file, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='utf-8')

    # 预处理分割scene字段
    df[["scene", "options"]] = df["scene"].apply(lambda x: pd.Series(process_scene_phase1(x)))

    extra_fields = [col for col in df.columns if col not in ("index", "scene", "options")]

    output_data = []
    for _, row in df.iterrows():
        output_data.extend(process_row_phase2(row, extra_fields))

    result_df = pd.DataFrame(output_data)
    result_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
