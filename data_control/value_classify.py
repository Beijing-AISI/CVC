# -*- coding: utf-8 -*-
"""按照层面分类"""
# import pandas as pd
#
# # 读取数据
# df = pd.read_csv('final.csv', encoding='gbk')
#
# # 定义分类函数
# def classify(row):
#     levels = set(row['level'].split('、'))
#     if levels == {'国家', '社会', '个人'}:
#         return 'National_Social_Personal'
#     elif levels == {'国家', '社会'}:
#         return 'National_Social'
#     elif levels == {'国家', '个人'}:
#         return 'National_Personal'
#     elif levels == {'社会', '个人'}:
#         return 'Social_Personal'
#     elif levels == {'国家'}:
#         return 'National'
#     elif levels == {'社会'}:
#         return 'Social'
#     elif levels == {'个人'}:
#         return 'Personal'
#     else:
#         return None  # 其他异常情况
#
# # 应用分类函数
# df['new_level'] = df.apply(classify, axis=1)
#
# # 处理无法分类的情况
# temp_df = df[df['new_level'].isna()].copy()
#
# # 定义七个分类
# categories = [
#     'National',
#     'Social',
#     'Personal',
#     'National_Social',
#     'National_Personal',
#     'Social_Personal',
#     'National_Social_Personal'
# ]
#
# # 遍历分类并保存文件
# for index, category in enumerate(categories):
#     category_df = df[df['new_level'] == category].copy()
#     if not category_df.empty:
#         category_df.drop(columns=['level', 'new_level'], inplace=True)
#         print(f"{category}: {len(category_df)} 条")
#         category_df.to_csv(f'层面/{index}.{category}.csv', index=False, encoding='utf-8-sig')
#         print(f"分类 '{category}' 的数据已保存为 '层面/{index}.{category}.csv'.")
#
# # 保存未分类的数据
# if not temp_df.empty:
#     temp_df.drop(columns=['level', 'new_level'], inplace=True)
#     print(f"未分类数据: {len(temp_df)} 条")
#     temp_df.to_csv('层面/temp.csv', index=False, encoding='utf-8-sig')
#     print("未分类数据已保存为 '层面/temp.csv'.")
#
# print("所有分类保存完成。")


"""按照核心价值分类"""
import pandas as pd

# 读取数据
df = pd.read_csv('final.csv', encoding='gbk')

# 定义单一价值和组合价值
# single_values = ['Prosperity', 'Democracy', 'Civilization', 'Harmony',
#                  'Freedom', 'Equality', 'Justice', 'Rule of Law',
#                  'Patriotism', 'Dedication', 'Integrity', 'Kindness']
single_values = ['富强', '民主', '文明', '和谐',
                 '自由', '平等', '公正', '法治',
                 '爱国', '敬业', '诚信', '友善']

# 组合价值文件
combo_value = 'Combination'


# 创建一个新的列 'Category_Core_Values' 来存储分类信息
def classify_core_values(row):
    # 如果 'Core Values' 为空或不是字符串，返回组合价值
    if pd.isna(row['core values']) or not isinstance(row['core values'], str):
        return combo_value

    values = row['core values'].split('、')

    # 判断是否为单一价值
    if len(values) == 1 and values[0] in single_values:
        return values[0]  # 单一价值
    else:
        return combo_value  # 组合价值


# 分类 'Core Values' 列
df['Category_Core_Values'] = df.apply(classify_core_values, axis=1)

# 保存每个分类的文件
for index, value in enumerate(single_values + [combo_value], start=1):
    category_df = df[df['Category_Core_Values'] == value].copy()

    # 如果分类不为空，删除临时的 'Category_Core_Values' 列并保存
    if not category_df.empty:
        category_df.drop(columns=['Category_Core_Values'], inplace=True)
        # 为单一价值文件名添加标号
        if value != combo_value:
            file_name = f'核心价值/{index}_{value}.csv'
        else:
            file_name = f'核心价值/{index}_{value}.csv'

        print(f"保存 {value} 分类数据，共 {len(category_df)} 条记录")
        category_df.to_csv(file_name, index=False)

print("Core Values 分类保存完成。")

"""按照衍生价值分类"""
# single_values_derived = [
#     'High-quality development', 'Balanced development', 'Technological innovation', 'Reform innovation',
#     'Democratic elections', 'Democratic decision-making',
#     'Civilized etiquette', 'Cultural cultivation', 'Cultural prosperity', 'Spiritual civilization',
#     'Ecological civilization',
#     'Interpersonal relationships', 'Social order', 'Class harmony', 'Ecological harmony',
#     'Free will', 'Free behavior', 'Social freedom',
#     'Equal opportunities', 'Equal rights', 'Equal personality', 'Equal distribution',
#     'Institutional justice', 'Procedural justice', 'Distributive justice', 'Corrective justice',
#     'Rule of law in governing the country', 'Rule of law in governance', 'Rule of law in administration',
#     'Scientific legislation',
#     'Strict law observance', 'Universal law observance', 'Legal education',
#     'Maintaining national unity', 'Cultural identity', 'Ethnic unity', 'Selfless dedication',
#     'Responsibility', 'Professional spirit', 'Teamwork', 'Striving awareness', 'Fair competition',
#     'Win-win cooperation',
#     'Sincerity', 'Keeping promises', 'Consistency between words and deeds', 'Seeking truth from facts',
#     'Respecting others', 'Caring for others', 'Tolerating others'
# ]
# import pandas as pd
#
# # 读取数据
# df = pd.read_csv('final.csv', encoding='gbk')
#
# # 定义单一衍生价值列表（中文）
# single_values_derived = [
#     '高质量发展', '均衡发展', '科技创新', '改革创新',
#     '民主选举', '民主决策',
#     '文明礼仪', '文化修养', '文化繁荣', '精神文明','生态文明',
#     '人际关系', '社会秩序', '阶层和谐', '生态和谐',
#     '意志自由', '行为自由', '社会自由',
#     '机会平等', '权利平等', '人格平等', '分配平等',
#     '制度公正', '程序公正', '分配公正', '矫正公正',
#     '依法治国', '依法执政', '依法行政', '科学立法','严格守法', '全民守法', '法制教育',
#     '维护国家', '文化认同', '民族团结', '自我奉献',
#     '责任担当', '专业精神', '团队合作', '奋斗意识', '公平竞争', '合作共赢',
#     '真诚无欺', '信守承诺', '言行一致', '实事求是',
#     '尊重他人', '关爱他人', '宽容待人'
# ]
#
# # 组合价值标签
# combo_value = '组合价值'
#
#
# # 创建分类列
# def classify_derived_values(row):
#     values = row['derived values'].split('、')  # 修改为中文顿号分割
#     if len(values) == 1 and values[0] in single_values_derived:
#         return values[0]
#     else:
#         return combo_value
#
#
# # 添加分类列
# df['Category_Derived_Values'] = df.apply(classify_derived_values, axis=1)
#
# # 按分类保存
# for index, value in enumerate(single_values_derived + [combo_value], start=1):
#     category_df = df[df['Category_Derived_Values'] == value].copy()
#     if not category_df.empty:
#         category_df.drop(columns=['Category_Derived_Values'], inplace=True)
#         file_name = f'衍生价值/{index}_{value}.csv'
#         print(f"保存 {value} 分类数据，共 {len(category_df)} 条记录")
#         category_df.to_csv(file_name, index=False, encoding='utf-8')
#
# print("Derived Values 分类保存完成。")
