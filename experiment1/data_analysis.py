# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
import numpy as np

matplotlib.use('Agg')

# in_file = "100/without_rule_scene_100.csv"
in_file = "100/with_rule_scene_100.csv"

try:
    df = pd.read_csv(in_file, encoding="gbk")
except UnicodeDecodeError:
    df = pd.read_csv(in_file, encoding="utf-8")

scenes = df['scene'].tolist()
themes = df['theme'].tolist()

zh2en = {
    "富强": "Prosperity",
    "民主": "Democracy",
    "文明": "Civility",
    "和谐": "Harmony",
    "自由": "Freedom",
    "平等": "Equality",
    "公正": "Justice",
    "法治": "Rule of Law",
    "爱国": "Patriotism",
    "敬业": "Dedication",
    "诚信": "Integrity",
    "友善": "Friendship"
}
themes_en = [zh2en.get(t, t) for t in themes]

local_model_path = "./all-MiniLM-L6-v2"
model = SentenceTransformer(local_model_path)
scene_embeddings = model.encode(scenes, show_progress_bar=True)


def visualize():
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['axes.unicode_minus'] = False

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    scene_2d = tsne.fit_transform(scene_embeddings)

    label_encoder = LabelEncoder()
    theme_ids = label_encoder.fit_transform(themes_en)
    theme_names = label_encoder.classes_

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(scene_2d[:, 0], scene_2d[:, 1], c=theme_ids, cmap='Set3', alpha=0.85)

    handles = [plt.Line2D([], [], marker="o", color=scatter.cmap(scatter.norm(i)), linestyle="",
                          label=theme_names[i]) for i in range(len(theme_names))]
    plt.legend(handles=handles, title="Value", bbox_to_anchor=(1.05, 1), loc='upper left')

    # plt.title("t-SNE Clustering Visualization by Theme")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    out_file = in_file.replace('.csv', '.png')
    plt.savefig(out_file, dpi=300)
    plt.show()


def distance_in_theme():
    theme_to_embeddings = defaultdict(list)
    for emb, theme in zip(scene_embeddings, themes):
        theme_to_embeddings[theme].append(emb)

    seen_themes = []
    for t in themes:
        if t not in seen_themes:
            seen_themes.append(t)

    theme_avg_distances = {}
    for theme in seen_themes:
        vectors = np.array(theme_to_embeddings[theme])
        if len(vectors) < 2:
            theme_avg_distances[theme] = 0
            continue
        dists = euclidean_distances(vectors)
        triu_indices = np.triu_indices_from(dists, k=1)
        avg_dist = dists[triu_indices].mean()
        theme_avg_distances[theme] = round(avg_dist, 2)

    rows = [(theme, theme_avg_distances[theme]) for theme in seen_themes]
    df_avg_dist = pd.DataFrame(rows, columns=["theme", "distance"])

    print(df_avg_dist)
    output_file = in_file.replace('_100.csv', '_distance.csv')
    df_avg_dist.to_csv(output_file, index=False)

    # 总平均距离
    overall_avg = df_avg_dist["distance"].mean()
    print("\nOverall Average Intra-class Distance:", round(overall_avg, 2))


if __name__ == '__main__':
    # visualize()
    distance_in_theme()
