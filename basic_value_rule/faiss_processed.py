"""英文rot去重"""
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

THRESHOLD = 0.8
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 512
K_NEIGHBORS = 5
N_LIST = 100
N_PROBE = 10
CHUNK_SIZE = 5000


def load_data(file_path):
    print(f"[INFO] 正在加载数据: {file_path} ...")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"[INFO] 加载完成，共 {len(df)} 条数据。")
    return df['rot'].tolist(), df


def compute_embeddings(sentences, model, batch_size=BATCH_SIZE):
    print("[INFO] 开始计算句子嵌入 ...")
    embeddings = []
    start_time = time.time()

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, device="cpu").cpu().numpy().astype('float32')
        embeddings.append(batch_embeddings)

        progress = (i + batch_size) / len(sentences) * 100
        print(f"[PROGRESS] 嵌入计算进度: {progress:.2f}% ({i + batch_size}/{len(sentences)})")

    print(f"[INFO] 嵌入计算完成，总耗时 {time.time() - start_time:.2f} 秒")
    return np.vstack(embeddings)


def find_duplicates(sentences, model, threshold=THRESHOLD):
    embeddings = compute_embeddings(sentences, model)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, N_LIST, faiss.METRIC_INNER_PRODUCT)

    print("[INFO] 训练 FAISS 索引 ...")
    index.train(embeddings)
    print("[INFO] 索引训练完成")

    print("[INFO] 开始添加数据到索引 ...")
    for start in range(0, len(embeddings), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(embeddings))
        index.add(embeddings[start:end])
        print(f"[PROGRESS] FAISS 索引构建: {end}/{len(embeddings)} 条数据")

    index.nprobe = min(N_PROBE, max(1, len(sentences) // 5000))
    print(f"[INFO] FAISS 索引构建完成，nprobe 设定为 {index.nprobe}")

    duplicates = set()
    print("[INFO] 开始查找重复数据 ...")
    start_time = time.time()

    batch_size = 512
    for batch_start in range(0, len(sentences), batch_size):
        batch_end = min(batch_start + batch_size, len(sentences))
        batch_embeddings = embeddings[batch_start:batch_end]

        distances, indices = index.search(batch_embeddings, k=min(K_NEIGHBORS, len(sentences)))

        for i, (dists, inds) in enumerate(zip(distances, indices)):
            sent_idx = batch_start + i
            if sent_idx in duplicates:
                continue
            for j, dist in zip(inds, dists):
                if j == sent_idx or j in duplicates:
                    continue
                if dist > threshold:
                    duplicates.add(j)

        if batch_start % 5000 == 0:
            print(f"[PROGRESS] 重复查找进度: {batch_start}/{len(sentences)} ({len(duplicates)} 个重复项)")

    print(f"[INFO] 重复项查找完成，总耗时 {time.time() - start_time:.2f} 秒")
    return duplicates


def main():
    input_file = "mic/origin.csv"
    output_file = "mic/de_duplicate.csv"

    sentences, df = load_data(input_file)
    model = SentenceTransformer(MODEL_NAME)
    duplicates = find_duplicates(sentences, model)

    df['label'] = df.index.to_series().apply(lambda idx: 1 if idx in duplicates else 0)

    print(f"[INFO] 去重完成，原始数据 {len(df)} 条，去重后 {len(df[df['label'] == 0])} 条")
    df.to_csv(output_file, index=False)
    print(f"[INFO] 结果已保存至 {output_file}")


if __name__ == "__main__":
    main()
