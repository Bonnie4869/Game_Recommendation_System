import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "games_with_cluster_and_desc.csv")
output_path = os.path.join(script_dir, "game_similarities.csv")

df = pd.read_csv(input_path, encoding='latin1')
df['app_id'] = df['app_id'].astype(str)  
print("Successfully loaded games_with_cluster_and_desc.csv")

print("loading Sentence-BERT")
model = SentenceTransformer('all-MiniLM-L6-v2')

onehot_path = os.path.join(script_dir, "one_hot_encoded_genres.csv")
df_onehot = pd.read_csv(onehot_path, encoding='latin1', header=None)
n_cols = df_onehot.shape[1]
df_onehot.columns = ['app_id', 'name'] + [f'genre_{i}' for i in range(n_cols - 2)]
df_onehot['app_id'] = df_onehot['app_id'].astype(str) 

results = []


for cluster_id, group in df.groupby('cluster'):
    print(f"处理 Cluster {cluster_id} ({len(group)} 款游戏)...")
    if len(group) < 2:
        continue

    descriptions = group['description'].fillna("").astype(str).tolist()
    app_ids = group['app_id'].astype(str).tolist()

    embeddings = model.encode(descriptions, show_progress_bar=False)
    text_sim_matrix = cosine_similarity(embeddings)

  
    onehot_group = df_onehot[df_onehot['app_id'].isin(app_ids)]
    onehot_group = onehot_group.set_index('app_id').reindex(app_ids).reset_index()
    X_genre = onehot_group.iloc[:, 2:].values


    genre_sim_matrix = cosine_similarity(X_genre)
    combined_sim_matrix = 0.5 * genre_sim_matrix + 0.5 * text_sim_matrix

    n = len(app_ids)
    for i in range(n):
        for j in range(i + 1, n):
            results.append({
                'app_id_1': app_ids[i],
                'app_id_2': app_ids[j],
                'cluster': cluster_id,
                'genre_similarity': round(genre_sim_matrix[i, j], 4),
                'text_similarity': round(text_sim_matrix[i, j], 4),
                'combined_similarity': round(combined_sim_matrix[i, j], 4)
            })


result_df = pd.DataFrame(results)
result_df.to_csv(output_path, index=False)
print(f" similarities has been saved to: {output_path}")
print(result_df.head())