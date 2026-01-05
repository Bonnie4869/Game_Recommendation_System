import pickle
import os


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pickle
import warnings
import os  # 用于检查文件是否存在

warnings.filterwarnings("ignore")


class SVDRecommender:
    """
    使用截断奇异值分解 (Truncated SVD) 实现的矩阵分解推荐系统。
    """

    def __init__(self, n_factors=50):
        """
        初始化推荐器。
        :param n_factors: 潜在因子的数量 (K)。
        """
        self.K = n_factors
        self.user_mapper = None
        self.app_mapper = None
        self.user_inv_mapper = None
        self.app_inv_mapper = None
        self.R = None  # 稀疏评分矩阵
        self.all_user_predicted_ratings = None  # 预测评分矩阵

    def _prepare_data(self, data_df):
        """
        数据预处理：评分定义、ID映射和索引创建。
        """
        # 1. 评分定义：对数转换游玩时长
        data_df["rating"] = np.log1p(data_df["hours"])
        data_df = data_df[data_df["rating"] > 0]

        # ***** 关键修改：强制转换为字符串 *****
        data_df["user_id"] = data_df["user_id"].astype(str)
        data_df["app_id"] = data_df["app_id"].astype(str)

        # 2. ID 映射到整数索引
        data_df["user_id"] = data_df["user_id"].astype("category")
        data_df["app_id"] = data_df["app_id"].astype("category")

        self.user_mapper = {
            user: idx for idx, user in enumerate(data_df["user_id"].cat.categories)
        }
        self.app_mapper = {
            app: idx for idx, app in enumerate(data_df["app_id"].cat.categories)
        }
        self.user_inv_mapper = {idx: user for user, idx in self.user_mapper.items()}
        self.app_inv_mapper = {idx: app for app, idx in self.app_mapper.items()}

        data_df["user_index"] = data_df["user_id"].cat.codes
        data_df["app_index"] = data_df["app_id"].cat.codes

        return data_df

    def fit(self, file_path):
        """
        训练模型：加载数据，构建稀疏矩阵，并执行 SVD 分解。
        """
        print(f"--- 步骤 1: 加载数据并准备 ---")
        data_df = pd.read_csv(file_path)
        data_df = self._prepare_data(data_df)

        print(f"用户总数: {len(self.user_mapper)}, 游戏总数: {len(self.app_mapper)}")

        # 2. 构建稀疏用户-物品矩阵
        self.R = csr_matrix(
            (data_df["rating"], (data_df["user_index"], data_df["app_index"]))
        )

        # 3. 训练模型 - 稀疏矩阵SVD (svds)
        print(f"--- 步骤 2: 训练 SVD 模型 (K={self.K}) ---")
        try:
            U, sigma, Vt = svds(self.R, k=self.K)
        except Exception as e:
            print(f"SVD 运行失败，错误: {e}")
            return

        # 4. 重构预测评分矩阵
        sigma = np.diag(sigma)
        self.all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        print("--- 模型训练完成 ---")

    def recommend(self, user_id, num_recommendations=10):
        """
        为指定用户生成 Top-N 推荐。
        """
        if self.all_user_predicted_ratings is None:
            return "错误：模型尚未训练。请先调用 fit() 方法。"

        # 1. 获取用户的索引和预测评分向量
        try:
            user_index = self.user_mapper[user_id]
        except KeyError:
            return f"用户ID {user_id} 不在训练数据集中。"

        predicted_ratings_vector = self.all_user_predicted_ratings[user_index]

        # 2. 获取用户已玩过的游戏索引 (从稀疏矩阵中查找非零项)
        played_games_indices = self.R[user_index, :].nonzero()[1]

        # 3. 屏蔽已玩过的游戏
        temp_ratings = predicted_ratings_vector.copy()
        temp_ratings[played_games_indices] = -np.inf

        # 4. 找到评分最高的 N 个游戏的索引
        top_game_indices = temp_ratings.argsort()[::-1][:num_recommendations]

        # 5. 映射回原始 app_id 并提取评分
        recommendations = []
        for app_index in top_game_indices:
            app_id = self.app_inv_mapper[app_index]
            predicted_score = predicted_ratings_vector[app_index]
            recommendations.append((app_id, predicted_score))

        return recommendations

    def save_model(self, filename="svd_recommender.pkl"):
        """
        将 SVDRecommender 实例保存到 .pkl 文件。
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        print(f"\n✅ 模型已成功保存为: {filename}")


import pickle
import os


# 调用模型
def load_model(filename="game_svd_recommender.pkl"):
    """
    从 .pkl 文件加载 SVDRecommender 实例。
    """
    if not os.path.exists(filename):
        print(f"错误：文件 {filename} 不存在。")
        return None

    with open(filename, "rb") as file:
        model = pickle.load(file)
    print(f"\n🎉 模型已成功从 {filename} 加载。")
    return model


loaded_model = load_model("game_svd_model.pkl")

user_id_str = "7056396"
recommendations = loaded_model.recommend(user_id_str)

print(f"\n推荐结果 for User {user_id_str}:")
if isinstance(recommendations, str):
    print(recommendations)
else:
    for app_id, predicted_rating in recommendations:
        print(f"Game ID: {app_id:<10} | 预测评分: {predicted_rating:.4f}")
