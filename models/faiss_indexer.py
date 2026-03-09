# models/faiss_indexer.py
import faiss
import numpy as np
import pandas as pd
from config import Config
from feature_engineering.multi_feature_fusion import MultiFeatureFusion


class FaissIndexer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = None
        self.circle_ids = None
        self.circle_data = None
        self.fusion = MultiFeatureFusion()

    def build_index(self, embeddings, circle_ids, circle_data):
        # 只使用文本特征部分进行Faiss索引（前n维）
        text_embedding_dim = embeddings.shape[1] - 3  # 减去地理(2)和热度(1)特征
        text_embeddings = embeddings[:, :text_embedding_dim]

        """构建Faiss索引"""
        # 归一化向量，以便使用内积（余弦相似度）进行搜索
        faiss.normalize_L2(embeddings)

        # 使用IndexFlatIP（内积）索引，效果等同于余弦相似度（因为向量已归一化）
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.circle_ids = circle_ids
        self.circle_data = circle_data
        print(f"Faiss索引构建完成，共 {self.index.ntotal} 个向量。")

    def search_similar(self, query_embedding, user_geo_hash, k=5):
        """搜索最相似的k个圈子"""
        if self.index is None:
            raise ValueError("请先构建索引!")

            # 打印调试信息
        print(f"用户GeoHash: {user_geo_hash}")

        # 只使用文本特征部分进行Faiss初步搜索
        text_embedding_dim = self.index.d
        query_text_embedding = query_embedding[:text_embedding_dim]

        # 归一化查询向量
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        # 执行初步搜索（获取更多候选结果）
        candidate_k = min(50, self.index.ntotal)  # 获取50个候选
        distances, indices = self.index.search(
            query_text_embedding.astype(np.float32).reshape(1, -1),
            candidate_k
        )

        # 多特征融合重排序
        results = []
        for i, (text_similarity, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # 无效索引跳过
                continue

            circle_id = self.circle_ids[idx]
            circle_info = self.circle_data[self.circle_data['circle_id'] == circle_id].iloc[0]

            # 获取地理距离
            geo_distance = self.fusion.calculate_geo_distance(
                user_geo_hash, circle_info['geo_hash']
            )

            # 打印每个圈子的地理信息
            print(f"圈子 {circle_info['name']} GeoHash: {circle_info['geo_hash']}, 距离: {geo_distance:.2f}km")

            # 获取热度值（归一化）
            popularity = circle_info['member_count'] / self.circle_data['member_count'].max()


            # 计算最终得分
            final_score = self.fusion.calculate_final_score(
                text_similarity, geo_distance, popularity
            )

            results.append({
                "circle_id": circle_id,
                "text_similarity": float(text_similarity),
                "geo_distance": float(geo_distance),
                "popularity": float(popularity),
                "final_score": float(final_score)
            })

        # 按最终得分排序
        results.sort(key=lambda x: x['final_score'], reverse=True)

        # 返回前k个结果
        for i, result in enumerate(results[:k]):
            result['rank'] = i + 1

        return results[:k]

    def save_index(self, file_path):
        """保存索引到文件"""
        if self.index is None:
            raise ValueError("没有可保存的索引!")
        faiss.write_index(self.index, file_path)
        print(f"索引已保存至: {file_path}")

    def load_index(self, file_path, circle_ids):
        """从文件加载索引"""
        self.index = faiss.read_index(file_path)
        self.circle_ids = circle_ids
        print(f"索引已从 {file_path} 加载，共 {self.index.ntotal} 个向量。")


def demo_faiss_search():
    """演示Faiss搜索功能"""
    # 加载数据
    # 加载数据
    circles_df = pd.read_csv(Config.CIRCLE_DATA_PATH)
    users_df = pd.read_csv(Config.USER_DATA_PATH)

    # 加载融合后的特征向量
    try:
        circle_embeddings = np.load(Config.CIRCLE_EMBEDDING_PATH)
        user_embeddings = np.load(Config.USER_EMBEDDING_PATH)
    except FileNotFoundError:
        from feature_engineering.feature_extractor import run_feature_extraction
        _, circle_embeddings, _, _ = run_feature_extraction()

    # 构建索引
    dimension = circle_embeddings.shape[1]
    indexer = FaissIndexer(dimension)
    indexer.build_index(circle_embeddings, circles_df['circle_id'].values, circles_df)

    # 随机选择一个用户进行演示
    import random
    random_user = users_df.iloc[random.randint(0, len(users_df) - 1)]
    random_user_embedding = user_embeddings[random_user.name]  # 获取对应用户的向量

    # 搜索相似圈子
    similar_circles = indexer.search_similar(
        random_user_embedding, random_user['geo_hash'], k=3
    )

    print(f"\n为用户 {random_user['user_id']} (兴趣: {random_user['interests']}) 推荐的圈子:")
    for result in similar_circles:
        circle_info = circles_df[circles_df['circle_id'] == result['circle_id']].iloc[0]
        print(f"{result['rank']}. {circle_info['name']}")
        print(f"   最终得分: {result['final_score']:.4f} = "
              f"文本({result['text_similarity']:.4f}) + "
              f"地理({1 - result['geo_distance'] / Config.MAX_GEO_DISTANCE_KM:.4f}) + "
              f"热度({result['popularity']:.4f})")
        print(f"   距离: {result['geo_distance']:.2f}km, 成员数: {circle_info['member_count']}")
        print(f"   标签: {circle_info['tags']}\n")


if __name__ == "__main__":
    demo_faiss_search()