# feature_engineering/feature_extractor.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config
import joblib  # 用于保存特征向量
from .multi_feature_fusion import MultiFeatureFusion  # 新增导入

class FeatureExtractor:
    def __init__(self):
        # 加载预训练的多语言句子Transformer模型
        self.model = SentenceTransformer(Config.SENTENCE_MODEL_NAME)
        self.fusion = MultiFeatureFusion()  # 新增多特征融合器
        print(f"模型 '{Config.SENTENCE_MODEL_NAME}' 加载成功。")

    def extract_text_embeddings(self, texts):
        """将文本列表转换为向量"""
        # 如果输入是字符串，转换为列表
        if isinstance(texts, str):
            texts = [texts]
        # 模型编码，得到numpy数组
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def generate_user_embeddings(self, users_df):
        """为每个用户生成特征向量"""
        print("正在生成用户特征向量...")
        user_embeddings = []
        for user in users_df.itertuples():
            # 将用户的兴趣标签组合成一段描述文本
            text = f"用户兴趣: {user.interests}"
            embedding = self.extract_text_embeddings(text)
            user_embeddings.append(embedding)

        user_embeddings = np.vstack(user_embeddings)
        np.save(Config.USER_EMBEDDING_PATH, user_embeddings)
        print(f"用户特征向量已保存至: {Config.USER_EMBEDDING_PATH}")
        return user_embeddings

    def generate_circle_embeddings(self, circles_df):
        """为每个圈子生成特征向量"""
        print("正在生成圈子特征向量...")
        circle_embeddings = []
        for circle in circles_df.itertuples():
            # 将圈子的名称、描述和标签组合成一段描述文本
            text = f"圈子: {circle.name}. 描述: {circle.description}. 标签: {circle.tags}"
            embedding = self.extract_text_embeddings(text)
            circle_embeddings.append(embedding)

        circle_embeddings = np.vstack(circle_embeddings)
        np.save(Config.CIRCLE_EMBEDDING_PATH, circle_embeddings)
        print(f"圈子特征向量已保存至: {Config.CIRCLE_EMBEDDING_PATH}")
        return circle_embeddings

    def generate_fused_embeddings(self, users_df, circles_df):
        """生成融合多源特征的嵌入向量"""
        print("正在生成融合多源特征的用户向量...")
        user_text_embeddings = np.load(Config.USER_EMBEDDING_PATH)
        user_fused_embeddings = self.fusion.create_fused_embeddings(
            users_df, circles_df, user_text_embeddings, "user"
        )

        print("正在生成融合多源特征的圈子向量...")
        circle_text_embeddings = np.load(Config.CIRCLE_EMBEDDING_PATH)
        circle_fused_embeddings = self.fusion.create_fused_embeddings(
            users_df, circles_df, circle_text_embeddings, "circle"
        )

        return user_fused_embeddings, circle_fused_embeddings


def run_feature_extraction():
    """运行特征提取流程"""
    # 加载模拟数据
    users_df = pd.read_csv(Config.USER_DATA_PATH)
    circles_df = pd.read_csv(Config.CIRCLE_DATA_PATH)

    # 初始化特征提取器
    extractor = FeatureExtractor()

    # 生成文本嵌入向量（如果尚未生成）
    try:
        user_text_embeddings = np.load(Config.USER_EMBEDDING_PATH)
        circle_text_embeddings = np.load(Config.CIRCLE_EMBEDDING_PATH)
    except FileNotFoundError:
        user_text_embeddings = extractor.generate_user_embeddings(users_df)
        circle_text_embeddings = extractor.generate_circle_embeddings(circles_df)

    # 生成融合多源特征的向量
    user_fused_embeddings, circle_fused_embeddings = extractor.generate_fused_embeddings(users_df, circles_df)

    return user_fused_embeddings, circle_fused_embeddings, users_df, circles_df

if __name__ == "__main__":
    run_feature_extraction()