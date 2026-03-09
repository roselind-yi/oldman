# feature_engineering/multi_feature_fusion.py
import numpy as np
import pandas as pd
from geohash2 import decode
from config import Config


class MultiFeatureFusion:
    def __init__(self):
        pass

    @staticmethod
    def calculate_geo_distance(geo_hash1, geo_hash2):
        """计算两个GeoHash之间的地理距离（公里）"""
        try:
            # 确保GeoHash长度一致且有效
            if not geo_hash1 or not geo_hash2 or len(geo_hash1) != len(geo_hash2):
                return Config.MAX_GEO_DISTANCE_KM

            # 解码GeoHash - 注意：decode返回的是字符串元组，需要转换为浮点数
            lat1_str, lon1_str = decode(geo_hash1)
            lat2_str, lon2_str = decode(geo_hash2)

            # 将经纬度转换为弧度
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

            # Haversine公式计算距离
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # 地球平均半径，单位公里
            distance = c * r

            # 确保距离不超过最大值
            return min(distance, Config.MAX_GEO_DISTANCE_KM)
        except Exception as e:
            print(f"地理距离计算错误: {e}, GeoHash1: {geo_hash1}, GeoHash2: {geo_hash2}")
            return Config.MAX_GEO_DISTANCE_KM  # 计算失败返回最大距离

    @staticmethod
    def normalize_feature(feature_values):
        """归一化特征值到[0,1]范围"""
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        if max_val == min_val:
            return np.ones_like(feature_values)
        return (feature_values - min_val) / (max_val - min_val)

    def create_fused_embeddings(self, users_df, circles_df, text_embeddings, embedding_type="user"):
        """
        创建融合多源特征的嵌入向量
        embedding_type: "user" 或 "circle"
        """
        if embedding_type == "user":
            entities_df = users_df
            geo_embedding_path = Config.USER_GEO_EMBEDDING_PATH
        else:
            entities_df = circles_df
            geo_embedding_path = Config.CIRCLE_GEO_EMBEDDING_PATH

        # 1. 提取并归一化热度特征（仅对圈子）
        if embedding_type == "circle":
            popularity_features = self.normalize_feature(entities_df['member_count'].values)
            popularity_features = popularity_features.reshape(-1, 1)
        else:
            popularity_features = np.zeros((len(entities_df), 1))

        # 2. 提取地理特征（GeoHash -> 经纬度 -> 2D向量）
        geo_features = []
        for geo_hash in entities_df['geo_hash']:
            try:
                lat, lon = decode(geo_hash)
                geo_features.append([lat, lon])
            except:
                geo_features.append([0, 0])  # 无效地理编码

        geo_features = np.array(geo_features, dtype=np.float64)

        # 归一化地理坐标
        lat_min, lat_max = Config.GEO_RANGE["min_lat"], Config.GEO_RANGE["max_lat"]
        lon_min, lon_max = Config.GEO_RANGE["min_lng"], Config.GEO_RANGE["max_lng"]

        geo_features[:, 0] = (geo_features[:, 0] - lat_min) / (lat_max - lat_min)  # 纬度归一化
        geo_features[:, 1] = (geo_features[:, 1] - lon_min) / (lon_max - lon_min)  # 经度归一化

        # 3. 融合所有特征 [文本特征, 地理特征, 热度特征]
        fused_embeddings = np.hstack([
            text_embeddings,  # 文本语义特征
            geo_features,  # 地理特征
            popularity_features  # 热度特征
        ])

        # 保存地理特征（后续用于距离计算）
        np.save(geo_embedding_path, geo_features)

        print(f"{embedding_type}特征融合完成，最终维度: {fused_embeddings.shape}")
        return fused_embeddings

    def calculate_final_score(self, text_similarity, geo_distance, popularity, weights=None):
        """计算最终推荐得分"""
        if weights is None:
            weights = Config.FEATURE_WEIGHTS

        # 将地理距离转换为相似度得分（距离越近，得分越高）
        geo_similarity = 1 - min(geo_distance / Config.MAX_GEO_DISTANCE_KM, 1.0)

        # 加权综合得分
        final_score = (
                weights['text_similarity'] * text_similarity +
                weights['geo_distance'] * geo_similarity +
                weights['popularity'] * popularity
        )

        return final_score