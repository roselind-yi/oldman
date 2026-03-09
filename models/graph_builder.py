# models/graph_builder.py
import pickle

import pandas as pd
import torch
from torch_geometric.data import Data
from config import Config
import numpy as np


class GraphBuilder:
    def __init__(self):
        pass

    def build_user_circle_graph(self, users_df, circles_df, behaviors_df):
        """构建用户-圈子异质图"""
        # 创建节点映射
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users_df['user_id'])}
        circle_id_to_idx = {circle_id: idx + len(users_df) for idx, circle_id in enumerate(circles_df['circle_id'])}

        # 准备节点特征
        user_embeddings = torch.tensor(np.load(Config.USER_EMBEDDING_PATH), dtype=torch.float)
        circle_embeddings = torch.tensor(np.load(Config.CIRCLE_EMBEDDING_PATH), dtype=torch.float)
        x = torch.cat([user_embeddings, circle_embeddings], dim=0)

        # 准备边信息（用户-圈子交互）
        edge_indices = []
        edge_attrs = []  # 边属性：交互权重

        # 定义交互类型权重
        event_weights = {'view': 0.2, 'like': 0.5, 'share': 0.7, 'join': 1.0, 'post': 1.2}

        for behavior in behaviors_df.itertuples():
            if behavior.event_type not in event_weights:
                continue

            user_idx = user_id_to_idx.get(behavior.user_id, -1)
            circle_idx = circle_id_to_idx.get(behavior.target_id, -1)

            if user_idx != -1 and circle_idx != -1:
                edge_indices.append([user_idx, circle_idx])
                edge_attrs.append([event_weights[behavior.event_type]])

        # 转换为PyG需要的格式
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # 创建图数据对象
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 保存图数据 - 使用pickle以确保兼容性
        with open(Config.GRAPH_DATA_PATH, 'wb') as f:
            pickle.dump(graph_data, f)

        print(f"图数据构建完成，包含 {x.size(0)} 个节点和 {edge_index.size(1)} 条边")

        return graph_data

def build_and_save_graph():
    """构建并保存图数据"""
    # 加载数据
    users_df = pd.read_csv(Config.USER_DATA_PATH)
    circles_df = pd.read_csv(Config.CIRCLE_DATA_PATH)
    behaviors_df = pd.read_csv(Config.BEHAVIOR_DATA_PATH)

    # 构建图
    builder = GraphBuilder()
    graph_data = builder.build_user_circle_graph(users_df, circles_df, behaviors_df)

    return graph_data