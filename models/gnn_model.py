# models/gnn_model.py (完整修复版)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from config import Config
import pickle


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.5)

        # 如果输出维度不等于输入维度，添加一个调整层
        if out_channels != in_channels:
            self.adjust_layer = torch.nn.Linear(out_channels, in_channels)
        else:
            self.adjust_layer = None

    def forward(self, x, edge_index, edge_weight=None):
        # 第一次图卷积
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二次图卷积
        x = self.conv2(x, edge_index, edge_weight)

        # 如果需要，调整输出维度
        if self.adjust_layer is not None:
            x = self.adjust_layer(x)

        return x


class GNNRecommender:
    def __init__(self, in_channels, hidden_channels, out_channels):
        # 确保输出维度与输入维度相同
        self.model = GCN(in_channels, hidden_channels, in_channels)  # 输出维度改为 in_channels
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.GNN_LEARNING_RATE
        )
        self.loss_fn = torch.nn.MSELoss()

    def train(self, graph_data, epochs=Config.GNN_EPOCHS):
        """训练GNN模型"""
        self.model.train()

        print(f"输入维度: {graph_data.x.size(1)}")

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # 前向传播
            out = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

            # 检查维度是否匹配
            if out.size(1) != graph_data.x.size(1):
                print(f"警告: 维度不匹配! 输入: {graph_data.x.size()}, 输出: {out.size()}")
                # 添加一个临时线性层来调整维度
                adjust_layer = torch.nn.Linear(out.size(1), graph_data.x.size(1)).to(out.device)
                out = adjust_layer(out)

            # 计算损失
            loss = self.loss_fn(out, graph_data.x)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    def get_embeddings(self, graph_data):
        """获取GNN生成的新嵌入向量"""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        return embeddings

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已从 {path} 加载")


def train_gnn_model():
    """训练GNN模型"""
    # 加载图数据
    try:
        with open(Config.GRAPH_DATA_PATH, 'rb') as f:
            graph_data = pickle.load(f)
    except:
        # 如果加载失败，重新构建图数据
        from models.graph_builder import build_and_save_graph
        graph_data = build_and_save_graph()

    # 初始化模型
    in_channels = graph_data.x.size(1)
    recommender = GNNRecommender(
        in_channels,
        Config.GNN_HIDDEN_CHANNELS,
        in_channels  # 输出维度与输入相同
    )

    # 简单训练几个epoch
    print("开始训练GNN模型...")
    recommender.train(graph_data, epochs=10)  # 减少训练轮数

    # 保存模型
    recommender.save_model(Config.GNN_MODEL_PATH)

    # 获取新嵌入向量
    new_embeddings = recommender.get_embeddings(graph_data)

    return new_embeddings, recommender