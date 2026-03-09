# config.py
class Config:
    # 模拟数据规模
    NUM_USERS = 1000
    NUM_CIRCLES = 100
    MAX_BEHAVIORS_PER_USER = 50

    # 兴趣标签池（5大类，40+子标签）
    INTEREST_TAGS = [
        # 文化休闲
        "书法", "绘画", "京剧", "剪纸", "茶道", "围棋", "民乐", "阅读",
        # 运动健康
        "太极", "广场舞", "健走", "按摩", "八段锦", "钓鱼", "登山",
        # 生活技艺
        "家常菜", "烘焙", "编织", "插花", "园艺", "维修", "理财",
        # 社交娱乐
        "合唱", "棋牌", "旅游", "摄影", "短视频", "带孙辈", "宠物",
        # 学习提升
        "智能手机", "中医养生", "书法班", "摄影班", "历史", "英语"
    ]

    # 地理坐标范围（以上海市大致范围为例）
    GEO_RANGE = {
        "min_lat": 31.0,
        "max_lat": 31.4,
        "min_lng": 121.2,
        "max_lng": 121.8
    }

    # 模型相关
    SENTENCE_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # 一个支持中文的小模型
    # sentence-transformers 模型列表:
    # https://www.sbert.net/docs/pretrained_models.html

    # 文件路径
    USER_DATA_PATH = "data/users.csv"
    CIRCLE_DATA_PATH = "data/circles.csv"
    BEHAVIOR_DATA_PATH = "data/behaviors.csv"
    USER_EMBEDDING_PATH = "data/user_embeddings.npy"
    CIRCLE_EMBEDDING_PATH = "data/circle_embeddings.npy"

    # ... [保留原有配置] ...

    # 阶段二：多源特征与GNN配置
    # 特征融合权重
    FEATURE_WEIGHTS = {
        'text_similarity': 0.6,
        'geo_distance': 0.3,
        'popularity': 0.1
    }

    # 地理距离计算参数（单位：公里）
    MAX_GEO_DISTANCE_KM = 50

    # config.py (GNN部分)
    # GNN训练参数
    GNN_HIDDEN_CHANNELS = 128
    GNN_OUTPUT_CHANNELS = 64  # 这个值可能不会被使用，因为我们在代码中强制输出维度与输入相同
    GNN_LEARNING_RATE = 0.01
    GNN_EPOCHS = 50

    # 文件路径
    USER_GEO_EMBEDDING_PATH = "data/user_geo_embeddings.npy"
    CIRCLE_GEO_EMBEDDING_PATH = "data/circle_geo_embeddings.npy"
    GRAPH_DATA_PATH = "data/graph_data.pt"
    GNN_MODEL_PATH = "models/gnn_model.pth"
