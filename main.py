# main.py (更新)
from config import Config
from data_generation.mock_data_generator import generate_all_data
from feature_engineering.feature_extractor import run_feature_extraction
from models.faiss_indexer import demo_faiss_search
from models.graph_builder import build_and_save_graph
from models.gnn_model import train_gnn_model
import pandas as pd


def main():
    print("=== 阶段二：多源特征融合与GNN探索 ===\n")

    # 第一步：生成模拟数据（如果尚未生成）
    print("步骤1/4：检查模拟数据...")
    try:
        users_df = pd.read_csv(Config.USER_DATA_PATH)
        circles_df = pd.read_csv(Config.CIRCLE_DATA_PATH)
        behaviors_df = pd.read_csv(Config.BEHAVIOR_DATA_PATH)
        print("模拟数据已存在，跳过生成。")
    except FileNotFoundError:
        print("模拟数据不存在，开始生成...")
        users_df, circles_df, behaviors_df = generate_all_data()

    # 第二步：特征提取与融合
    print("\n步骤2/4：特征提取与融合...")
    user_embeddings, circle_embeddings, users_df, circles_df = run_feature_extraction()

    # 第三步：演示多特征融合推荐
    print("\n步骤3/4：演示多特征融合推荐...")
    demo_faiss_search()

    # 第四步：构建图数据并训练GNN
    print("\n步骤4/4：构建图数据并训练GNN...")
    try:
        graph_data = build_and_save_graph()
        gnn_embeddings, gnn_model = train_gnn_model()
        print("GNN训练完成！")
    except Exception as e:
        print(f"GNN训练过程中出现错误: {e}")
        print("跳过GNN训练，继续执行...")

    print("\n=== 阶段二完成！ ===")
    print("下一步：")
    print("1. 使用GNN生成的新嵌入向量替换原有向量")
    print("2. 进行A/B测试比较推荐效果")
    print("3. 模型服务化部署")


if __name__ == "__main__":
    main()