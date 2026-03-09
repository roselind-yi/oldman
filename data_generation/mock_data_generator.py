# data_generation/mock_data_generator.py
import os

import pandas as pd
import numpy as np
from faker import Faker
import geohash2
from datetime import datetime, timedelta
import random
from config import Config

fake = Faker('zh_CN')  # 生成中文假数据


def generate_users(num_users):

    """生成模拟用户数据"""
    users = []
    for i in range(num_users):
        lat = np.random.uniform(Config.GEO_RANGE["min_lat"], Config.GEO_RANGE["max_lat"])
        lng = np.random.uniform(Config.GEO_RANGE["min_lng"], Config.GEO_RANGE["max_lng"])
        geo_hash = geohash2.encode(lat, lng, precision=8)

        # 每个用户随机分配3-6个兴趣标签
        interests = random.sample(Config.INTEREST_TAGS, k=random.randint(3, 6))

        users.append({
            "user_id": f"user_{i:06d}",
            "name": fake.name(),
            "age": random.randint(45, 85),
            "geo_hash": geo_hash,
            "interests": "|".join(interests),  # 用竖线分隔多个标签
            "created_at": fake.date_time_between(start_date="-2y", end_date="now")
        })
    return pd.DataFrame(users)


def generate_circles(num_circles):
    """生成模拟圈子数据"""
    circles = []
    for i in range(num_circles):
        lat = np.random.uniform(Config.GEO_RANGE["min_lat"], Config.GEO_RANGE["max_lat"])
        lng = np.random.uniform(Config.GEO_RANGE["min_lng"], Config.GEO_RANGE["max_lng"])
        geo_hash = geohash2.encode(lat, lng, precision=8)

        # 为每个圈子分配1-3个主题标签
        circle_tags = random.sample(Config.INTEREST_TAGS, k=random.randint(1, 3))
        name_prefix = random.choice(["快乐", "幸福", "温馨", "活力", "智慧"])

        circles.append({
            "circle_id": f"circle_{i:05d}",
            "name": f"{name_prefix}{circle_tags[0]}圈",
            "description": f"欢迎热爱{''.join(circle_tags)}的朋友加入！",
            "tags": "|".join(circle_tags),
            "geo_hash": geo_hash,
            "member_count": random.randint(5, 300),
            "created_at": fake.date_time_between(start_date="-1y", end_date="now")
        })
    return pd.DataFrame(circles)


def generate_behavior_logs(users, circles, max_per_user):
    """生成用户行为日志"""
    behaviors = []
    event_types = ["view", "join", "like", "share", "post"]

    for user in users.itertuples():
        num_behaviors = random.randint(5, max_per_user)
        user_circles = random.sample(list(circles['circle_id']), k=min(10, len(circles)))  # 用户可能交互的圈子

        for i in range(num_behaviors):
            circle = random.choice(user_circles)
            event_time = fake.date_time_between(start_date=user.created_at, end_date="now")

            behaviors.append({
                "log_id": f"log_{len(behaviors):08d}",
                "user_id": user.user_id,
                "event_type": random.choice(event_types),
                "target_id": circle,
                "timestamp": event_time,
                "content": ""  # 模拟内容，可为空
            })
    return pd.DataFrame(behaviors)


def generate_all_data():
    """生成所有模拟数据并保存到CSV"""
    # 新增：创建data目录（如果不存在）
    if not os.path.exists("data"):
        os.makedirs("data")
        print("已创建data目录")

    print("正在生成用户数据...")
    users_df = generate_users(Config.NUM_USERS)
    users_df.to_csv(Config.USER_DATA_PATH, index=False, encoding='utf-8-sig')

    print("正在生成圈子数据...")
    circles_df = generate_circles(Config.NUM_CIRCLES)
    circles_df.to_csv(Config.CIRCLE_DATA_PATH, index=False, encoding='utf-8-sig')

    print("正在生成用户行为数据...")
    behaviors_df = generate_behavior_logs(users_df, circles_df, Config.MAX_BEHAVIORS_PER_USER)
    behaviors_df.to_csv(Config.BEHAVIOR_DATA_PATH, index=False, encoding='utf-8-sig')

    print(f"模拟数据生成完成！\n用户数: {len(users_df)}\n圈子数: {len(circles_df)}\n行为记录数: {len(behaviors_df)}")
    return users_df, circles_df, behaviors_df


if __name__ == "__main__":
    generate_all_data()