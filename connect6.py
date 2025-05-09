
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import random

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 假设 image 是输入图像，output_image 是输出图像，distance_threshold 是距离阈值
image = cv2.imread('/home/xuyuan/pan/data/global.png')
output_image = image.copy()
distance_threshold = 150  # 示例阈值
# 用于记录每次连接的颜色类别对和连接次数
color_pairs_count = defaultdict(int)
# 获取图像尺寸
height, width = image.shape[:2]

# 将图像像素转换为 (N, 3) 数组，用于聚类
pixels = image.reshape(-1, 3)  # 每个像素是 [R, G, B]
pixel_coords = [(x, y) for y in range(height) for x in range(width)]  # 对应坐标

# 过滤黑色和白色像素
valid_mask = ~((pixels[:, 0] == 0) & (pixels[:, 1] == 0) & (pixels[:, 2] == 0)) & \
             ~((pixels[:, 0] == 255) & (pixels[:, 1] == 255) & (pixels[:, 2] == 255))
valid_pixels = pixels[valid_mask]
valid_coords = np.array(pixel_coords)[valid_mask]

# 使用 K-means 聚类确定颜色类别
n_clusters = 20  # 可根据需要调整，或使用自动估计方法
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(valid_pixels)

# 输出不同颜色的个数
print(f"Number of distinct colors (clusters): {n_clusters}")

# 存储每种集群标签的像素坐标
color_pixels = defaultdict(list)
for idx, label in enumerate(cluster_labels):
    color_pixels[label].append(tuple(valid_coords[idx]))

# 定义颜色调色板，用于可视化每个集群（最多20种颜色）
cluster_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 128, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255)
]

# 可视化每个集群的像素点
for label in range(n_clusters):
    # 选择颜色：使用调色板或集群中心的RGB值
    color = cluster_colors[label % len(cluster_colors)]  # 循环使用颜色
    # 或者使用集群中心的RGB值（注释掉以下行以使用调色板）
    # color = tuple(kmeans.cluster_centers_[label].astype(int))
    for x, y in color_pixels[label]:
        cv2.circle(output_image, (x, y), 2, color, -1)  # 绘制半径为2的实心圆
# 为每种集群标签随机采样最多 max_samples 个像素（避免内存问题）
max_samples = 100  # 可根据需要调整
sampled_color_pixels = defaultdict(list)
for label in range(n_clusters):
    pixels = color_pixels[label]
    if len(pixels) > max_samples:
        sampled_color_pixels[label] = random.sample(pixels, max_samples)
    else:
        sampled_color_pixels[label] = pixels

# 存储已连接的颜色对，避免重复连接
connected_pairs = set()

# 遍历所有集群标签对
for label1 in range(n_clusters):
    for label2 in range(n_clusters):
        if label1 >= label2:  # 避免重复处理 (label1, label2) 和 (label2, label1)
            continue

        # 获取两种集群标签的采样像素坐标
        pixels1 = sampled_color_pixels[label1]
        pixels2 = sampled_color_pixels[label2]

        # 寻找距离最近的一对像素
        min_distance = float('inf')
        best_pair = None

        for x1, y1 in pixels1:
            for x2, y2 in pixels2:
                distance = euclidean_distance(x1, y1, x2, y2)
                if distance < distance_threshold and distance < min_distance:
                    min_distance = distance
                    best_pair = ((x1, y1), (x2, y2))

        # 如果找到有效对，绘制连接线
        if best_pair:
            (x1, y1), (x2, y2) = best_pair
            color_pair = tuple(sorted([label1, label2]))
            if color_pair not in connected_pairs:
                cv2.line(output_image, (x1, y1), (x2, y2), (255,255, 255), 3)
                connected_pairs.add(color_pair)
                # 记录连接次数（如果需要）
                color_pairs_count[color_pair] = 1  # 连接一次

# 保存或显示结果
# cv2.imwrite('output_image.jpg', output_image)
cv2.imshow('Output', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


