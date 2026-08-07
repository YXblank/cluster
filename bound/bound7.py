import open3d as o3d
import numpy as np
import json
from collections import defaultdict
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
# 假设已有的 segments 列表
vertex_indices = [0]
def compute_distance(i, j, vertices):
    p1 = vertices[i]
    p2 = vertices[j]
    return np.linalg.norm(p1 - p2)
def custom_distance(x, y):
    # 比如根据某些自定义规则，计算 x 和 y 之间的差异
    dx = abs(x[0] - y[0])  # x 坐标的绝对差异
    dy = abs(x[1] - y[1])  # y 坐标的绝对差异
    return dx + dy  # 或者其他自定义的距离度量
# 读取 JSON 文件
with open('/home/xuyuan/pan/xybag/scene0072_02/scene0072_02.aggregation.json', 'r') as file:
    scene_data = json.load(file)

# 遍历 JSON 文件中的 segGroups，将每个 segGroup 的 segments 添加到 existing_segments 中
for seg_group in scene_data['segGroups']:
    vertex_indices.extend(seg_group['segments'])  # extend() 会将列表中的元素添加到 existing_segments 中

#with open('/home/xuyuan/pan/xybag/210cdbbd-9e8d-2832-87c1-a1a2c71cde80/semseg.v2.json', 'r') as file:
#    scene_data = json.load(file)
#for seg_group in scene_data['segGroups']:
#    vertex_indices.extend(seg_group['segments'])

# 1. 读取PLY文件
ply_file = "/home/xuyuan/pan/xybag/scene0072_02/scene0072_02_vh_clean_2.labels.ply"   # 替换成你的PLY文件路径
mesh = o3d.io.read_triangle_mesh(ply_file)

# 2. 提取顶点数据
vertices = np.asarray(mesh.vertices)

# 3. 假设你有一列顶点索引，你可以将它定义为一个列表或数组
 # 你想获取坐标的索引列表
distance_threshold =0.3 # 例如，设置阈值为 0.05
# 4. 获取这些顶点的坐标
vertex_indices = np.clip(vertex_indices, 0, len(vertices) - 1)
target_vertices = vertices[vertex_indices]  # 通过索引提取多个顶点的坐标
print("Target Vertices:")
print(target_vertices)
# 创建连接这些顶点的线段（边）
lines = []
connected_points_set = set()  # 用来存储有连接的点索引
point_line_count = defaultdict(int)  # 用于记录每个点连接了多少条线


# 标准化顶点数据
scaler = StandardScaler()
target_vertices_scaled = scaler.fit_transform(target_vertices)
dist_matrix = pairwise_distances(target_vertices, metric=custom_distance)
db = DBSCAN(eps=0.9, min_samples=85, metric='precomputed') # eps 是距离阈值，min_samples 是最小点数
labels = db.fit_predict(dist_matrix)

# 输出聚类结果
print("聚类标签：", labels)

z_min = -2.0  # z 轴下限
z_max = 1.5  # z 轴上限

for i in range(len(target_vertices)):
    for j in range(i + 1, len(target_vertices)):
        if labels[i] == labels[j] and labels[i] != -1:
            # 先检查两个点的 z 坐标是否在范围内
            z_i = target_vertices[i][2]  # 获取第 i 个点的 z 坐标
            z_j = target_vertices[j][2]  # 获取第 j 个点的 z 坐标

            if z_min <= z_i <= z_max and z_min <= z_j <= z_max:  # 判断 z 坐标是否在范围内
                distance = custom_distance(target_vertices[i], target_vertices[j])
                if distance < distance_threshold:
                    lines.append([i, j])  # 只有距离小于阈值时才连线
                    connected_points_set.add(i)  # 将有连接的点加入集合
                    connected_points_set.add(j)  # 将有连接的点加入集合
num_points = len(connected_points_set)
distance_matrix = np.zeros((num_points, num_points))

# 遍历所有连接边
for i, j in lines:
    if i < num_points and j < num_points:  # 检查索引是否有效
        distance = compute_distance(i, j, target_vertices)
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance  # 对称

# 绘制距离的 2D 热力图
plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Distance')
plt.title("Distance Heatmap of Connections")
plt.xlabel("Point Index")
plt.ylabel("Point Index")
plt.show()

for i, j in lines:
    if i < num_points and j < num_points:
        distance_matrix[i, j] += 1
        distance_matrix[j, i] += 1  # 对称

# 绘制连接边数量的 2D 热力图
plt.imshow(distance_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Number of Connections')
plt.title("Connection Count Heatmap")
plt.xlabel("Point Index")
plt.ylabel("Point Index")
plt.show()
n = len(labels)
clustering_matrix = np.zeros((n, n))

# 填充矩阵，如果两个点属于同一类，则设置为1，否则为0
for i in range(n):
    for j in range(i + 1, n):
        if labels[i] == labels[j] and labels[i] != -1:  # 同一类且不是噪音
            clustering_matrix[i, j] = 1
            clustering_matrix[j, i] = 1  # 对称矩阵

# 生成热力图
plt.figure(figsize=(8, 6))
sns.heatmap(clustering_matrix, cmap="Blues", annot=False, cbar=True)
plt.title("Clustering Heatmap between Vertices")
plt.xlabel("Vertex Index")
plt.ylabel("Vertex Index")
plt.show()

# 使用 NearestNeighbors 来计算每个点的 K 最近邻
#neighbors = NearestNeighbors(n_neighbors=4)  # 你可以选择合适的 k 值
#neighbors.fit(target_vertices)
#distances, indices = neighbors.kneighbors(target_vertices)

# 计算每个点到第 K 最近邻的距离（这里是 4 最近邻）
#k_distances = distances[:, -1]

# 可视化 K 距离（通常选择 K 为 min_samples 的值）
#plt.plot(np.sort(k_distances))
#plt.ylabel('Distance to 4th nearest neighbor')
#plt.xlabel('Points sorted by distance')
#plt.title('K-distance graph')
#plt.show()


# 创建新的点云对象存储这些目标顶点
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(target_vertices)

# 创建 LineSet 对象来表示这些连接
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(target_vertices)
line_set.lines = o3d.utility.Vector2iVector(lines)

# 设置线的颜色，方便查看
line_set.paint_uniform_color([0, 0, 0])  # 设置所有线的颜色为黑色

# 可视化只有有连接的点和它们之间的线段
o3d.visualization.draw_geometries([point_cloud, line_set])

