import open3d as o3d
import numpy as np
import json

# 假设已有的 segments 列表
vertex_indices = [1]

# 读取 JSON 文件
with open('/home/xuyuan/pan/xybag/scene0106_01.aggregation.json', 'r') as file:
    scene_data = json.load(file)

# 遍历 JSON 文件中的 segGroups，将每个 segGroup 的 segments 添加到 existing_segments 中
for seg_group in scene_data['segGroups']:
    vertex_indices.extend(seg_group['segments'])  # extend() 会将列表中的元素添加到 existing_segments 中
# 1. 读取PLY文件
ply_file = "/home/xuyuan/pan/xybag/scene0106_01_vh_clean_2.labels.ply"   # 替换成你的PLY文件路径
mesh = o3d.io.read_triangle_mesh(ply_file)

# 2. 提取顶点数据
vertices = np.asarray(mesh.vertices)

# 3. 假设你有一列顶点索引，你可以将它定义为一个列表或数组
 # 你想获取坐标的索引列表
distance_threshold = 0.3# 例如，设置阈值为 0.05
# 4. 获取这些顶点的坐标
target_vertices = vertices[vertex_indices]  # 通过索引提取多个顶点的坐标
print("Target Vertices:")
print(target_vertices)
lines = []
# 5. 创建新的点云对象存储这些目标顶点
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(target_vertices)

for i in range(len(target_vertices)):
    for j in range(i + 1, len(target_vertices)):  # 只考虑 i < j 的组合，避免重复计算
        # 计算两个点之间的欧氏距离
        distance = np.linalg.norm(target_vertices[i] - target_vertices[j])
        if distance < distance_threshold:
            lines.append([i, j])  # 只有距离小于阈值时才连线
# 创建 LineSet 对象来表示这些连接
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(target_vertices)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.paint_uniform_color([0, 0, 0])  # 设置所有线的颜色为黑色

# 可视化点云和连接的线段
o3d.visualization.draw_geometries([point_cloud, line_set])

