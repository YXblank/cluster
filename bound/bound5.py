import open3d as o3d
import numpy as np
import json

# 假设已有的 segments 列表
vertex_indices = [1]

# 读取 JSON 文件
with open('/home/xuyuan/pan/xybag/569d8f15-72aa-2f24-8b0a-2a9bbe4c9a3d/semseg.v2.json', 'r') as file:
    scene_data = json.load(file)

# 遍历 JSON 文件中的 segGroups，将每个 segGroup 的 segments 添加到 existing_segments 中
for seg_group in scene_data['segGroups']:
  segments = seg_group['segments']
  if segments:  # 确保 segments 数组非空
      vertex_indices.append(segments[0])  # 将每个 segments 数组中的第一个元素添加到 vertex_indices
# 1. 读取PLY文件
ply_file = "/home/xuyuan/pan/xybag/569d8f15-72aa-2f24-8b0a-2a9bbe4c9a3d/labels.instances.annotated.v2.ply"   # 替换成你的PLY文件路径
mesh = o3d.io.read_triangle_mesh(ply_file)

# 2. 提取顶点数据
vertices = np.asarray(mesh.vertices)

# 3. 假设你有一列顶点索引，你可以将它定义为一个列表或数组
 # 你想获取坐标的索引列表

# 4. 获取这些顶点的坐标
vertex_indices = np.clip(vertex_indices, 0, len(vertices) - 1)
target_vertices = vertices[vertex_indices]
 # 通过索引提取多个顶点的坐标
print("Target Vertices:")
print(target_vertices)

# 5. 创建新的点云对象存储这些目标顶点
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(target_vertices)

# 6. 创建连接这些顶点的线段（边）
lines = []
for i in range(len(target_vertices) - 1):
    lines.append([i, i + 1])  # 连接相邻的点

# 如果需要一个封闭的环形连接最后一个点与第一个点
lines.append([len(target_vertices) - 1, 0])

# 创建 LineSet 对象来表示这些连接
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(target_vertices)
line_set.lines = o3d.utility.Vector2iVector(lines)

# 7. 为了更好区分，给点云和线段上色
#point_cloud.paint_uniform_color([0, 1, 0])  # 绿色顶点
line_set.paint_uniform_color([0, 0, 0])    # 蓝色线段

# 8. 可视化：单独生成点和边的图形
o3d.visualization.draw_geometries([mesh, point_cloud, line_set])
o3d.visualization.draw_geometries([point_cloud, line_set])
