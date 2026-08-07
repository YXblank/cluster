import pyvista as pv
import numpy as np


import open3d as o3d


# 1. 读取PLY文件
ply_file = "/home/xuyuan/pan/3RScan/0ad2d3a1-79e2-2212-9b99-a96495d9f7fe/labels.instances.annotated.v2.ply"  # 替换成你的PLY文件路径
mesh = o3d.io.read_triangle_mesh(ply_file)

# 2. 提取顶点数据
vertices = np.asarray(mesh.vertices)

# 3. 假设你要标记第N个顶点，设置N的值（比如第5个顶点）
vertex_indices = [1, 232, 443, 496, 619, 1118, 1505, 1535, 1553, 1641, 1728, 1766, 1783, 1824, 1990, 1998, 2115, 2463, 2551, 2567, 2596, 2967, 4685, 4704, 4999, 8249, 8723, 8809, 9812, 9833, 9851, 9860, 10735, 10740, 10744, 10794, 10831, 14404, 14462, 14475, 14490, 14626, 14642, 17119, 17168, 17182, 17206, 17610, 17924, 17940, 18742, 19328, 19811, 19818, 19841, 19862, 19945, 24230, 24255, 24356, 24484, 24487, 24499, 24531, 24532, 25973, 26014, 26970, 26984, 26987, 26991, 27015, 27056, 27096, 27098, 27216, 27222, 27278, 27359, 27462, 27466, 27718, 27988, 10946, 17988]

# 4. 获取这些顶点的坐标
target_vertices = vertices[vertex_indices]  # 通过索引提取多个顶点的坐标
print("Target Vertices:")
print(target_vertices)

# 5. 可视化这些顶点和对应的圆圈
point_cloud = o3d.geometry.PointCloud()

# 将目标顶点添加到点云
point_cloud.points = o3d.utility.Vector3dVector(target_vertices)

# 创建小圆圈来标记这些顶点
circle_radius = 0.05  # 圆圈的半径
circle_points = []

# 为每个目标顶点生成圆圈的点
for target_vertex in target_vertices:
    theta = np.linspace(0, 2 * np.pi, 100)
    for t in theta:
        # 生成一个圆圈的点，保持z坐标不变
        circle_points.append([target_vertex[0] + circle_radius * np.cos(t),
                              target_vertex[1] + circle_radius * np.sin(t),
                              target_vertex[2]])

# 将circle_points转换为NumPy数组
circle_points = np.array(circle_points)

# 打印输出，确认shape
print("Shape of circle_points:", circle_points.shape)

# 创建一个新的点云对象存储圆圈点
circle_cloud = o3d.geometry.PointCloud()
circle_cloud.points = o3d.utility.Vector3dVector(circle_points)

lines = []
for i in range(len(target_vertices) - 1):
    # 连接相邻的点
    lines.append([i, i + 1])

# 如果需要一个封闭的环形连接最后一个点与第一个点
lines.append([len(target_vertices) - 1, 0])

# 创建 LineSet 对象来可视化这些连线
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(target_vertices)
line_set.lines = o3d.utility.Vector2iVector(lines)

# 为了更好区分，给点云和线段上色
point_cloud.paint_uniform_color([0, 1, 0])  # 绿色顶点
circle_cloud.paint_uniform_color([1, 0, 0])  # 红色圆圈
line_set.paint_uniform_color([0, 0, 1])  # 蓝色线段

# 7. 可视化
o3d.visualization.draw_geometries([mesh, point_cloud, circle_cloud, line_set])

