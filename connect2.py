import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取图像
image = cv2.imread('/home/xuyuan/图片/2025-04-13 21-26-35 的屏幕截图.png')

# 获取图像的高度、宽度和通道数
height, width, _ = image.shape

# 创建一张与原图大小相同的副本，用于绘制连接线
output_image = image.copy()

# 用一个集合来存储已经连接过的像素颜色块（以RGB值作为键）
connected_colors = set()
connection_distances = []
# 设置距离阈值
distance_threshold = 50  # 距离阈值，可以根据需要调整

# 计算欧几里得距离
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 遍历图像中的每一对像素，y 轴采样更密集，x 轴间隔较大
for y1 in range(0, height, 10):  # y 轴每10个像素采样一次（密集采样）
    for x1 in range(0, width, 10):  # x 轴每20个像素采样一次
        # 获取第一个像素的 RGB 值
        r1, g1, b1 = image[y1, x1]

        # 排除黑色或白色像素 
        if (r1 == 0 and g1 == 0 and b1 == 0) or (r1 == 255 and g1 == 255 and b1 == 255):
            continue  # 跳过黑色或白色像素

        # 将该像素的颜色 (r1, g1, b1) 转为一个颜色标记
        color_label = (r1, g1, b1)

        # 如果这个颜色块已经连接过，则跳过
        if color_label in connected_colors:
            continue

        for y2 in range(y1 + 10, height, 10):  # y 轴采样间隔为10
            for x2 in range(x1 + 10, width, 10):  # x 轴采样间隔为20
                # 获取第二个像素的 RGB 值
                r2, g2, b2 = image[y2, x2]

                # 排除黑色或白色像素
                if (r2 == 0 and g2 == 0 and b2 == 0) or (r2 == 255 and g2 == 255 and b2 == 255):
                    continue  # 跳过黑色或白色像素

                # 计算两个像素的欧几里得距离
                distance = euclidean_distance(x1, y1, x2, y2)

                # 如果距离大于阈值，则跳过连接
                if distance > distance_threshold:
                    continue

                # 确保 RGB 差异不会导致溢出，转换为整数进行比较
                r_diff = abs(int(r1) - int(r2))
                g_diff = abs(int(g1) - int(g2))
                b_diff = abs(int(b1) - int(b2))

                # 检查 RGB 任意通道的差异是否大于 20
                if r_diff > 20 or g_diff > 20 or b_diff > 20:
                    # 连接这两个像素的中心点
                    cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 0), 1)
                    connection_distances.append(distance)

        # 标记该颜色块已经连接过
        connected_colors.add(color_label)

# 显示结果
cv2.imshow("Connected Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.hist(connection_distances, bins=50, color='g', alpha=0.7)
plt.title("Histogram of Connection Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()
