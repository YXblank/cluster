import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('/home/xuyuan/图片/test.png')

# 获取图像的高度、宽度和通道数
height, width, _ = image.shape

# 创建一张与原图大小相同的副本，用于绘制连接线
output_image = image.copy()

# 颜色类别的映射
def classify_color(r, g, b):
    if r > g and r > b:
        return 'Red'
    elif g > r and g > b:
        return 'Green'
    elif b > r and b > g:
        return 'Blue'
    elif r > 100 and b > 100 and abs(r - b) < 50:  # 粉色
        return 'Pink'
    elif r > 100 and b > 100 and abs(r - b) >= 50:  # 紫色
        return 'Purple'
    elif r > 100 and g > 100 and abs(r - g) < 50:  # 黄色
        return 'Yellow'
    elif r == g == b:  # 灰色
        return 'Gray'
    else:  # 青色
        return 'Cyan'

# 设置距离阈值
distance_threshold = 50  # 距离阈值，可以根据需要调整

# 计算欧几里得距离
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 用于记录每次连接的颜色类别对和距离
color_categories = []
connection_distances = []

# 遍历图像中的每一对像素，y 轴采样更密集，x 轴间隔较大
for y1 in range(0, height, 10):  # y 轴每10个像素采样一次（密集采样）
    for x1 in range(0, width, 20):  # x 轴每20个像素采样一次
        # 获取第一个像素的 RGB 值
        r1, g1, b1 = image[y1, x1]

        # 排除黑色或白色像素
        if (r1 == 0 and g1 == 0 and b1 == 0) or (r1 == 255 and g1 == 255 and b1 == 255):
            continue  # 跳过黑色或白色像素

        # 通过颜色分类函数确定颜色类别
        color1_category = classify_color(r1, g1, b1)

        for y2 in range(y1 + 10, height, 10):  # y 轴采样间隔为10
            for x2 in range(x1 + 20, width, 20):  # x 轴采样间隔为20
                # 获取第二个像素的 RGB 值
                r2, g2, b2 = image[y2, x2]

                # 排除黑色或白色像素
                if (r2 == 0 and g2 == 0 and b2 == 0) or (r2 == 255 and g2 == 255 and b2 == 255):
                    continue  # 跳过黑色或白色像素

                # 通过颜色分类函数确定颜色类别
                color2_category = classify_color(r2, g2, b2)

                # 计算两个像素的欧几里得距离
                distance = euclidean_distance(x1, y1, x2, y2)

                # 如果距离大于阈值，则跳过连接
                if distance > distance_threshold:
                    continue

                # 记录颜色类别对和它们之间的距离
                color_categories.append((color1_category, color2_category))  # 记录颜色对
                connection_distances.append(distance)  # 记录连接距离

                # 连接这两个像素的中心点
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

# 显示结果图像
cv2.imshow("Connected Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 绘制直方图：横坐标是颜色对，纵坐标是连接距离
plt.figure(figsize=(10, 6))

# 将颜色对转换为字符串表示
color_pair_str = ['[' + color1 + ', ' + color2 + ']' for color1, color2 in color_categories]

# 绘制颜色对与距离的散点图
plt.scatter(color_pair_str, connection_distances, color='b', alpha=0.5)

plt.title("Color Category Pair vs Connection Distance")
plt.xlabel("Color Category Pair")
plt.ylabel("Connection Distance (Euclidean)")

# 由于横坐标包含很多信息，可能需要旋转标签
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

