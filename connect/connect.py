import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/xuyuan/pan/data/global.png')

# 将图像转换为 RGB 格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像数据 reshape 成一维数组，K-means 需要一维数据
pixels = image_rgb.reshape((-1, 3))

# 使用 K-means 聚类将颜色分为 5 类（可根据需要调整）
k = 25
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将每个像素的标签应用到图像，重新生成每个像素对应颜色的聚类图
segmented_image = centers[labels.flatten()].reshape(image_rgb.shape).astype(np.uint8)
cv2.imshow("segmented_image ", segmented_image)
# 通过轮廓检测提取每个颜色区域的节点
contour_image = np.zeros_like(segmented_image)
hsv_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
lower = np.array([10, 10, 10])
upper = np.array([245, 245, 245])

# 创建掩码
mask = cv2.inRange(hsv_image, lower, upper)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在图像上绘制轮廓
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # 绿色轮廓，线宽2

# 显示轮廓
cv2.imshow("Contours", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 提取轮廓的中心点作为节点
nodes = []
for contour in contours:
    if cv2.contourArea(contour) > 1:  # 过滤掉小区域
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            nodes.append((cx, cy))

# 最大连接距离
max_distance = 100

# 连接节点
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        dist = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
        if dist <= max_distance:
            cv2.line(image, nodes[i], nodes[j], (0, 0, 0), 2)  # 绿色连接线

# 显示结果
cv2.imshow("Connected Nodes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

