import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/xuyuan/图片/2025-04-13 21-26-35 的屏幕截图.png')

# 获取图像的高度、宽度和通道数
height, width, _ = image.shape

# 创建一张与原图大小相同的副本，用于绘制连接线
output_image = image.copy()

# 遍历图像中的每一对像素
for y1 in range(0, height, 10):  # 每20个像素采样一次
    for x1 in range(0, width, 30):  # 每20个像素采样一次
        # 获取第一个像素的 RGB 值
        r1, g1, b1 = image[y1, x1]

        # 排除黑色或白色像素
        if (r1 == 0 and g1 == 0 and b1 == 0) or (r1 == 255 and g1 == 255 and b1 == 255):
            continue  # 跳过黑色或白色像素

        for y2 in range(y1 + 30, height, 10):  # 也按步长20进行采样
            for x2 in range(x1 + 30, width, 30):  # 也按步长20进行采样
                # 获取第二个像素的 RGB 值
                r2, g2, b2 = image[y2, x2]

                # 排除黑色或白色像素
                if (r2 == 0 and g2 == 0 and b2 == 0) or (r2 == 255 and g2 == 255 and b2 == 255):
                    continue  # 跳过黑色或白色像素

                # 确保 RGB 差异不会导致溢出，转换为整数进行比较
                r_diff = abs(int(r1) - int(r2))
                g_diff = abs(int(g1) - int(g2))
                b_diff = abs(int(b1) - int(b2))

                # 检查 RGB 任意通道的差异是否大于 20
                if r_diff > 30 or g_diff > 30 or b_diff > 30:
                    # 连接这两个像素的中心点
                    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

# 显示结果
cv2.imshow("Connected Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

