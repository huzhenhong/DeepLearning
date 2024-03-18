'''
Author: huzhenhong 455879568@qq.com
Date: 2023-12-19 18:10:09
LastEditors: huzhenhong 455879568@qq.com
LastEditTime: 2023-12-22 17:08:37
FilePath: /DeepLearning/huzh/job/goods_move/sift.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2

# 读取图像，转灰度图进行检测
# img = cv2.imread('/Users/huzh/Pictures/Screenshots/yinguang_fensuijian-0003.jpg')

# imageA = cv2.imread("/Users/huzh/Documents/gitlab/goodsmovedetector/goods_move_debug/1606工房2_20230920151000-20230920173000_3.mp4/2023-12-19 14_01_13.189773_4552980504/2023-12-19 15_11_24.247364/cur.jpg")
# imageB = cv2.imread("/Users/huzh/Documents/gitlab/goodsmovedetector/goods_move_debug/1606工房2_20230920151000-20230920173000_3.mp4/2023-12-19 14_01_13.189773_4552980504/2023-12-19 15_11_24.247364/ref.jpg")

# roi = [0.22875,
#         0.7577777777777778,
#         0.35625,
#         0.999]

imageA = cv2.imread("/Users/huzh/Pictures/Screenshots/yinguang_fensuijian-0002.jpg")
imageB = cv2.imread("/Users/huzh/Pictures/Screenshots/yinguang_fensuijian-0003.jpg")

roi = [0.60625,
       0.6288888888888889,
       0.74125,
       0.8666666666666667]

height, width = imageA.shape[:2]
imageA = imageA[int(roi[1] * height): int(roi[3] * height), int(roi[0] * width): int(roi[2] * width), :]
imageB = imageB[int(roi[1] * height): int(roi[3] * height), int(roi[0] * width): int(roi[2] * width), :]

img = imageA
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sift实例化对象
sift = cv2.SIFT_create()

# 关键点检测
keypoint = sift.detect(img_gray)

# 关键点信息查看
# print(keypoint)  # [<KeyPoint 000001872E1E2960>, <KeyPoint 000001872E1E2B10>]
original_kp_set = {(int(i.pt[0]), int(i.pt[1])) for i in keypoint}  # pt查看关键点坐标
print(original_kp_set)

# 在图像上绘制关键点的检测结果
cv2.drawKeypoints(img, keypoint, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示图像
cv2.imshow("img", img)
cv2.imwrite('002.jpg', img)
cv2.waitKey()
