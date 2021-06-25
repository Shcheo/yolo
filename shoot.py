# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:41:15 2021

@author: 78053
"""


import numpy as np
import cv2 as cv
cap = cv.VideoCapture(1)
 
def getcampic(fname):
    print()
    print('按下t键拍摄图片')
    print('...........................................................................')
    cap = cv.VideoCapture(1)        # 打开摄像头
    while True:
    # 获取一帧又一帧
        ret, frame = cap.read()
        cv.imshow("video", frame)
        if cv.waitKey(1) == ord("t"):
            #time.sleep(1) # 等待1秒，避免黑屏
            ret, frame = cap.read()       # 读摄像头
            cv.imwrite("D:/yolo/final.jpg",frame)
            print()
            print('保存成功！')
            print('...........................................................................')
            cap.release()      
            cv.destroyAllWindows() 
            return True



##测试
getcampic("final.jpg")


# 通常，序列CNN网络在最终只会给出一个输出结果，
# 在YOLO v3版本中，会输出多个预测层。每一个输出的预测层不与任何下一个层连接。
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[arr[0] - 1] for arr in net.getUnconnectedOutLayers()]
    return output_layers
 
 
# 画框和分类文字。
def draw_rec(image, x, y, width, height, color, label, number):
    cv.rectangle(img=image, pt1=(x, y), pt2=(x + width, y + height), color=color[0], thickness=2)
    text = "{}:{:.3f}".format(label, number)
    cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color, thickness=1)
 
 
if __name__ == '__main__':
    print()
    print('图片分析中')
    print('...........................................................................')
    weightsPath = "D:/yolo/yolov3.weights"
    configPath = "D:/yolo/yolov3.cfg"
    labelsPath = "D:/yolo/coco.names"
    imagePath = "D:/yolo/final.jpg"
 
    conf_threshold = 0.5
    nms_threshold = 0.4
 
    LABELS = open(labelsPath).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
 
    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
 
    # 待检测的图像.
    image = cv.imread(imagePath)
    (H, W) = image.shape[:2]
 
    scale = 1 / 255
 
    blob = cv.dnn.blobFromImage(image, scale, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
 
    outputs = net.forward(get_output_layers(net))
 
    boxes = []
    confidences = []
    classIDs = []
 
    # 循环处理每个输出的预测层。
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
 
            # 过滤置信度较小的检测结果
            if confidence > 0.5:
                confidences.append(float(confidence))
                classIDs.append(classID)
 
                # 框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
 
                (centerX, centerY, width, height) = box.astype("int")
 
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
 
                boxes.append([x, y, int(width), int(height)])
 
    # 最大值抑制。
    idxs = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
 
    print(boxes)
    print()
    print('分析成功')
    print('...........................................................................')
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
 
            # 原图上绘制边框和分类.
            color = [int(c) for c in COLORS[classIDs[i]]]
            # cv.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=1)
            # text = "{}:{:.3f}".format(LABELS[classIDs[i]], confidences[i])
            # cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_COMPLEX, 0.5, color=color, thickness=1)
            draw_rec(image, x, y, w, h, color, LABELS[classIDs[i]], confidences[i])
 
    cv.imshow("image", image)
    cv.imwrite("D:/yolo/results.jpg",image)
    cv.waitKey(0)
