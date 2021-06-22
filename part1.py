# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:36:09 2021

@author: 78053
"""

import numpy as np
import cv2 as cv
import time
cap = cv.VideoCapture(1)
from PIL import ImageGrab
import keyboard

def capture(left, top, right, bottom):
    img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    r, g, b = cv.split(img)
    cv.merge([b, g, r], img)
    return img

def test_c():
    print()
    print('启动截图功能')
    print('...........................................................................')
    print()
    print('你有3秒钟的时间来展现图片')
    print('...........................................................................')
    time.sleep(3)
    #cv2.imshow("screen", capture(0,0,1920,1080))
    cv.imwrite("D:/yolo/original.jpg",capture(0,0,1920,1080))
    cv.waitKey(0)
        
        
        
    time.sleep(1)
    img = 'D:/yolo/original.jpg'
    img = cv.imread(img)
    cv.imshow('original', img)
    print()
    print('截取你想要的部分，按下Enter键以确定保持')
    print('...........................................................................')
      
    
        # 选择ROI
    roi = cv.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    print(roi)
        
        # 显示ROI并保存图片
    if roi != (0, 0, 0, 0):
        crop = img[y:y+h, x:x+w]
        #cv.imshow('crop', crop)
        cv.imwrite("D:/yolo/final.jpg", crop)
        print()
        print('保存成功！')
        print('...........................................................................')
        # 退出
        cv.waitKey(0)
        cv.destroyAllWindows()
def test_p():
    print()
    print('启动拍摄功能')
    print('...........................................................................')
    def getcampic(fname):
        print()
        print('按下t键拍摄图片')
        print('...........................................................................')
        cap = cv.VideoCapture(1)        # 打开摄像头
        while True:
        # 获取一帧又一帧
            ret, frame = cap.read()
            cv.imshow("vipdeo", frame)
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
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    keyboard.add_hotkey('p', test_p)
    keyboard.add_hotkey('c', test_c)