import serial
import time
import cv2
from picamera import PiCamera
import numpy as np
# from scipy import io
import multiprocessing
import math
import threading


# import matplotlib.pyplot as plt


class MySerial:
    def __init__(self):
        self._ser = serial.Serial('/dev/ttyAMA0', 9600)  # 初始化串口
        # 命令初始化
        self.STEP_CMD_ON = '<StepON>'.encode('ascii')
        self.STEP_CMD_OFF = '<StepOFF>'.encode('ascii')
        self.STEP_CMD_ONE_STEP = '<StepOne>'.encode('ascii')
        self.STEP_CMD_RUN_TEST = '<StepRUN>'.encode('ascii')
        self.STEP_CMD_DIR_RIGHT = '<directRight>'.encode('ascii')
        self.STEP_CMD_DIR_LEFT = '<directLeft>'.encode('ascii')
        self.LIGHT_CMD_ON = '<lightON>'.encode('ascii')
        self.LIGHT_CMD_OFF = '<lightOFF>'.encode('ascii')

    def write(self, cmd):
        '''
        :param cmd: 待发送的命令
        :return:
        '''
        try:
            self._ser.write(cmd)
        except KeyboardInterrupt as serialException:
            print(serialException)

    """
    def write1(self, cmd):
        '''
        :param cmd: 待发送的命令
        :return:
        '''
        if not self.ser.isOpen():
            self.ser.open()
        try:
            self.ser.write(cmd)
        except KeyboardInterrupt as serialException:
            print(serialException)
    """

    def close(self):
        self._ser.close()

    def open(self):
        self._ser.open()


class Stepper:
    ROUND = 1600

    def __init__(self):
        self._ser = MySerial()
        self.closeStep()
        self.rightRun()

    def openStep(self):
        self._ser.write(self._ser.STEP_CMD_ON)

    def closeStep(self):
        self._ser.write(self._ser.STEP_CMD_OFF)

    def rightRun(self):
        self._ser.write(self._ser.STEP_CMD_DIR_RIGHT)

    def leftRun(self):
        self._ser.write(self._ser.STEP_CMD_DIR_LEFT)

    def stepOne(self):
        self.openStep()
        self._ser.write(self._ser.STEP_CMD_ONE_STEP)

    def stepRound(self):
        self.openStep()
        self._ser.write(self._ser.STEP_CMD_RUN_TEST)
        self.closeStep()

    def angleRun(self, angle=360):
        '''
        :param angle: 转动角度
        :return:
        '''
        times = int((angle / 360) * self.ROUND)  # 步进次数
        for t in range(times):
            self._ser.write(self._ser.STEP_CMD_ONE_STEP)


class Light:
    def __init__(self):
        self._ser = MySerial()

    def openLight(self):
        self._ser.write(self._ser.LIGHT_CMD_ON)

    def closeLight(self):
        self._ser.write(self._ser.LIGHT_CMD_OFF)


class MyCamera:
    def __init__(self):
        self.camera = PiCamera()  # 初始化相机
        self.PHOTO_INFO = {'width': int(1920), 'height': int(1408), 'mode': 3}  # 照片格式
        self.camera.resolution = (self.PHOTO_INFO['width'], self.PHOTO_INFO['height'])  # 设置照片格式
        self.camera.start_preview()  # 启动相机

    def openCamera(self):
        self.camera.start_preview()  # 启动相机

    def getImage(self):
        '''
        :return: 返回采集的照片
        '''
        img = np.empty((self.PHOTO_INFO['height'] * self.PHOTO_INFO['width'] * self.PHOTO_INFO['mode'],),
                       dtype=np.uint8)
        self.camera.capture(img, 'bgr')
        img = img.reshape((self.PHOTO_INFO['height'], self.PHOTO_INFO['width'], self.PHOTO_INFO['mode']))
        return img

    def closeCamera(self):
        self.camera.close()


class ImageHandle:
    def __init__(self):
        self._InterestRange = {'left_col': 720, 'top_row': 840, 'right_col': 940, 'bottom_row': 1270}  # 参考范围
        sLine = self.findCenterLine(
            self.rangeByColors(self.getInterestRange(cv2.imread('standerd.png'))))  # 参考直线

    def getImgSize(self):
        return self._InterestRange['bottom_row'] - self._InterestRange['top_row'], self._InterestRange['right_col'] - \
               self._InterestRange['left_col']

    def getGrayImg(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def getOutLine(self, img):
        ''' 获取轮廓 '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 30, 90)
        return edges

    def getGaussianBlur(self, img):
        '''高斯模糊'''
        kernel_size = (5, 5)
        sigma = 2.0
        return cv2.GaussianBlur(img, kernel_size, sigma)

    def getLines(self, img):
        edges = cv2.Canny(img, 50, 200)
        minLineLength = 300
        maxLineGap = 15
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img

    def getInterestRange(self, img):
        img = img[self._InterestRange['top_row']:self._InterestRange['bottom_row'],
              self._InterestRange['left_col']:self._InterestRange['right_col']]

        return img

    def rangeByColors(self, img, *colors):

        '''
        根据给定的颜色范围范围颜色范围图片
        :param img: 兴趣范围的图片
        :type colors: np.array,给定的任何颜色
        '''
        # 生成颜色范围
        if len(colors) == 0:
            # 未给定颜色使用默认颜色范围
            lower_color = np.array([0, 0, 135])
            upper_color = np.array([136, 150, 255])
        else:
            lower_color = np.min(np.array(colors), axis=0)
            upper_color = np.max(np.array(colors), axis=0)
        mask = cv2.inRange(img, lower_color, upper_color)
        return mask

    def rangeByGray(self, img, *colors):

        '''
        根据灰度滤波图像
        :param img: 兴趣范围的图片
        :type colors: np.array,给定的二值颜色范围
        '''
        # 生成颜色范围
        if len(colors) == 0:
            # 未给定颜色使用默认颜色范围
            lower_color = 100
            upper_color = 255
        else:
            lower_color = np.min(np.array(colors), axis=0)
            upper_color = np.max(np.array(colors), axis=0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, lower_color, upper_color)
        return mask

    def findCenterLine(self, img):
        '''
        :param img: 二值图片
        :return: 直线
        无奈~~~时间太长了
        '''
        # 根据给定图片生成空图片
        # start = time.time()
        lineImg = np.empty(img.shape, dtype=np.uint8)
        lineImg[:] = 0
        # 包含特征像素的行
        line_rows = []
        rowP, colP = np.where(img == 255)
        [line_rows.append(i) for i in (rowP) if not i in line_rows]
        # 根据每行目标像素位置算出每行中心像素位置
        for row in line_rows:
            # 查询以获得一行中拥有目标像素的点的索引
            cols_index = np.where(rowP == row)
            # 目标点列的位置
            cols = []
            col = None
            # 根据索引以获取目标的列位置
            cols.append((colP[cols_index]))
            # 计算每行的直线中心
            col = int(np.average(colP[cols_index]))
            lineImg[row][col] = 255
        # print('get rang time:' + str(time.time() - start))
        return lineImg

    def getPosition(self, img, angle):
        '''
        标定参数：1cm为24个像素点 1920/2下（转台中心位置）
                  1cm为42个像素点 1920下
        焦距：F = (P x D) / W
        P:41,D:42.1,W:1===>F:1880
        每向前移动1cm，激光为止平移3个像素 1920/2下
        每向前移动1cm，激光为止平移13个像素  1920下
        摄像头距离转台中心为42.1cm
        '''
        sLine = self.findCenterLine(
            self.rangeByColors(self.getInterestRange(cv2.imread('standerd.png'))))  # 参考直线,建议该变量放到全局（读写操作费时间）
        # cv2.imshow('sl', sLine)
        # cv2.waitKey(0)
        interestImg = self.getInterestRange(img)
        blurImg = self.getGaussianBlur(interestImg)  # 图片模糊处理
        mask1 = self.rangeByGray(blurImg)  # 灰度滤波
        mask2 = self.rangeByColors(blurImg)  # 阈值滤波
        laserLine = self.findCenterLine(mask1 | mask2)  # 实际激光直线
        # cv2.imshow('ll', laserLine)
        # cv2.waitKey(0)
        sx, sy = np.where(sLine == 255)
        lx, ly = np.where(laserLine == 255)
        # sx = sx.tolist()
        lx = lx.tolist()
        #  pointInfo = np.empty(img.shape)
        # print(type(lx))
        lineInfo = np.empty([len(sx), 3])
        lineInfo[:] = 0
        for sp in range(len(sx)):
            splot_y = sy[sp]
            if sp in lx:
                lplot_y = ly[lx.index(sp)]
                # print(sp, lplot_y - splot_y)
                pixDistence = lplot_y - splot_y
                # print(pixDistence)
                # if pixDistence > 0:
                #     deepth_cam = 42.5 - (abs(pixDistence) * (1 / 13))
                # else:
                #     deepth_cam = 42.5 + abs(pixDistence) * (1 / 13)
                deepth = round(pixDistence * (1 / 13), 3)
                p_x = round(deepth * math.cos(angle), 3)
                p_y = round(deepth * math.sin(angle), 3)
                p_z = round((len(sx) - sp) * (1 / 15))
                lineInfo[sp, 0] = p_x
                lineInfo[sp, 1] = p_y
                lineInfo[sp, 2] = p_z
        #print(lineInfo)
        return lineInfo
            # print('x:' + str(p_x) + '\ty:' + str(p_y) + '\tz:' + str(p_z))


class Scanner:

    # PIX_LENGTH = 100 / 24  # 像素长度，单位毫米

    def __init__(self):
        self._stepper = Stepper()
        self._light = Light()
        self._camera = MyCamera()
        self._imageHandle = ImageHandle()
        self._stepper.openStep()  # 电机使能
        self._light.closeLight()  # 关闭激光
        time.sleep(1)  # 等待设备时间
        self.model = None  # np.empty([0])  # 创建空模型

    def scan(self, angle):
        times = int((angle / 360) * self._stepper.ROUND)  # 根据角度计算出转动的次数
        # print(times)  # 测试
        self._stepper.openStep()  # 电机使能
        self._stepper.rightRun()  # 设置转动方向
        for t in range(times):
            # color_img = self._camera.getImage()  # 采集颜色图像
            # cv2.imwrite('color.jpg', color_img)  # 保存颜色图片
            self._light.openLight()
            laser_img = self._camera.getImage()  # 采集激光图像
            self._light.closeLight()
            start = time.time()
            print(t)
            lineInfo = self._imageHandle.getPosition(img=laser_img, angle=(math.pi) * (t * angle / self._stepper.ROUND))
            for l in lineInfo:
                print(l)
            if self.model is None:
                self.model = lineInfo
            else:
                self.model = np.append(self.model, lineInfo, axis=0)
            print(self.model.shape)
            print(time.time() - start)
            scanner.saveAsFile('model', scanner.model)
            print("-----------------Next Time-----------------")

            self._stepper.stepOne()  # 步进

    def saveAsFile(self, filename, file):
        # io.savemat(filename + ".mat", {'array': file})
        np.save(filename, file)

    def closeScanner(self):
        self._stepper.closeStep()
        self._light.closeLight()
        # self.saveAsFile('model', self.model)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    scanner = Scanner()
    try:
        # scanner.scan(20)
        pool.apply_async(scanner.scan(360), (1,))
    finally:
        scanner.closeScanner()
        # scanner.saveAsFile('model', scanner.model)
