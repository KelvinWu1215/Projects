import cv2
import numpy as np
import time
from random import randint

contour_up = 0
contour_down = 0

video_stored = cv2.VideoCapture("example1.mp4")

# Get width and height
width = video_stored.get(3)
height = video_stored.get(4)
screenArea = height * width
areaTH = screenArea / 400

# Lines
up_line = int(2 * (height / 5))
down_line = int(3 * (height / 5))

# Present the Lines
line1 = [0, down_line]
line2 = [width, down_line]
lines_L1 = np.array([line1, line2], np.int32)
lines_L1 = lines_L1.reshape((-1, 1, 2))

line3 = [0, up_line]
line4 = [width, up_line]
lines_L2 = np.array([line3, line4], np.int32)
lines_L2 = lines_L2.reshape((-1, 1, 2))

# limits for each object
car_limit = int(5 * (height / 5))
up_limit = int(1 * (height / 5))
down_limit = int(4 * (height / 5))

# Kernels
kernelOp = np.ones((3, 3), np.uint8)
kernelCl = np.ones((11, 11), np.uint)

cars = []
max_age = 5
id = 1
total_car = 0

# The Lines Color for Each Class
line_down_color = (255, 0, 0)
line_up_color = (255, 0, 255)

# Background Subtractor
BackGround_Sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)


class Vehicle:
    vehicles = []

    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.vehicles = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None

    def getRGB(self):  # For the RGB colour
        return self.R, self.G, self.B

    def getTracks(self):
        return self.vehicles

    def getId(self):  # For the ID
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):  # for x coordinate
        return self.x

    def getY(self):  # for y coordinate
        return self.y

    def updating(self, xn, yn):
        self.age = 0
        self.vehicles.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def setDone(self):
        self.done = True

    def timedOut(self):
        return self.done

    def going_UP(self, mid_end):
        if len(self.vehicles) >= 2:
            if self.state == '0':
                if self.vehicles[-1][1] < mid_end <= self.vehicles[-2][1]:
                    state = '1'
                    self.dir = 'up'
                    return state
                else:
                    return False
            else:
                return False
        else:
            return False

    def going_DOWN(self, mid_start):
        if len(self.vehicles) >= 2:
            if self.state == '0':
                if self.vehicles[-1][1] > mid_start >= self.vehicles[-2][1]:
                    state = '1'
                    self.dir = 'down'
                    return state
                else:
                    return False
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True


while video_stored.isOpened():
    reading, frame = video_stored.read()
    for i in cars:
        i.age_one()

    # Background subtract to increase the accuracy of the detection
    BackGround_Sub_apply = BackGround_Sub.apply(frame)
    BackGround_Sub_apply2 = BackGround_Sub.apply(frame)

    if reading:

        # Binarization
        reading, Bin = cv2.threshold(BackGround_Sub_apply, 200, 255, cv2.THRESH_BINARY)
        reading, Bin2 = cv2.threshold(BackGround_Sub_apply2, 200, 255, cv2.THRESH_BINARY)

        # Opening i.e First Erode the dilate
        BackGround_Sub_apply = cv2.morphologyEx(Bin, cv2.MORPH_OPEN, kernelOp)
        BackGround_Sub_apply2 = cv2.morphologyEx(Bin2, cv2.MORPH_CLOSE, kernelOp)

        # Closing i.e First Dilate then Erode
        BackGround_Sub_apply = cv2.morphologyEx(BackGround_Sub_apply, cv2.MORPH_CLOSE, kernelCl)
        BackGround_Sub_apply2 = cv2.morphologyEx(BackGround_Sub_apply2, cv2.MORPH_CLOSE, kernelCl)

        # Find Contours
        contours, level = cv2.findContours(BackGround_Sub_apply, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > areaTH:
                # Tracking
                m = cv2.moments(contour)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                x, y, width, height = cv2.boundingRect(contour)

                new = True
                if cy in range(up_limit, down_limit):

                    for i in cars:

                        if abs(x - i.getX()) <= width and abs(y - i.getY()) <= height:
                            new = False
                            i.updating(cx, cy)

                            if i.going_UP(up_line):
                                contour_up += 1
                                print("ID:", i.getId(), 'the object is going up at', time.strftime("%c"))
                                total_car += 1

                            elif i.going_DOWN(down_line):
                                contour_down += 1
                                print("ID:", i.getId(), 'the object is going down at', time.strftime("%c"))
                                total_car += 1

                            break

                        if i.getState() == '1':
                            if i.getDirection() == 'down' and i.getY() > down_limit:
                                i.setDone()
                            elif i.getDirection() == 'up' and i.getY() < up_limit:
                                i.setDone()

                        if i.timedOut():
                            index = cars.index(i)
                            cars.pop(index)
                            del i

                    # If nothing is detected,create new
                    if new:
                        p = Vehicle(id, cx, cy, max_age)
                        cars.append(p)
                        id += 1

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                img = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        for i in cars:
            cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), cv2.FONT_HERSHEY_SIMPLEX, 0.3, i.getRGB(), 1,
                        cv2.LINE_AA)

        # Display the Texts on the Screen
        GoingUp = 'UP: ' + str(contour_up)
        GoingDown = 'DOWN: ' + str(contour_down)
        cars_Object = 'TOTAL VEHICLES: ' + str(total_car)

        # The Line in video for making sure the range of counting system
        frame1 = cv2.polylines(frame, [lines_L1], False, line_down_color, thickness=2)

        # The bottom of the Text (To make sure user is able to see the value)
        cv2.putText(frame1, GoingDown, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # The style of Text
        cv2.putText(frame1, GoingDown, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        frame2 = cv2.polylines(frame, [lines_L2], False, line_up_color, thickness=2)

        cv2.putText(frame2, GoingUp, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame2, GoingUp, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame1, cars_Object, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame2, cars_Object, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Frame', frame)

        # When user click "Space", then the video will turn off
        if cv2.waitKey(1) & 0xff == ord(' '):
            break

    else:
        break

video_stored.release()
cv2.destroyAllWindows()
