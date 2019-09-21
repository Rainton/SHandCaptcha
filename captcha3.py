from __future__ import print_function
import sys
import argparse
import numpy as np
import math
import time
import socket
import cv2
from lib.ssds import ObjectDetector
from lib.utils.config_parse import cfg_from_file

VOC_CLASSES = 'person'
CROP_RATE = 0.6

ip_port = ('192.168.1.121', 9000)
BUFSIZE = 1024
udp_server_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
                        help='the address of optional config file', default=None, type=str, required=True)
    parser.add_argument('--demo', dest='demo_file',
                        help='the address of the demo file', default=None, type=str, required=True)
    parser.add_argument('-t', '--type', dest='type',
                        help='the type of the demo file, could be "image", "video", "camera" or "time", default is '
                             '"image"',
                        default='image', type=str)
    parser.add_argument('-d', '--display', dest='display',
                        help='whether display the detection result, default is True', default=True, type=bool)
    parser.add_argument('-s', '--save', dest='save',
                        help='whether write the detection result, default is False', default=False, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def random_point(x_min, x_max, y_min, y_max):
    def _inrange(x, y):
        if x_min <= x < x_max and y_min <= y < y_max:
            return True
        return False

    x_init = np.random.randint(x_min + int((x_max - x_min) / 4), x_max - int((x_max - x_min) / 4))
    y_init = np.random.randint(y_min + int((y_max - y_min) / 4), y_max - int((y_max - y_min) / 4))
    points = [[x_init, y_init]]
    point_num = np.random.randint(1, 3)
    tmp_x = x_init
    tmp_y = y_init
    while point_num != 0:
        rand_len = np.random.randint(100, 200)
        rand_theta = np.random.uniform(0, 2 * np.pi)
        new_x = int(tmp_x + rand_len * np.sin(rand_theta))
        new_y = int(tmp_y + rand_len * np.cos(rand_theta))
        if _inrange(new_x, new_y):
            points.append([new_x, new_y])
            point_num -= 1
            tmp_x = new_x
            tmp_y = new_y
    return points


def judge(hands_points, points):
    # x, y 是手动操作的点的x坐标和y坐标，其维度是几表示有几段折线
    flag = 0
    x = []
    y = []
    x_i = []
    y_i = []
    # p, q 是显示的点的x坐标和y坐标
    p = []
    q = []
    # K 是直线斜率k的集合， B 是b的集合  -- y = kx + b
    K = []
    B = []

    D = []
    X = []
    L2 = 0  # L2 表示所有点到目标直线的L2距离
    center = 0  # 到直线的距离大于100的所有的点，到这些点中心的距离之和除以其点的个数

    sum = []  # d大于100的点的数量
    for position in hands_points:
        for hand in position:
            if len(hand) == 2:
                # print(hand)
                x_i.append(hand[0])
                y_i.append(hand[1])
            else:
                if flag == 0:
                    x.append(x_i)
                    y.append(y_i)
                    x_i = []
                    y_i = []
                    flag = 1
    x.append(x_i)
    y.append(y_i)

    for corner in points:
        p.append(corner[0])
        q.append(corner[1])

    for i in range(len(p) - 1):
        k = (q[i] - q[i + 1]) / (p[i] - p[i + 1])
        b = q[i] - k * p[i]
        K.append(k)
        B.append(b)

    for i in range(len(x)):
        # print( " y = " + str(K[i]) + '* x + ' + str(b) + '\n')
        for index in range(len(x[i])):
            px = x[i][index]
            py = y[i][index]
            d = abs((K[i] * px - py + B[i]) / math.sqrt(k ** 2 + 1))
            if d > 100:
                sum.append(d)
            else:
                X.append(x[i][index])
                D.append(d)
            L2 += d
    if len(sum) != 0:
        num = 0
        for redundancy in sum:
            num += redundancy
        average = num / len(sum)

        num = 0
        for redundancy in sum:
            num += ((redundancy - average) ** 2)
        center = math.sqrt(num) / len(sum)
        if center > 2:
            return False
    cnt = 0
    # print(max(D))
    up = max(D) * 0.6
    for i in D:
        if i < up:
            cnt += 1
    # print(cnt, len(D), cnt / len(D))
    if max(D) < 20 or cnt / len(D) > 0.7:
        return True
    else:
        return False


def demo_live(config_file, index):
    # 1. load the configure file
    cfg_from_file(config_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load video
    state = "init"
    cam = cv2.VideoCapture(0)
    points = []
    record_points = []
    count_frames = 0
    all_count = 0
    success_count = 0
    circle_radius = 20
    line_thickness = 16
    close_distance = 20
    wait_frames = 10
    font_scale = 1
    font_thickness = 2
    middle_state = False
    middle_frame = 0
    bias = 50
    t = 0
    while True:
        retval, frame = cam.read()
        img_h, img_w, c = frame.shape
        # print('img_h,img_w:',img_h,img_w)
        s_y = int((1080 - img_h) / 2)
        s_x = int((1920 - img_w) / 2)
        bg_image = np.full((1080, 1920, 3), 127, np.uint8)
        bg_image[s_y:s_y + img_h, s_x:s_x + img_w] = frame
        frame = bg_image[:, int(1920 * (1 - CROP_RATE) / 2):int(1920 * (1 + CROP_RATE) / 2)]
        frame = cv2.flip(frame, 1)
        show_img = np.copy(frame)
        if state == "init":
            points = random_point(s_x - int(1920 * (1 - CROP_RATE) / 2) + bias,
                                  s_x + img_w - int(1920 * (1 - CROP_RATE) / 2) - bias, s_y + bias, s_y + img_h - bias)
            print("points:", points)

            pointsNum = len(points)
            starttt = str(points[0]).replace(' ', '').split('[')[1].split(']')[0]
            midddd = str(points[1]).replace(' ', '').split('[')[1].split(']')[0]
            if pointsNum == 3:
                endddd = str(points[2]).replace(' ', '').split('[')[1].split(']')[0]
                msg = starttt + ' ' + midddd + ' ' + endddd
                print('3point msg:', msg)
            else:
                msg = starttt + ' ' + midddd
                print('2point msg:', msg)
            if not msg:
                continue
       #     udp_server_client.sendto(msg.encode('utf-8'), ip_port)

            time_start = time.time()

            state = "waiting"

            # back_msg, addr = udp_server_client.recvfrom(BUFSIZE)
            # print(back_msg.decode('utf-8'), addr)

        for i in range(len(points) - 1):
            cv2.line(show_img, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), (255, 0, 0),
                     line_thickness)
        cv2.circle(show_img, tuple(points[0]), circle_radius, (0, 255, 0), -1)
        cv2.circle(show_img, tuple(points[-1]), circle_radius, (0, 0, 255), -1)
        for i in range(1, len(points) - 1):
            cv2.circle(show_img, tuple(points[i]), circle_radius, (255, 0, 255), -1)
        _labels, _scores, _coords = object_detector.predict(frame)
        hands_points = []
        for labels, scores, coords in zip(_labels, _scores, _coords):
            tmp_x = int((coords[0] + coords[2]) / 2)
            tmp_y = int((coords[1] + coords[3]) / 2)
            cv2.circle(show_img, (tmp_x, tmp_y), circle_radius, (0, 255, 255), -1)
            hands_points.append([tmp_x, tmp_y])

        time_interval = time.time() - time_start
        if time_interval > 200:
            print("======================> time out  ==================> time interval > 30s")
            state = "init"
            print("----------------")
            print("no")
            all_count += 1
            count_frames = 0
            middle_frame = 0
            middle_state = False
            print("{}:{}".format("time", time_interval))
            record_points = []

        else:
            if state == "waiting":
                if count_frames > wait_frames:
                    state = "Recording"
                    t = time.time()
                    count_frames = 0
                has_found = False
                for h_point in hands_points:
                    if np.sqrt((h_point[0] - points[0][0]) ** 2 + (h_point[1] - points[0][1]) ** 2) < close_distance:
                        has_found = True
                        count_frames += 1
                if has_found == False:
                    count_frames = 0
            if state == "Recording":
                record_points.append(hands_points)
                cv2.putText(show_img, "Recording", (10, 150), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255),
                            font_thickness)
                if middle_frame > wait_frames:
                    middle_state = True
                    record_points.append("Middle")
                    middle_frame = 0
                if middle_state:
                    cv2.circle(show_img, tuple(points[1]), circle_radius, (0, 255, 0), -1)
                if count_frames > wait_frames:
                    state = "init"
                    print("----------------")
                    if len(points) == 3:
                        if judge(record_points, points) and middle_state:
                            success_count += 1
                            print("yes")
                        else:
                            print("no")
                    else:
                        if judge(record_points, points):
                            success_count += 1
                            print("yes")
                        else:
                            print("no")
                    all_count += 1
                    count_frames = 0
                    middle_frame = 0
                    middle_state = False
                    print("{}:{}".format("time", time.time() - t))

                    record_points = []
                has_found = False
                middle_found = False
                for h_point in hands_points:
                    if np.sqrt((h_point[0] - points[-1][0]) ** 2 + (h_point[1] - points[-1][1]) ** 2) < close_distance:
                        has_found = True
                        count_frames += 1
                    if len(points) > 2 and middle_state is False and np.sqrt(
                            (h_point[0] - points[1][0]) ** 2 + (h_point[1] - points[1][1]) ** 2) < close_distance:
                        middle_frame += 1
                        middle_found = True
                if has_found == False:
                    count_frames = 0
                if middle_found == False:
                    middle_frame = 0

        cv2.putText(show_img, "Result:{}/{}".format(success_count, all_count), (10, 50), cv2.FONT_HERSHEY_COMPLEX,
                    font_scale, (0, 0, 255), font_thickness)
        cv2.putText(show_img, "{} {}".format(state, count_frames), (10, 100), cv2.FONT_HERSHEY_COMPLEX, font_scale,
                    (0, 0, 255), font_thickness)
        cv2.imshow("Frame", show_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == '__main__':
    CROP_RATE = 0.6
    demo_live("./experiments/cfgs/yolo_v3_small_inceptionv4_v3_8.14.yml", 0)  # AP: 95.05%
