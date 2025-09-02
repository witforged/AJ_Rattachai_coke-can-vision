import cv2
import numpy as np
import robomaster
from robomaster import robot
from robomaster import vision
import detactfile 


class MarkerInfo:

    def __init__(self, x, y, w, h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * 1280), int((self._y - self._h / 2) * 720)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * 1280), int((self._y + self._h / 2) * 720)

    @property
    def center(self):
        return int(self._x * 1280), int(self._y * 720)

    # @property     
    # def text(self):
    #     return self._info


# markers = []

mask1 = detactfile.mask1

def on_detect_marker(mask1):
    # number = len(marker_info)
    # markers.clear()
    # for i in range(0, number):
        # x, y, w, h, info = marker_info[i]
    matchT = cv2.matchTemplate(img_process, mask1, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchT)
    top_left = max_loc
    h, w = mask1.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left,bottom_right

    # markers.append(MarkerInfo(x, y, w, h))
    
    # print("marker:{0} x:{1}, y:{2}, w:{3}, h:{4}".format(info, x, y, w, h))


def process_img(img):
    img[600:,:] = (0,0,0)
    blur = cv2.GaussianBlur(img, (7,7), 0)
    img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    img_bi1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
    img_bi2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)
    img_bi = cv2.bitwise_or(img_bi1, img_bi2)
    return img_bi


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_gimbal.moveto(pitch=-15, yaw=0).wait_for_completed()

    ep_camera.start_video_stream(display=False)
    # result = ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    
    for i in range(0, 500):
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        img_process = process_img(img)

        # for j in range(0, len(markers)):
        top_left,bottom_right = on_detect_marker(mask1)
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, top_left, bottom_right, (0, 255, 0), 3)
        # cv2.rectangle(img, markers[j].pt1, markers[j].pt2, (255, 255, 255))
            # cv2.putText(img, markers[j].text, markers[j].center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow("Markers", img_with_box)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    result = ep_vision.unsub_detect_info(name="marker")
    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_robot.close()
