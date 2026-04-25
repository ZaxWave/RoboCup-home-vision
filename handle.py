import cv2
import mediapipe as mp
import math
import time
from collections import deque


def points_cos_angle(point1, point2):
    try:
        angle_ = math.degrees(math.acos(
            (point1[0] * point2[0] + point1[1] * point2[1]) / (
                    ((point1[0] ** 2 + point1[1] ** 2) * (point2[0] ** 2 + point2[1] ** 2)) ** 0.5)))
    except:
        angle_ = 65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_


def get_fingers_angle(handPoints_list):
    angle_list = []
    # thumb
    angle_list.append(points_cos_angle(
        (int(handPoints_list[0][0]) - int(handPoints_list[2][0]),
         int(handPoints_list[0][1]) - int(handPoints_list[2][1])),
        (int(handPoints_list[3][0]) - int(handPoints_list[4][0]),
         int(handPoints_list[3][1]) - int(handPoints_list[4][1]))
    ))
    # index
    angle_list.append(points_cos_angle(
        (int(handPoints_list[0][0]) - int(handPoints_list[6][0]),
         int(handPoints_list[0][1]) - int(handPoints_list[6][1])),
        (int(handPoints_list[7][0]) - int(handPoints_list[8][0]),
         int(handPoints_list[7][1]) - int(handPoints_list[8][1]))
    ))
    # middle
    angle_list.append(points_cos_angle(
        (int(handPoints_list[0][0]) - int(handPoints_list[10][0]),
         int(handPoints_list[0][1]) - int(handPoints_list[10][1])),
        (int(handPoints_list[11][0]) - int(handPoints_list[12][0]),
         int(handPoints_list[11][1]) - int(handPoints_list[12][1]))
    ))
    # ring
    angle_list.append(points_cos_angle(
        (int(handPoints_list[0][0]) - int(handPoints_list[14][0]),
         int(handPoints_list[0][1]) - int(handPoints_list[14][1])),
        (int(handPoints_list[15][0]) - int(handPoints_list[16][0]),
         int(handPoints_list[15][1]) - int(handPoints_list[16][1]))
    ))
    # pinky
    angle_list.append(points_cos_angle(
        (int(handPoints_list[0][0]) - int(handPoints_list[18][0]),
         int(handPoints_list[0][1]) - int(handPoints_list[18][1])),
        (int(handPoints_list[19][0]) - int(handPoints_list[20][0]),
         int(handPoints_list[19][1]) - int(handPoints_list[20][1]))
    ))
    return angle_list


def get_hand_gesture(fingers_angle_list):
    thr_bend = 60.
    thr_thumb = 45.
    thr_straight = 20.
    gesture_str = None
    if 65535. not in fingers_angle_list:
        f = fingers_angle_list
        if f[0] > thr_thumb:
            if f[1] > thr_bend and f[2] > thr_bend and f[3] > thr_bend and f[4] > thr_bend:
                gesture_str = "fist"
            elif f[1] < thr_straight and f[2] < thr_straight and f[3] < thr_straight and f[4] < thr_straight:
                gesture_str = "four"
            elif f[1] < thr_straight and f[2] < thr_straight and f[3] < thr_straight and f[4] > thr_bend:
                gesture_str = "three"
            elif f[1] < thr_straight and f[2] < thr_straight and f[3] > thr_bend and f[4] > thr_bend:
                gesture_str = "two"
            elif f[1] < thr_straight and f[2] > thr_bend and f[3] > thr_bend and f[4] > thr_bend:
                gesture_str = "one"
        elif f[0] < thr_straight:
            if f[1] < thr_straight and f[2] < thr_straight and f[3] < thr_straight and f[4] < thr_straight:
                gesture_str = "five"
            elif f[1] > thr_bend and f[2] > thr_bend and f[3] > thr_bend and f[4] > thr_bend:
                gesture_str = "thumbUp"
    return gesture_str


def get_pointing_direction(clm, fingers_angle_list):
    """
    检测食指指向方向。
    仅在食指伸直、其余三指弯曲时（即 one 手势）才判定为指向。
    返回 (angle_deg, direction_str) 或 None。
    angle_deg: 0=右, 90=上, 180=左, 270=下（屏幕坐标系已修正镜像）
    """
    if fingers_angle_list is None or 65535. in fingers_angle_list:
        return None
    f = fingers_angle_list
    thr_bend = 60.
    thr_straight = 20.
    # 食指伸直，中/无名/小指弯曲（拇指不限）
    if not (f[1] < thr_straight and f[2] > thr_bend and f[3] > thr_bend and f[4] > thr_bend):
        return None

    # 向量：食指MCP(5) → 指尖(8)，屏幕y轴向下需取反才符合直觉（上为正）
    # 同时因为画面已镜像翻转显示，x方向也取反
    dx = -(clm[8][0] - clm[5][0])  # 镜像修正
    dy = -(clm[8][1] - clm[5][1])  # y轴修正：屏幕向下为正，取反后向上为正

    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    nx, ny = dx / length, dy / length  # 归一化方向向量

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360

    if 45 <= angle_deg < 135:
        direction = "up"
    elif 135 <= angle_deg < 225:
        direction = "left"
    elif 225 <= angle_deg < 315:
        direction = "down"
    else:
        direction = "right"

    return angle_deg, direction, (nx, ny)


class WaveDetector:
    """
    检测挥手动作：在 time_window 秒内，手腕水平方向反转 >= min_reversals 次，
    且每次移动幅度超过 min_distance 像素，则判定为挥手。
    """
    def __init__(self, time_window=1.5, min_reversals=3, min_distance=40):
        self.time_window = time_window
        self.min_reversals = min_reversals
        self.min_distance = min_distance
        self._positions = deque()  # (timestamp, x)
        self._last_direction = 0
        self._reversals = 0
        self._last_peak_x = None

    def update(self, x, timestamp):
        self._positions.append((timestamp, x))
        while self._positions and timestamp - self._positions[0][0] > self.time_window:
            self._positions.popleft()

        if self._last_peak_x is None:
            self._last_peak_x = x
            return False

        dx = x - self._last_peak_x
        if abs(dx) < self.min_distance:
            return False

        direction = 1 if dx > 0 else -1
        if direction != self._last_direction and self._last_direction != 0:
            self._reversals += 1
        self._last_direction = direction
        self._last_peak_x = x

        if self._reversals >= self.min_reversals:
            self._reversals = 0
            return True
        return False

    def reset(self):
        self._positions.clear()
        self._last_direction = 0
        self._reversals = 0
        self._last_peak_x = None


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    wave_detector = WaveDetector()
    wave_display_until = 0
    previous_time = time.time()
    gesture_str = None

    while True:
        success, img = cap.read()
        if not success:
            continue

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        current_time = time.time()
        gesture_str = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                clm = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]

                # 食指指尖标记
                cv2.circle(img, (int(clm[8][0]), int(clm[8][1])), 8, (0, 0, 255), -1)

                # 手势识别（需先于挥手检测，用于排除指向动作）
                angle_list = get_fingers_angle(clm)
                gesture_str = get_hand_gesture(angle_list)

                # 挥手检测：指向时跳过并重置，避免误触发
                pointing = get_pointing_direction(clm, angle_list)
                if pointing:
                    wave_detector.reset()
                else:
                    wrist_x = clm[0][0]
                    if wave_detector.update(wrist_x, current_time):
                        wave_display_until = current_time + 2.0
                        print("挥手检测到！")

                if pointing:
                    gesture_str = None
                    angle_deg, direction, (nx, ny) = pointing
                    print(f"pointing: {direction}  vec=({nx:.3f}, {ny:.3f})  angle={angle_deg:.1f}deg")
                    tip = (int(clm[8][0]), int(clm[8][1]))
                    mcp = (int(clm[5][0]), int(clm[5][1]))
                    cv2.arrowedLine(img, mcp, tip, (0, 255, 255), 3, tipLength=0.4)
                    cv2.putText(img, f"point:{direction} ({nx:.2f},{ny:.2f})",
                                (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        img = cv2.flip(img, 1)

        fps = 1 / (current_time - previous_time) if previous_time else 0
        previous_time = current_time

        cv2.putText(img, f"{int(fps)} FPS", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        if current_time < wave_display_until:
            cv2.putText(img, "WAVING!", (10, 75), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        elif gesture_str:
            cv2.putText(img, gesture_str, (10, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Hand Gesture", img)
        if cv2.waitKey(2) & 0xFF == 27:
            break
        if cv2.getWindowProperty("Hand Gesture", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
