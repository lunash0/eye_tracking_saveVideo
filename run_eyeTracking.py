"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
# 1.불필요한계산줄이기
# 2.비디오해상도 낮춰작업햇다가 필요히 출력할때원래해상도로
# 3.코랩쥐피유사용해보기

import cv2
import time
import os
from gaze_tracking import GazeTracking
import functions

gaze = GazeTracking()

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, 'input_videos', 'test_video_0.mp4')

video = cv2.VideoCapture(video_path)  # 'video capture object'인 video 생성
fps = int(video.get(cv2.CAP_PROP_FPS))  # fps == Frames Per Second
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_directory = os.path.join(current_dir, 'output_videos')
output_file_path = os.path.join(output_directory, 'output.mp4')
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # setting video codec (A codec specifies the algorithm used to compress and decode video.)
output = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

blink_counter = 0
blink_msg_shown = False
start_time = time.time()

frame_id = 0

while True:
    # 비디오에서 새로운 프레임을 가져옴
    ret, frame = video.read()
    if not ret:
        break

    # frame = functions.crop(frame)
    # 프레임을 GazeTracking에 전달하여 분석
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
        blink_counter += 1

    if time.time() - start_time >= 2:
        if blink_counter >= 3 and not blink_msg_shown:
            cv2.putText(frame, "Blink too often!", (90, 180), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            blink_message_shown = True
            start_time = time.time()
        else:
            blink_message_shown = False
            blink_counter = 0

    if gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 110), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 145), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)

    # 결과를 출력 및 저장
    # cv2.imshow("Gaze Tracking", frame)
    output.write(frame)

    if cv2.waitKey(1) == 27:
        break

video.release()  # video capture 객체 해제: 리소스 누수 방지
output.release()  # video capture 객체 해제: 리소스 누수 방지
cv2.destroyAllWindows()
