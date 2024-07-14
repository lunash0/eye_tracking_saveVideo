import numpy as np
import cv2

# 클래스 'Pupil': 눈의 동공을 감지하고 해당 동공의 위치를 추정
class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    # 초기화 메서드 '__init__'
    def __init__(self, eye_frame, threshold):  # 매개변수: 눈이미지 프레임, 임계값
        self.iris_frame = None  # 동공을 감지한 결과 frame 저장을 위한 변수
        self.threshold = threshold  # image 이진화에 사용할 임계값 저장 변수
        self.x = None  # 동공의 추정 위치 중 x 좌표 저장 변수
        self.y = None  # 동공의 추정 위치 중 y 좌표 저장 변수

        self.detect_iris(eye_frame)

    # 메서드 'image_processing': 주어진 눈 이미지인 'eye_frame'을 처리하여 동곰을 감지할 프레임을 반환
    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)  # c2.bilateralFilter을 사용하여 이미지에 양방향 필터링 적용->노이즈 제거
        new_frame = cv2.erode(new_frame, kernel, iterations=3)  # cv2.erode를 사용하여 이미지를 침식시킴(침식 연산)->작은 노이즈 제거
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]  # 이미지 이진화 -> 동공 감지하기 쉽게 만들어줌

        return new_frame

    # 메서드 'detect_iris': 동공 감지 및 위치 추정
    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)  # 이미지 처리된 동공 프레임 저장하는 변수

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]  # 이진화된 이미지에서 윤곽선(contours) 찾아 저장
        contours = sorted(contours, key=cv2.contourArea)  # 윤곽선 넓이에 따라 정렬

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])  # 동공의 중심 위치 x 좌표
            self.y = int(moments['m01'] / moments['m00'])  # 동공의 중심 위치 y 좌표
        except (IndexError, ZeroDivisionError):  # ZeroDivisionError나 IndexError가 발생하는 경우를 처리
            pass
