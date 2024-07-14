import math
import numpy as np
import cv2
from pupil import Pupil


# 클래스 'Eye': 눈을 검출하고 해당 눈을 분석
class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """
    # 왼쪽 & 오른쪽 눈의 point index 정의
    # 이 point index는 68개의 얼굴 랜드마크 포인트 중 눈을 지정
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    # 초기화 메서드 '__init__'
    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None  # 눈 영역 이미지
        self.origin = None  # 눈 영역 이미지의 원본 이미지에서의 위치 정보
        self.center = None  # 눈 영역 이미지의 중심 좌표
        self.pupil = None  # 눈동자 객체
        self.landmark_points = None  # 눈 영역의 랜드마크 포인트 정보

        self._analyze(original_frame, landmarks, side, calibration)  # _analyze 메서드를 호출->눈 분석 및 눈동자 검출 수행

    # 메서드 '_middle_point': 두 점 사이의 중간점을 계산하는 정적 메서드
    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)

        return (x, y)

    # 메서드 '_isolate': 얼굴 이미지에서 눈 영역만 분리하고 이진 마스크를 적용하여 눈 영역 이미지 생성
    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        # landmarks에 있는 눈의 포인트 좌표를 사용하여 눈 영역 나타내는 다각형 영역 정의 후 region 변수에 저장
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]  # 원본 이미지의 크기 가져오기
        black_frame = np.zeros((height, width), np.uint8)  # 눈 영역 추출을 위한 빈 검은색 이미지
        mask = np.full((height, width), 255, np.uint8)  # 초기에는 모두 흰색(255)로 채워진 마스크
        cv2.fillPoly(mask, [region], (0, 0, 0))  # 눈 영역을 검은색으로 채운 이진 마스크 'mask' 생성
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)  # 마스크를 적용하여 얼굴 이미지에서 눈 영역만 추출

        # Cropping on the eye
        margin = 5  # 눈 영역 주변에 추가적인 여백 설정에 사용
        min_x = np.min(region[:, 0]) - margin  # min_x, max_x, min_y, min_y를 계산->눈 영역 이미지를 원하는 영역으로 crop
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]  # 크롭된 눈 영역 이미지 저장
        self.origin = (min_x, min_y)  # 눈 영역 이미지의 원래 위치 저장

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)  # 눈 영역 이미지의 중심 좌표 저장

    # 주어진 얼굴 랜드마크 및 눈 영역 관련 포인트를 사용해, 눈을 감았는지 여부를 나타낼 수 있는 비율을 계산
    def _blinking_ratio(self, landmarks, points):
        # landmarks: 얼굴 영역의 랜드마크를 포함하는 'dlib.full_object_detection' 객체
        # points: 눈 영역을 나타내는 포인트 list
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)  # 왼쪽 눈의 가장 왼쪽 포인트의 x, y 좌표 가져옴
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)  # 왼쪽 눈의 가장 오른쪽 포인트의 x, y 좌표 가져옴
        # landmarks.part(points[1])부터 landmarks.part(points[5])를 사용 -> 눈의 상하단 중심 좌표 계산
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))
        # 눈의 너비 및 높이 계산
        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height  # ratio는 눈을 감은 정도를 나타내는 비율
        except ZeroDivisionError:  # 눈 높이가 0인 경우(눈을 감은 경우)
            ratio = None

        return ratio

    # 주어진 프레임에서 눈을 감지&분리 후 보정(눈 깜빡힘 비율 및 적절한 이진화 임계값 계산)을 수행하며 Pupil 객체를 초기화하는 함수
    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user  # 사용자로부터 전달받은 원본 프레임인 'numpy.ndarray'
            landmarks (dlib.full_object_detection): Facial landmarks for the face region  # 얼굴 영역의 랜드마크를 포함하는 'dlib.full_object_detection' 객체
            side: Indicates whether it's the left eye (0) or the right eye (1)  # 눈의 위치를 나타내며 왼쪽 눈은 0, 오른쪽 눈은 1
            calibration (calibration.Calibration): Manages the binarization threshold value  # 이진화 임계값을 관리하는 'calibration.Calibration' 객체
        """
        # side 변수 확인을 통해 왼쪽 또는 오른쪽 눈에 해당하는 point list를 points에 할당
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)  # '_blinking_ratio' 메서드를 호출하여 현재 눈의 깜빡임 비율을 계산하고 'self.blinking'에 저장
        self._isolate(original_frame, landmarks, points)  # '_isolate' 메서드를 호출하여 현재 눈의 깜빡임 비율을 계산하고 'self.blinking'에 저장

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
