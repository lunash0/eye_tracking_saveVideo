from __future__ import division  # __future__ 모듈을 사용 -> 파이썬 2, 3의 나눗셈 동작을 통일
import cv2
from pupil import Pupil


# 클래스 'Calibration': 눈동자 검출 알고리즘을 보정하는 역할
class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    # 초기화 메서드 '__init_'
    def __init__(self):
        self.nb_frames = 20  # 보정을 위해 사용할 프레임 수. 여기서는 20 프레임 시용
        self.thresholds_left = []  # 왼쪽 눈의 이진화 임계값 저장할 리스트
        self.thresholds_right = []  # 오른쪽 눈의 이진화 임계값 저장할 리스트

    # 메서드 'is_complete': 보정이 완료되었는지 여부 확인
    def is_complete(self):
        """Returns true if the calibration is completed"""
        # thresholds_left와 thresholds_right list의 길이가 nb_frames와 같거나 크면 보정이 완료된 것으로 간주하고 True를 반환
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    # 메서드 'threshold': 왼쪽, 오른쪽 눈에 대한 이 진화 임계값 반환
    def threshold(self, side):
        """Returns the threshold value for the given eye.

        Argument:
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if side == 0:  # 0은 왼쪽 눈
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:  # 1은 오른쪽 눈
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    # 메서드 'iris_size': 이진화된 동공 프레임에서 동공이 차지하는 면적의 밴분율 반환
    @staticmethod
    def iris_size(frame):
        """Returns the percentage of space that the iris takes up on
        the surface of the eye.

        Argument:
            frame (numpy.ndarray): Binarized iris frame
        """
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    # 메서드 'find_best_threshold': 주어진 눈 이미지 'eye_frame'에 대해 최적의 이진화 임계값 찾기
    @staticmethod
    def find_best_threshold(eye_frame):
        """Calculates the optimal threshold to binarize the
        frame for the given eye.

        Argument:
            eye_frame (numpy.ndarray): Frame of the eye to be analyzed
        """
        average_iris_size = 0.48  # 평균 동공 면적을 0.48로 가정
        trials = {}  # dictionary 'trials': key는 임계값, value는 각 임계값에 대한 동공 면적

        # 5~95까지 5 단위로 변화시키면서 각 임계값에 대한 동공 면적 계산 후 trials에 저장
        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        # min()을 통해 trials dic에서 평균 동공 면적과 가장 가까운 임계점 찾기
        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    # 메서드 'evaluate': parm은 눈 이미지 'eye_frame', 왼쪽 또는 오른쪽 눈. 보정 개선.
    def evaluate(self, eye_frame, side):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        threshold = self.find_best_threshold(eye_frame)  # 눈 이미지에 대한 최적의 이진화 임계값 찾아 저장

        if side == 0:  # 0은 왼쪽 눈
            self.thresholds_left.append(threshold)  # 이진화 임계값을 리스트에 추가
        elif side == 1:  # 1은 오른쪽 눈
            self.thresholds_right.append(threshold)  # 이진화 임계값을 리스트에 추가
