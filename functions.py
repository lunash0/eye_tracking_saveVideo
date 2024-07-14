import cv2


def crop(video_path):
    # 비디오 파일 열기
    video = cv2.VideoCapture(video_path)

    # 비디오의 프레임 크기 및 FPS 가져오기
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height), isColor=True)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 주위 부분을 강제적으로 트리밍
        h, w = frame.shape[:2]
        h1, h2 = int(h * 0.05), int(h * 0.95)
        w1, w2 = int(w * 0.05), int(w * 0.95)
        cropped_frame = frame[h1: h2, w1: w2]

        # 결과 프레임 저장
        output_video.write(cropped_frame)

    # 비디오 파일 닫기
    video.release()

    # 여백이 제거된 비디오 파일 반환
    return 'output_video.mp4'