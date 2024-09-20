import cv2

img_dir = '/home/user/Study/crossfit/mediapipe-samples/241_003_dumbbell-snatch-right_009.mp4'

cap = cv2.VideoCapture(img_dir)
# 프레임을 반복적으로 읽고 화면에 출력
while True:
    ret, frame = cap.read()

    # 프레임을 성공적으로 읽었는지 확인
    if not ret:
        print("End of video stream.")
        break

    # 프레임을 화면에 표시
    cv2.imshow('Video Playback', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 모든 리소스 해제
cap.release()
cv2.destroyAllWindows()




