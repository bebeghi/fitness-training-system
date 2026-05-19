import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()

    # 일시적으로 프레임을 못 받으면 바로 종료하지 않고 다음 루프로 넘어감
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()