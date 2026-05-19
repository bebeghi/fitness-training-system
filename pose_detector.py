"""
스쿼트 감지기 - 3~4단계
- MediaPipe Pose로 관절 좌표 추출 및 스켈레톤 표시
- 엉덩이-무릎-발목 각도 계산
- UP/DOWN 상태 판정 + 횟수 카운트
"""

import cv2
import pygame
import mediapipe as mp
import numpy as np
import sys

# ── 설정 ──────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1280, 720   # 전체 창 크기
CAM_W,    CAM_H    = 860, 720    # 왼쪽 웹캠 영역
PANEL_W            = WINDOW_W - CAM_W  # 오른쪽 패널 너비 (420)

# 스쿼트 각도 기준 (사이드뷰 기준)
ANGLE_DOWN = 110   # 이 값 이하면 DOWN (충분히 내려간 상태)
ANGLE_UP   = 155   # 이 값 이상이면 UP (선 상태)

# 색상
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GREEN  = ( 50, 205,  50)
RED    = (220,  50,  50)
YELLOW = (255, 215,   0)
CYAN   = (  0, 220, 220)
GRAY   = ( 40,  40,  40)
DARK   = ( 20,  20,  30)

# ── MediaPipe 초기화 ───────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ── 각도 계산 함수 ─────────────────────────────────────
def calc_angle(a, b, c):
    """
    세 점 a-b-c에서 b를 꼭짓점으로 하는 각도(도) 반환
    a, b, c : (x, y) 튜플
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

# ── 랜드마크 좌표 추출 헬퍼 ────────────────────────────
def get_landmark_px(landmarks, idx, w, h):
    """랜드마크 인덱스 → 픽셀 좌표 (x, y)"""
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))

# ── Pygame 텍스트 렌더 헬퍼 ────────────────────────────
def draw_text(surface, text, pos, font, color=WHITE, center=False):
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(img, rect)

# ── OpenCV 프레임 → Pygame Surface 변환 ───────────────
def cv2_to_pygame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb, k=3)          # 세로 → 가로 보정
    surface   = pygame.surfarray.make_surface(frame_rgb)
    surface   = pygame.transform.flip(surface, True, False)
    return surface

# ── 메인 ──────────────────────────────────────────────
def main():
    # Pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Squat Detector")
    clock  = pygame.time.Clock()

    font_lg = pygame.font.SysFont("Arial", 48, bold=True)
    font_md = pygame.font.SysFont("Arial", 30, bold=True)
    font_sm = pygame.font.SysFont("Arial", 22)

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        sys.exit(1)

    # 상태 변수
    squat_count = 0
    state       = "UP"      # "UP" | "DOWN"
    angle_knee  = 180.0
    feedback    = "준비하세요"
    feedback_color = WHITE

    running = True
    while running:
        # ── 이벤트 처리 ───────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # ── 웹캠 프레임 읽기 ──────────────────────────
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)   # 좌우 반전 (거울 모드)

        # ── MediaPipe 자세 분석 ───────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = pose.process(frame_rgb)

        detected = results.pose_landmarks is not None

        if detected:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]

            # 오른쪽 관절 좌표 추출 (카메라 정면 기준 → 사용자 오른쪽)
            hip   = get_landmark_px(landmarks, mp_pose.PoseLandmark.RIGHT_HIP,   w, h)
            knee  = get_landmark_px(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE,  w, h)
            ankle = get_landmark_px(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)

            # 왼쪽도 계산해서 더 신뢰도 높은 쪽 사용
            l_hip   = get_landmark_px(landmarks, mp_pose.PoseLandmark.LEFT_HIP,   w, h)
            l_knee  = get_landmark_px(landmarks, mp_pose.PoseLandmark.LEFT_KNEE,  w, h)
            l_ankle = get_landmark_px(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)

            angle_r = calc_angle(hip,   knee,  ankle)
            angle_l = calc_angle(l_hip, l_knee, l_ankle)
            angle_knee = min(angle_r, angle_l)   # 더 많이 굽힌 쪽 기준

            # ── 스쿼트 상태 판정 ──────────────────────
            if state == "UP" and angle_knee < ANGLE_DOWN:
                state    = "DOWN"
                feedback = "좋아요! 올라오세요 ↑"
                feedback_color = GREEN

            elif state == "DOWN" and angle_knee > ANGLE_UP:
                state       = "UP"
                squat_count += 1
                feedback    = f"{squat_count}회 완료! 🎉"
                feedback_color = CYAN

            elif state == "UP" and angle_knee < ANGLE_UP:
                feedback = "조금 더 내려가세요 ↓"
                feedback_color = YELLOW

            # ── OpenCV 프레임에 스켈레톤 그리기 ──────
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 120), thickness=3, circle_radius=5),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2),
            )

            # 무릎 각도 텍스트를 웹캠 프레임에 직접 표시
            cv2.putText(
                frame, f"{int(angle_knee)}°",
                (knee[0] + 10, knee[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2
            )

            # 엉덩이-무릎-발목 강조 선
            cv2.line(frame, hip,   knee,  (0, 200, 255), 3)
            cv2.line(frame, knee,  ankle, (0, 200, 255), 3)
            cv2.circle(frame, knee,  10, (255, 100,   0), -1)
            cv2.circle(frame, hip,    7, (  0, 200, 255), -1)
            cv2.circle(frame, ankle,  7, (  0, 200, 255), -1)

        # ── OpenCV 프레임 → Pygame Surface ────────────
        cam_surface = cv2.resize(frame, (CAM_W, CAM_H))
        pg_surface  = pygame.surfarray.make_surface(
            cv2.cvtColor(cam_surface, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )

        # ── Pygame 화면 그리기 ────────────────────────
        screen.fill(DARK)

        # 웹캠 영역
        screen.blit(pg_surface, (0, 0))

        # 오른쪽 패널 배경
        panel_rect = pygame.Rect(CAM_W, 0, PANEL_W, WINDOW_H)
        pygame.draw.rect(screen, GRAY, panel_rect)
        pygame.draw.line(screen, CYAN, (CAM_W, 0), (CAM_W, WINDOW_H), 2)

        px = CAM_W + 20   # 패널 왼쪽 여백
        py = 30

        # 제목
        draw_text(screen, "SQUAT", (CAM_W + PANEL_W // 2, py + 10),
                  font_lg, CYAN, center=True)
        draw_text(screen, "DETECTOR", (CAM_W + PANEL_W // 2, py + 60),
                  font_lg, WHITE, center=True)
        py += 120

        pygame.draw.line(screen, CYAN, (CAM_W + 10, py), (WINDOW_W - 10, py), 1)
        py += 20

        # 횟수
        draw_text(screen, "스쿼트 횟수", (px, py), font_sm, YELLOW)
        py += 30
        draw_text(screen, str(squat_count), (CAM_W + PANEL_W // 2, py + 20),
                  font_lg, WHITE, center=True)
        py += 80

        pygame.draw.line(screen, GRAY, (CAM_W + 10, py), (WINDOW_W - 10, py), 1)
        py += 20

        # 상태
        state_color = GREEN if state == "DOWN" else WHITE
        draw_text(screen, "현재 상태", (px, py), font_sm, YELLOW)
        py += 30
        draw_text(screen, state, (CAM_W + PANEL_W // 2, py + 10),
                  font_md, state_color, center=True)
        py += 60

        pygame.draw.line(screen, GRAY, (CAM_W + 10, py), (WINDOW_W - 10, py), 1)
        py += 20

        # 무릎 각도
        draw_text(screen, "무릎 각도", (px, py), font_sm, YELLOW)
        py += 30

        angle_color = GREEN if angle_knee < ANGLE_DOWN else RED
        draw_text(screen, f"{int(angle_knee)}°",
                  (CAM_W + PANEL_W // 2, py + 10),
                  font_md, angle_color, center=True)
        py += 60

        # 각도 게이지 바
        bar_x, bar_y = px, py
        bar_w, bar_h = PANEL_W - 40, 20
        ratio = 1.0 - np.clip((angle_knee - ANGLE_DOWN) / (ANGLE_UP - ANGLE_DOWN), 0, 1)
        pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_w, bar_h), border_radius=10)
        pygame.draw.rect(screen, GREEN if ratio > 0.6 else YELLOW,
                         (bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=10)
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 2, border_radius=10)
        py += 40

        pygame.draw.line(screen, GRAY, (CAM_W + 10, py), (WINDOW_W - 10, py), 1)
        py += 20

        # 피드백
        draw_text(screen, "피드백", (px, py), font_sm, YELLOW)
        py += 35

        # 피드백 문구가 길면 줄바꿈
        words = feedback
        draw_text(screen, words, (CAM_W + PANEL_W // 2, py),
                  font_sm, feedback_color, center=True)
        py += 40

        # 사람 미감지 경고
        if not detected:
            draw_text(screen, "⚠ 화면에 서 주세요",
                      (CAM_W + PANEL_W // 2, WINDOW_H - 80),
                      font_sm, RED, center=True)

        # 종료 안내
        draw_text(screen, "Q 키: 종료",
                  (CAM_W + PANEL_W // 2, WINDOW_H - 30),
                  font_sm, (130, 130, 130), center=True)

        pygame.display.flip()
        clock.tick(30)

    # 종료 처리
    cap.release()
    pose.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()