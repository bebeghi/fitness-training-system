"""
스쿼트 자세 교정 시스템
────────────────────────────────────────────
[모드 흐름]
  CALIBRATION  : 처음 3회 스쿼트를 자동 학습 → 개인 기준값 생성
  READY        : 학습 완료, 운동 시작 대기 (SPACE)
  EXERCISE     : 실제 운동 — 잘못된 자세면 경고 + 카운트 제외

[판정 기준 — 모두 개인화된 기준값 대비 허용 범위로 비교]
  ① 무릎 각도  : 기준 DOWN 각도 ± 허용치로 UP/DOWN 판정
  ② 허리 굽음  : 어깨 중점-엉덩이 중점-수직축 각도가 기준 대비 크게 기울면 경고
  ③ 고관절 위치: DOWN 시 엉덩이 x좌표가 기준 대비 너무 앞/뒤로 벗어나면 경고
"""

import cv2
import pygame
import mediapipe as mp
import numpy as np
import sys
import os
from collections import deque


# ══════════════════════════════════════════════════════
#  한글 폰트 로더 (macOS)
# ══════════════════════════════════════════════════════
def load_korean_font(size):
    """
    macOS 시스템 한글 폰트를 순서대로 탐색하여 반환.
    AppleSDGothicNeo → AppleGothic → Arial Unicode → 시스템 폴백
    """
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/ArialHB.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                continue
    # 최후 폴백 — 시스템 폰트
    for name in ["applegothic", "arial", "freesansbold"]:
        try:
            f = pygame.font.SysFont(name, size)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.Font(None, size)

# ── 화면 설정 ──────────────────────────────────────────
WINDOW_W, WINDOW_H = 1280, 720
CAM_W,    CAM_H    =  860, 720
PANEL_W            = WINDOW_W - CAM_W   # 420

# ── 색상 ───────────────────────────────────────────────
WHITE   = (255, 255, 255)
BLACK   = (  0,   0,   0)
GREEN   = ( 50, 220,  80)
RED     = (220,  60,  60)
YELLOW  = (255, 210,   0)
CYAN    = (  0, 210, 220)
ORANGE  = (255, 140,   0)
GRAY    = ( 45,  45,  55)
DARK    = ( 18,  18,  28)
PANEL_BG= ( 28,  28,  40)

# ── 허용 오차 (비율 또는 각도) ─────────────────────────
KNEE_ANGLE_TOLERANCE  = 15    # 무릎 각도 허용 오차 (도)
BACK_ANGLE_TOLERANCE  = 12    # 허리 기울기 허용 오차 (도)
HIP_X_TOLERANCE       = 0.08  # 고관절 x 이동 허용 비율 (정규화 좌표 기준)

# 캘리브레이션 목표 횟수
CALIB_REPS = 3

# ── MediaPipe ─────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)


# ══════════════════════════════════════════════════════
#  수학 유틸
# ══════════════════════════════════════════════════════
def calc_angle(a, b, c):
    """b를 꼭짓점으로 a-b-c 각도(도)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_v   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))


def vertical_tilt(top, bottom):
    """
    top → bottom 벡터와 '수직 아래(0,1)' 사이 각도(도)
    허리가 앞으로 기울수록 커짐
    """
    vec = np.array([bottom[0] - top[0], bottom[1] - top[1]], dtype=float)
    vertical = np.array([0.0, 1.0])
    cos_v = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))


def lm_xy(landmarks, idx):
    """랜드마크 정규화 좌표 (x, y) — 0~1"""
    lm = landmarks[idx]
    return (lm.x, lm.y)


def lm_px(landmarks, idx, w, h):
    """랜드마크 픽셀 좌표"""
    x, y = lm_xy(landmarks, idx)
    return (int(x * w), int(y * h))


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ══════════════════════════════════════════════════════
#  캘리브레이션 데이터 클래스
# ══════════════════════════════════════════════════════
class CalibData:
    def __init__(self):
        self.reset()

    def reset(self):
        # 수집된 DOWN 프레임들의 측정값 리스트
        self.knee_angles_down  = []   # 무릎 각도 (DOWN 구간)
        self.back_angles_down  = []   # 허리 기울기 (DOWN 구간)
        self.hip_x_down        = []   # 엉덩이 x (DOWN 구간, 정규화)
        self.knee_angle_up_ref = []   # UP 구간 무릎 각도
        self.reps_collected    = 0
        self.is_done           = False

        # 최종 기준값
        self.ref_knee_down  = None
        self.ref_knee_up    = None
        self.ref_back_down  = None
        self.ref_hip_x_down = None

    def add_down_frame(self, knee_angle, back_angle, hip_x):
        self.knee_angles_down.append(knee_angle)
        self.back_angles_down.append(back_angle)
        self.hip_x_down.append(hip_x)

    def add_up_frame(self, knee_angle):
        self.knee_angle_up_ref.append(knee_angle)

    def finalize_rep(self):
        self.reps_collected += 1
        if self.reps_collected >= CALIB_REPS:
            self._compute_refs()
            self.is_done = True

    def _compute_refs(self):
        """수집된 샘플 평균으로 기준값 확정"""
        self.ref_knee_down  = float(np.mean(self.knee_angles_down))
        self.ref_back_down  = float(np.mean(self.back_angles_down))
        self.ref_hip_x_down = float(np.mean(self.hip_x_down))
        self.ref_knee_up    = float(np.mean(self.knee_angle_up_ref)) \
                              if self.knee_angle_up_ref else 160.0

    def summary(self):
        return (
            f"무릎(DOWN)={self.ref_knee_down:.1f}°  "
            f"무릎(UP)={self.ref_knee_up:.1f}°  "
            f"허리={self.ref_back_down:.1f}°  "
            f"고관절x={self.ref_hip_x_down:.3f}"
        )


# ══════════════════════════════════════════════════════
#  자세 분석 함수
# ══════════════════════════════════════════════════════
def analyze_pose(landmarks, w, h):
    """
    현재 프레임에서 주요 각도/좌표 추출.
    반환: dict
    """
    # ── 픽셀 좌표
    r_hip   = lm_px(landmarks, mp_pose.PoseLandmark.RIGHT_HIP,    w, h)
    r_knee  = lm_px(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE,   w, h)
    r_ankle = lm_px(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE,  w, h)
    l_hip   = lm_px(landmarks, mp_pose.PoseLandmark.LEFT_HIP,     w, h)
    l_knee  = lm_px(landmarks, mp_pose.PoseLandmark.LEFT_KNEE,    w, h)
    l_ankle = lm_px(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE,   w, h)
    r_sho   = lm_px(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
    l_sho   = lm_px(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER,  w, h)

    # ── 정규화 좌표 (x 이동 판정용)
    r_hip_n = lm_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    l_hip_n = lm_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP)

    # ── 무릎 각도 (좌우 중 더 굽힌 쪽)
    angle_r = calc_angle(r_hip, r_knee, r_ankle)
    angle_l = calc_angle(l_hip, l_knee, l_ankle)
    knee_angle = min(angle_r, angle_l)

    # ── 허리 기울기: 어깨 중점 → 엉덩이 중점과 수직축 사이 각도
    sho_mid = midpoint(r_sho, l_sho)
    hip_mid = midpoint(r_hip, l_hip)
    back_angle = vertical_tilt(sho_mid, hip_mid)

    # ── 엉덩이 x 정규화 좌표 평균
    hip_x_norm = (r_hip_n[0] + l_hip_n[0]) / 2.0

    return {
        "knee_angle" : knee_angle,
        "back_angle" : back_angle,
        "hip_x_norm" : hip_x_norm,
        # 픽셀 좌표 (그리기용)
        "r_hip": r_hip, "r_knee": r_knee, "r_ankle": r_ankle,
        "l_hip": l_hip, "l_knee": l_knee, "l_ankle": l_ankle,
        "sho_mid": sho_mid, "hip_mid": hip_mid,
    }


# ══════════════════════════════════════════════════════
#  자세 오류 판정
# ══════════════════════════════════════════════════════
def check_errors(metrics, calib, state):
    """
    errors : list of str  (빈 리스트면 정상)
    """
    errors = []
    if state != "DOWN":
        return errors   # DOWN 구간에서만 정밀 판정

    # ① 허리 굽음
    if metrics["back_angle"] > calib.ref_back_down + BACK_ANGLE_TOLERANCE:
        errors.append("허리가 너무 앞으로 굽었어요")

    # ② 고관절(엉덩이) 위치
    hip_diff = abs(metrics["hip_x_norm"] - calib.ref_hip_x_down)
    if hip_diff > HIP_X_TOLERANCE:
        direction = "앞" if metrics["hip_x_norm"] < calib.ref_hip_x_down else "뒤"
        errors.append(f"엉덩이가 기준보다 {direction}로 벗어났어요")

    return errors


# ══════════════════════════════════════════════════════
#  Pygame 그리기 유틸
# ══════════════════════════════════════════════════════
def draw_text(surface, text, pos, font, color=WHITE, center=False):
    img  = font.render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(img, rect)


def draw_bar(surface, rect, ratio, fg_color, bg_color=BLACK):
    x, y, w, h = rect
    pygame.draw.rect(surface, bg_color, (x, y, w, h), border_radius=8)
    if ratio > 0:
        pygame.draw.rect(surface, fg_color,
                         (x, y, int(w * ratio), h), border_radius=8)
    pygame.draw.rect(surface, WHITE, (x, y, w, h), 2, border_radius=8)


def overlay_skeleton(frame, metrics, errors):
    """웹캠 프레임에 핵심 관절·선 오버레이"""
    color = (0, 60, 220) if errors else (0, 220, 100)

    for (a, b) in [
        (metrics["r_hip"],  metrics["r_knee"]),
        (metrics["r_knee"], metrics["r_ankle"]),
        (metrics["l_hip"],  metrics["l_knee"]),
        (metrics["l_knee"], metrics["l_ankle"]),
    ]:
        cv2.line(frame, a, b, color, 3)

    # 허리 선
    sho = tuple(map(int, metrics["sho_mid"]))
    hip = tuple(map(int, metrics["hip_mid"]))
    back_color = (220, 60, 60) if errors and any("허리" in e for e in errors) else (255, 200, 0)
    cv2.line(frame, sho, hip, back_color, 3)

    for pt in [metrics["r_knee"], metrics["l_knee"],
               metrics["r_hip"],  metrics["l_hip"],
               metrics["r_ankle"], metrics["l_ankle"]]:
        cv2.circle(frame, pt, 7, color, -1)

    # 무릎 각도 텍스트
    kx, ky = metrics["r_knee"]
    cv2.putText(frame, f"{metrics['knee_angle']:.0f}deg",
                (kx + 12, ky - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)


# ══════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════
def main():
    pygame.init()
    pygame.font.init()   # 한글 폰트 서브시스템 명시적 초기화
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Squat Posture Coach")
    clock = pygame.time.Clock()

    font_xl = load_korean_font(52)
    font_lg = load_korean_font(42)
    font_md = load_korean_font(28)
    font_sm = load_korean_font(21)
    font_xs = load_korean_font(17)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        sys.exit(1)

    # ── 상태 머신 ─────────────────────────────────────
    # CALIBRATION → READY → EXERCISE → (DONE)
    mode         = "CALIBRATION"
    calib        = CalibData()
    squat_state  = "UP"        # UP / DOWN
    squat_count  = 0
    bad_rep      = False       # 현재 rep에 오류 있었는지
    errors       = []          # 현재 프레임 오류 목록
    feedback_msg = ""
    feedback_col = WHITE

    # 스무딩용 각도 버퍼
    knee_buf = deque(maxlen=5)
    back_buf = deque(maxlen=5)

    running = True
    while running:
        # ── 이벤트 ────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE and mode == "READY":
                    mode = "EXERCISE"
                if event.key == pygame.K_r:   # 캘리브레이션 리셋
                    calib.reset()
                    mode        = "CALIBRATION"
                    squat_count = 0
                    squat_state = "UP"
                    bad_rep     = False
                    errors      = []

        # ── 카메라 프레임 ─────────────────────────────
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = pose.process(frame_rgb)
        detected  = results.pose_landmarks is not None

        metrics = None
        if detected:
            lms     = results.pose_landmarks.landmark
            metrics = analyze_pose(lms, w, h)

            # 스무딩
            knee_buf.append(metrics["knee_angle"])
            back_buf.append(metrics["back_angle"])
            metrics["knee_angle"] = float(np.mean(knee_buf))
            metrics["back_angle"] = float(np.mean(back_buf))

            # ── MediaPipe 기본 스켈레톤
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1),
            )

            # ──────────────────────────────────────────
            #  CALIBRATION 모드 로직
            # ──────────────────────────────────────────
            if mode == "CALIBRATION":
                ka = metrics["knee_angle"]

                # UP 구간 샘플 수집
                if squat_state == "UP":
                    calib.add_up_frame(ka)
                    if ka < 120:          # 내려가기 시작
                        squat_state = "DOWN"

                elif squat_state == "DOWN":
                    # DOWN 구간 샘플 수집
                    calib.add_down_frame(
                        ka,
                        metrics["back_angle"],
                        metrics["hip_x_norm"],
                    )
                    if ka > 145:          # 올라옴 → 1회 완료
                        squat_state = "UP"
                        calib.finalize_rep()
                        if calib.is_done:
                            mode = "READY"

                overlay_skeleton(frame, metrics, [])

            # ──────────────────────────────────────────
            #  EXERCISE 모드 로직
            # ──────────────────────────────────────────
            elif mode == "EXERCISE":
                ka = metrics["knee_angle"]

                # DOWN 판정: 기준 무릎 각도 + 허용치
                down_thresh = calib.ref_knee_down + KNEE_ANGLE_TOLERANCE
                up_thresh   = calib.ref_knee_up   - KNEE_ANGLE_TOLERANCE

                if squat_state == "UP" and ka < down_thresh:
                    squat_state = "DOWN"
                    bad_rep     = False   # 새 rep 시작

                elif squat_state == "DOWN":
                    # 오류 판정 (DOWN 구간 내내 체크)
                    frame_errors = check_errors(metrics, calib, squat_state)
                    if frame_errors:
                        bad_rep = True
                        errors  = frame_errors
                    else:
                        errors  = []

                    if ka > up_thresh:   # 올라옴
                        if bad_rep:
                            feedback_msg = "[X] 자세 오류 - 카운트 제외"
                            feedback_col = RED
                        else:
                            squat_count += 1
                            feedback_msg = f"[OK] {squat_count}회 완료!"
                            feedback_col = GREEN
                        squat_state = "UP"
                        bad_rep     = False
                        errors      = []

                overlay_skeleton(frame, metrics, errors)

        # ── 웹캠 → Pygame ─────────────────────────────
        cam_surf = cv2.resize(frame, (CAM_W, CAM_H))
        pg_surf  = pygame.surfarray.make_surface(
            cv2.cvtColor(cam_surf, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )

        # ── Pygame 렌더 ───────────────────────────────
        screen.fill(DARK)
        screen.blit(pg_surf, (0, 0))

        # 오른쪽 패널
        pygame.draw.rect(screen, PANEL_BG, (CAM_W, 0, PANEL_W, WINDOW_H))
        pygame.draw.line(screen, CYAN, (CAM_W, 0), (CAM_W, WINDOW_H), 2)

        cx = CAM_W + PANEL_W // 2   # 패널 중앙 x
        px = CAM_W + 18
        py = 22

        # 제목
        draw_text(screen, "SQUAT COACH", (cx, py + 8), font_md, CYAN, center=True)
        py += 45
        pygame.draw.line(screen, CYAN, (CAM_W + 10, py), (WINDOW_W - 10, py), 1)
        py += 14

        # ── CALIBRATION 패널 ──────────────────────────
        if mode == "CALIBRATION":
            draw_text(screen, "캘리브레이션 중", (cx, py), font_md, YELLOW, center=True)
            py += 38
            prog = calib.reps_collected / CALIB_REPS
            draw_text(screen, f"{calib.reps_collected} / {CALIB_REPS} 회 학습됨",
                      (cx, py), font_sm, WHITE, center=True)
            py += 34
            draw_bar(screen, (px, py, PANEL_W - 36, 18), prog, YELLOW)
            py += 36

            pygame.draw.line(screen, GRAY, (CAM_W + 10, py), (WINDOW_W - 10, py), 1)
            py += 16

            guide_lines = [
                "바른 자세로",
                f"{CALIB_REPS}회 스쿼트를 해주세요.",
                "",
                "이 동작이 기준이 됩니다.",
                "천천히, 정확하게!",
            ]
            for line in guide_lines:
                draw_text(screen, line, (cx, py), font_sm,
                          WHITE if line else WHITE, center=True)
                py += 28

            if detected and metrics:
                py += 10
                pygame.draw.line(screen, GRAY, (CAM_W+10, py), (WINDOW_W-10, py), 1)
                py += 12
                draw_text(screen, "현재 무릎 각도", (px, py), font_xs, YELLOW)
                py += 22
                draw_text(screen, f"{metrics['knee_angle']:.0f}°",
                          (cx, py), font_md, CYAN, center=True)
                py += 36
                draw_text(screen, "허리 기울기", (px, py), font_xs, YELLOW)
                py += 22
                draw_text(screen, f"{metrics['back_angle']:.1f}°",
                          (cx, py), font_md, CYAN, center=True)

        # ── READY 패널 ────────────────────────────────
        elif mode == "READY":
            draw_text(screen, "학습 완료!", (cx, py), font_md, GREEN, center=True)
            py += 40

            draw_text(screen, "기준값", (cx, py), font_sm, YELLOW, center=True)
            py += 28
            refs = [
                ("무릎 DOWN 각도", f"{calib.ref_knee_down:.1f}°"),
                ("무릎 UP  각도",  f"{calib.ref_knee_up:.1f}°"),
                ("허리 기울기",    f"{calib.ref_back_down:.1f}°"),
            ]
            for label, val in refs:
                draw_text(screen, label, (px, py), font_xs, (180, 180, 180))
                draw_text(screen, val, (WINDOW_W - 18, py), font_xs, CYAN,
                          center=False)
                py += 24
            py += 10
            pygame.draw.line(screen, GRAY, (CAM_W+10, py), (WINDOW_W-10, py), 1)
            py += 20
            draw_text(screen, "SPACE — 운동 시작", (cx, py), font_sm, WHITE, center=True)
            py += 28
            draw_text(screen, "R — 다시 캘리브레이션", (cx, py), font_xs,
                      (150, 150, 150), center=True)

        # ── EXERCISE 패널 ─────────────────────────────
        elif mode == "EXERCISE":
            # 횟수
            draw_text(screen, "스쿼트 횟수", (px, py), font_xs, YELLOW)
            py += 24
            draw_text(screen, str(squat_count), (cx, py + 4), font_xl, WHITE, center=True)
            py += 62

            pygame.draw.line(screen, GRAY, (CAM_W+10, py), (WINDOW_W-10, py), 1)
            py += 14

            # 상태
            state_col = GREEN if squat_state == "DOWN" else WHITE
            draw_text(screen, "상태", (px, py), font_xs, YELLOW)
            py += 24
            draw_text(screen, squat_state, (cx, py), font_lg, state_col, center=True)
            py += 48

            pygame.draw.line(screen, GRAY, (CAM_W+10, py), (WINDOW_W-10, py), 1)
            py += 14

            # 무릎 각도 + 게이지
            if metrics:
                ka = metrics["knee_angle"]
                draw_text(screen, "무릎 각도", (px, py), font_xs, YELLOW)
                py += 22
                angle_col = GREEN if ka < calib.ref_knee_down + KNEE_ANGLE_TOLERANCE else RED
                draw_text(screen, f"{ka:.0f}°", (cx, py), font_md, angle_col, center=True)
                py += 34
                ratio = 1.0 - np.clip(
                    (ka - calib.ref_knee_down) / max(calib.ref_knee_up - calib.ref_knee_down, 1),
                    0, 1)
                draw_bar(screen, (px, py, PANEL_W - 36, 16),
                         ratio, GREEN if ratio > 0.6 else YELLOW)
                py += 34

                # 허리 기울기
                draw_text(screen, "허리 기울기", (px, py), font_xs, YELLOW)
                py += 22
                ba       = metrics["back_angle"]
                back_ok  = ba <= calib.ref_back_down + BACK_ANGLE_TOLERANCE
                back_col = GREEN if back_ok else RED
                draw_text(screen, f"{ba:.1f}°", (cx, py), font_md, back_col, center=True)
                py += 38

            pygame.draw.line(screen, GRAY, (CAM_W+10, py), (WINDOW_W-10, py), 1)
            py += 14

            # 피드백 / 오류
            if errors:
                draw_text(screen, "[!] 자세 오류", (cx, py), font_sm, RED, center=True)
                py += 30
                for err in errors:
                    draw_text(screen, err, (cx, py), font_xs, ORANGE, center=True)
                    py += 24
            elif feedback_msg:
                draw_text(screen, feedback_msg, (cx, py), font_sm, feedback_col, center=True)
                py += 30

            # 웹캠 미감지
            if not detected:
                draw_text(screen, "[!] 화면 밖으로 나갔어요",
                          (cx, WINDOW_H - 70), font_xs, RED, center=True)

        # 하단 공통 안내
        pygame.draw.line(screen, GRAY,
                         (CAM_W + 10, WINDOW_H - 38), (WINDOW_W - 10, WINDOW_H - 38), 1)
        draw_text(screen, "Q: 종료   R: 재캘리브레이션",
                  (cx, WINDOW_H - 20), font_xs, (110, 110, 110), center=True)

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pose.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()