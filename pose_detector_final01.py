"""
스쿼트 감지기 - 완전판 v2
────────────────────────────────────────────────────────
수정
  - 한글 폰트 PIL 렌더링
  - 이모지 → emojis/ 폴더 PNG 파일로 표시
  - UP→DOWN→UP 완전 사이클만 카운트
  - 얼굴 랜드마크 스켈레톤 제거

제스처
  - 시작화면 → 게임화면 : 두 손 모으기 (양손이 화면 중앙 근처에서 감지)
  - 게임화면 → 시작화면 : 팔 뻗기 (양 손목이 어깨보다 바깥으로 크게 벌어짐)

추가
  - EDM 배경음악 (numpy 합성)
  - 시작화면 + 랜덤 동기부여 글귀
  - 게임 요소: 목표/점수/콤보/포인트/색상단계(10회마다)
  - 기프티콘 5000원 이벤트 (첫 10회 달성 시)
  - 자세 점수 알고리즘: 엉덩이 깊이 + 상체 기울기 + 좌우 균형 종합 평가
  - 엉덩이가 무릎 높이와 같거나 더 낮아진 경우에만 카운트 인정
────────────────────────────────────────────────────────
"""

import cv2
import pygame
import mediapipe as mp
import numpy as np
import sys, os, random, time, math

from PIL import Image, ImageDraw, ImageFont

# ── 경로 설정 ──────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
EMOJI_DIR  = os.path.join(BASE_DIR, "emojis")
RUNNER_IMG_PATH = os.path.join(BASE_DIR, "runner_neon.png")

# ── 창 크기 ────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1280, 720
CAM_W,    CAM_H    = 860,  720
PANEL_W            = WINDOW_W - CAM_W   # 420

# ── 스쿼트 각도 ────────────────────────────────────────
ANGLE_DOWN = 110
ANGLE_UP   = 155

# ── 자세 점수 알고리즘 기준 ─────────────────────────────
# MediaPipe의 정규화 좌표는 화면 아래로 갈수록 y값이 커집니다.
HIP_DEPTH_MARGIN = 0.04       # 자세 점수용: 엉덩이가 무릎보다 이 정도 위에 있어도 허용
COUNT_HIP_DEPTH_MARGIN = 0.00 # 카운트용: 엉덩이가 무릎 높이와 같거나 더 낮을 때만 인정
BALANCE_DIFF_WARN = 18        # 좌우 무릎 각도 차이 주의 기준
BALANCE_DIFF_BAD  = 28        # 좌우 무릎 각도 차이 위험 기준
TORSO_LEAN_WARN   = 35        # 상체 기울기 주의 기준(수직 기준, 도)
TORSO_LEAN_BAD    = 50        # 상체 기울기 위험 기준(수직 기준, 도)
FORM_BONUS_SCORE  = 85        # 이 점수 이상이면 좋은 자세 보너스
FORM_BONUS_POINTS = 15        # 좋은 자세 보너스 점수

# ── 색상 테마 (10회마다 전환) ──────────────────────────
THEME_STAGES = [
    {"name": "스타터",   "primary": (0,   220, 220), "accent": (255, 215,   0), "bg": (20, 20, 30)},
    {"name": "브론즈",   "primary": (205, 127,  50), "accent": (255, 200, 100), "bg": (25, 15, 10)},
    {"name": "실버",     "primary": (192, 192, 192), "accent": (220, 240, 255), "bg": (15, 20, 30)},
    {"name": "골드",     "primary": (255, 215,   0), "accent": (255, 140,   0), "bg": (30, 20,  5)},
    {"name": "플래티넘", "primary": (100, 240, 200), "accent": (200, 255, 250), "bg": (10, 25, 25)},
    {"name": "다이아",   "primary": (120, 180, 255), "accent": (255, 100, 200), "bg": (10, 10, 40)},
    {"name": "레인보우", "primary": (255,  80, 180), "accent": ( 80, 255, 200), "bg": ( 5,  5, 20)},
]

MOTIVATIONAL_QUOTES = [
    "천천히 가도 괜찮다. 멈추지만 않으면 된다. ",
    "아무도 보지 않는 시간에 하는 노력이 진짜 실력이 된다.",
    "결과는 반복의 누적이다.",
    "성공하려면 체력이 먼저다.",
    "오늘 하기 싫은 일을 해야 내일 하고 싶은 일을 할 수 있다.",
    "오늘의 땀이 내일의 자신감을 만든다.",
    "운동은 결국 나를 위한 선택이다.",
    "지금 흘리는 땀 한 방울이 나중의 자신감 한 조각이 된다.",
    "시작하기에 완벽한 순간은 없다. 지금 바로 시작하라.",
    "나 자신을 이기는 것이 가장 큰 승리다.",
]

GOAL_SEQUENCE = [10, 20, 30, 50, 75, 100]

# ── MediaPipe ──────────────────────────────────────────
mp_pose  = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ── 한글 폰트 ──────────────────────────────────────────
def find_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

KOREAN_FONT_PATH = find_korean_font()
_pil_font_cache  = {}

def get_pil_font(size):
    if size not in _pil_font_cache:
        if KOREAN_FONT_PATH:
            _pil_font_cache[size] = ImageFont.truetype(KOREAN_FONT_PATH, size)
        else:
            _pil_font_cache[size] = ImageFont.load_default()
    return _pil_font_cache[size]

def draw_korean_text(surface, text, pos, size, color, center=False):
    font  = get_pil_font(size)
    dummy = Image.new("RGBA", (1, 1))
    dd    = ImageDraw.Draw(dummy)
    bbox  = dd.textbbox((0, 0), text, font=font)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]
    # 충분한 패딩으로 글자 잘림/화질 깨짐 방지
    pad   = max(6, size // 6)
    img   = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(*color, 255))
    pg_img = pygame.image.fromstring(img.tobytes(), img.size, "RGBA")
    x, y = pos
    if center:
        x -= tw // 2
        y -= th // 2
    surface.blit(pg_img, (x - pad, y - pad))
    return tw, th

# ── 이모지 PNG 로더 ────────────────────────────────────
_emoji_cache = {}

def load_emoji(name, size=32):
    key = (name, size)
    if key not in _emoji_cache:
        path = os.path.join(EMOJI_DIR, f"{name}.png")
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.smoothscale(img, (size, size))
            _emoji_cache[key] = img
        else:
            _emoji_cache[key] = None
    return _emoji_cache[key]

def draw_emoji(surface, name, pos, size=32, center=False):
    img = load_emoji(name, size)
    if img is None:
        return
    x, y = pos
    if center:
        x -= size // 2
        y -= size // 2
    surface.blit(img, (x, y))

def draw_emoji_text(surface, emoji_name, text, pos, emoji_size, text_size, color, center=False):
    """이모지 + 텍스트를 한 줄에 표시"""
    # 텍스트 너비 측정
    font = get_pil_font(text_size)
    dummy = Image.new("RGBA", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    gap = 6
    total_w = emoji_size + gap + tw
    x, y = pos
    if center:
        x -= total_w // 2
    draw_emoji(surface, emoji_name, (x, y + (text_size - emoji_size) // 2), emoji_size)
    draw_korean_text(surface, text, (x + emoji_size + gap, y), text_size, color)

_runner_cache = {}

def load_runner_sprite(height=160, facing=1):
    key = (height, facing)

    if key not in _runner_cache:
        if not os.path.exists(RUNNER_IMG_PATH):
            _runner_cache[key] = None
            return None

        img = pygame.image.load(RUNNER_IMG_PATH).convert_alpha()

        # 원본 비율 유지하면서 높이 기준으로 크기 조절
        iw, ih = img.get_size()
        new_w = int(iw * (height / ih))
        img = pygame.transform.smoothscale(img, (new_w, height))

        # 오른쪽 캐릭터는 좌우 반전
        if facing == -1:
            img = pygame.transform.flip(img, True, False)

        _runner_cache[key] = img

    return _runner_cache[key]


def draw_runner_sprite(surface, cx, cy, phase, facing=1, height=160):
    img = load_runner_sprite(height=height, facing=facing)
    if img is None:
        return

    # 살짝 위아래로 뛰는 느낌
    bounce = int(abs(math.sin(phase)) * 8)

    rect = img.get_rect()
    rect.center = (cx, cy - bounce)

    # 네온 그림자 느낌
    glow = pygame.Surface((rect.width + 20, rect.height + 20), pygame.SRCALPHA)
    glow_rect = glow.get_rect(center=(rect.width // 2 + 10, rect.height // 2 + 10))
    pygame.draw.ellipse(glow, (255, 0, 200, 45), glow_rect.inflate(-20, -20))
    surface.blit(glow, (rect.x - 10, rect.y - 10))

    surface.blit(img, rect)

# ── 각도 계산 ──────────────────────────────────────────
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_a   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def get_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))

def get_norm(landmarks, idx):
    lm = landmarks[idx]
    return (lm.x, lm.y)

def midpoint(p1, p2):
    """두 점의 중간 좌표"""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def calc_angle_from_vertical(top, bottom):
    """
    bottom에서 top으로 향하는 벡터가 수직선에서 얼마나 기울었는지 계산합니다.
    값이 클수록 상체가 좌우/앞뒤 방향으로 많이 기울어진 것으로 판단합니다.
    """
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

def evaluate_squat_form(landmarks, angle_l, angle_r):
    """
    무릎 각도만 보던 기존 방식에 다음 3가지를 추가한 자세 점수 알고리즘입니다.
      1) 엉덩이 깊이: 엉덩이가 무릎 높이 근처까지 내려왔는지
      2) 상체 기울기: 어깨-골반 중심선이 수직에서 과도하게 벗어났는지
      3) 좌우 균형: 왼쪽/오른쪽 무릎 각도 차이가 큰지
    반환값: 점수, 피드백, 세부 측정값
    """
    ls = get_norm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs = get_norm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    lh = get_norm(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    rh = get_norm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    lk = get_norm(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    rk = get_norm(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)

    shoulder_mid = midpoint(ls, rs)
    hip_mid      = midpoint(lh, rh)
    knee_mid     = midpoint(lk, rk)

    angle_knee = min(angle_l, angle_r)
    in_squat_zone = angle_knee < ANGLE_UP

    score = 100
    problems = []

    # 1) 엉덩이 깊이: 내려가는 구간에서만 평가합니다.
    # 화면 좌표는 y가 클수록 아래쪽이므로, hip_y가 knee_y에 가까워지면 충분히 앉은 것으로 봅니다.
    hip_depth_gap = knee_mid[1] - hip_mid[1]
    # 카운트용 깊이 기준: 화면 좌표에서 y값이 클수록 아래쪽입니다.
    # hip_mid.y >= knee_mid.y 이면 엉덩이가 무릎과 일자이거나 더 낮은 상태입니다.
    count_depth_ok = hip_mid[1] >= knee_mid[1] - COUNT_HIP_DEPTH_MARGIN
    hip_depth_ok = True
    if in_squat_zone:
        hip_depth_ok = hip_mid[1] >= knee_mid[1] - HIP_DEPTH_MARGIN
        if not hip_depth_ok:
            score -= 25
            problems.append("엉덩이를 더 낮춰주세요")

    # 2) 상체 기울기: 어깨 중심-골반 중심선이 수직선에서 얼마나 벗어났는지 계산합니다.
    torso_lean = calc_angle_from_vertical(shoulder_mid, hip_mid)
    if torso_lean > TORSO_LEAN_BAD:
        score -= 25
        problems.append("상체가 너무 기울었습니다")
    elif torso_lean > TORSO_LEAN_WARN:
        score -= 15
        problems.append("상체를 조금 세워주세요")

    # 3) 좌우 균형: 양쪽 무릎 각도의 차이가 큰지 확인합니다.
    balance_diff = abs(angle_l - angle_r)
    if balance_diff > BALANCE_DIFF_BAD:
        score -= 25
        problems.append("좌우 균형이 많이 흔들립니다")
    elif balance_diff > BALANCE_DIFF_WARN:
        score -= 15
        problems.append("좌우 다리 균형을 맞춰주세요")

    score = int(np.clip(score, 0, 100))

    if not in_squat_zone:
        feedback = "천천히 내려가며 자세를 확인하세요"
    elif problems:
        feedback = problems[0]
    else:
        feedback = "자세 좋습니다!"

    return {
        "score": score,
        "feedback": feedback,
        "hip_depth_ok": hip_depth_ok,
        "count_depth_ok": count_depth_ok,
        "hip_depth_gap": hip_depth_gap,
        "torso_lean": torso_lean,
        "balance_diff": balance_diff,
        "problems": problems,
    }

# ── 스켈레톤 연결 (얼굴 제외) ─────────────────────────
BODY_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW,     mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW,    mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE,      mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP,      mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE,     mp_pose.PoseLandmark.RIGHT_ANKLE),
]
BODY_LANDMARKS = set(idx for pair in BODY_CONNECTIONS for idx in pair)

def draw_body_skeleton(frame, landmarks, w, h, color=(0, 255, 120)):
    for a_idx, b_idx in BODY_CONNECTIONS:
        a = get_px(landmarks, a_idx, w, h)
        b = get_px(landmarks, b_idx, w, h)
        cv2.line(frame, a, b, (255, 255, 255), 2)
    for idx in BODY_LANDMARKS:
        pt = get_px(landmarks, idx, w, h)
        cv2.circle(frame, pt, 5, color, -1)

# ── 파티클 ─────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 7)
        self.x    = x
        self.y    = y
        self.vx   = math.cos(angle) * speed
        self.vy   = math.sin(angle) * speed - 3
        self.life = 1.0
        self.color = color
        self.r    = random.randint(3, 7)

    def update(self):
        self.x   += self.vx
        self.y   += self.vy
        self.vy  += 0.18
        self.life -= 0.035

    def draw(self, surface):
        if self.life <= 0:
            return
        s = pygame.Surface((self.r * 2, self.r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, int(self.life * 255)), (self.r, self.r), self.r)
        surface.blit(s, (int(self.x) - self.r, int(self.y) - self.r))

def draw_pixel_rect(surface, rect, outline, fill=None, width=2, bg=None, notch=10):
    if fill is not None:
        pygame.draw.rect(surface, fill, rect)
    for i in range(width):
        pygame.draw.rect(surface, outline, rect.inflate(-i * 2, -i * 2), 1)

    if bg is None:
        return

    x1, y1, x2, y2 = rect.left, rect.top, rect.right, rect.bottom
    cuts = [
        (x1, y1, notch, width), (x1, y1, width, notch),
        (x2 - notch, y1, notch, width), (x2 - width, y1, width, notch),
        (x1, y2 - width, notch, width), (x1, y2 - notch, width, notch),
        (x2 - notch, y2 - width, notch, width), (x2 - width, y2 - notch, width, notch),
    ]
    for cut in cuts:
        pygame.draw.rect(surface, bg, cut)

def draw_segment_bar(surface, x, y, w, h, total, filled, active_color, empty_color, outline_color):
    gap = 5
    seg_w = max(2, (w - gap * (total - 1)) // total)
    for i in range(total):
        sx = x + i * (seg_w + gap)
        col = active_color if i < filled else empty_color
        pygame.draw.rect(surface, col, (sx, y, seg_w, h))
        pygame.draw.rect(surface, outline_color, (sx, y, seg_w, h), 1)

def draw_scanlines(surface, alpha=34):
    scan = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    for y in range(0, WINDOW_H, 4):
        pygame.draw.line(scan, (0, 0, 0, alpha), (0, y), (WINDOW_W, y))
    surface.blit(scan, (0, 0))

def draw_webcam_hud(surface, detected, feedback, feedback_color):
    sun   = (237, 255, 0)
    cyan  = (0, 229, 229)
    red   = (255, 59, 59)
    muted = (112, 112, 128)
    dark  = (9, 11, 22)

    cam_rect = pygame.Rect(24, 48, CAM_W - 42, WINDOW_H - 72)
    draw_pixel_rect(surface, cam_rect, cyan, width=2, bg=None)

    top_bar = pygame.Rect(40, 62, CAM_W - 74, 40)
    pygame.draw.rect(surface, dark, top_bar)
    pygame.draw.rect(surface, cyan, top_bar, 1)
    draw_korean_text(surface, "LIVE WEBCAM", (top_bar.x + 16, top_bar.y + 11), 18, sun)
    rec_col = red if int(time.time() * 2) % 2 == 0 else (120, 35, 35)
    draw_korean_text(surface, "REC", (top_bar.right - 38, top_bar.y + 11), 16, rec_col)
    pygame.draw.circle(surface, rec_col, (top_bar.right - 14, top_bar.y + 12), 5)

    corners = [
        (48, 122, 1, 1), (CAM_W - 42, 122, -1, 1),
        (48, WINDOW_H - 78, 1, -1), (CAM_W - 42, WINDOW_H - 78, -1, -1),
    ]
    for x, y, sx, sy in corners:
        pygame.draw.line(surface, sun, (x, y), (x + sx * 56, y), 3)
        pygame.draw.line(surface, sun, (x, y), (x, y + sy * 56), 3)

    for y in (250, 420, 570):
        pygame.draw.line(surface, (0, 90, 90), (34, y), (54, y), 1)
        pygame.draw.line(surface, (0, 90, 90), (CAM_W - 48, y), (CAM_W - 28, y), 1)

    strip = pygame.Rect(40, WINDOW_H - 70, CAM_W - 74, 34)
    pygame.draw.rect(surface, dark, strip)
    pygame.draw.rect(surface, (45, 52, 52), strip, 1)
    status_text = feedback if detected else "카메라 앞에 서 주세요"
    draw_korean_text(surface, status_text, (strip.x + 16, strip.y + 9), 16, feedback_color)

# ── EDM 합성 ───────────────────────────────────────────
def make_boing_sound(sample_rate=44100):
    duration = 0.34
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, False)
    sound = np.zeros(n)

    for start_t, base_hz in [(0.00, 420), (0.16, 560)]:
        start = int(start_t * sample_rate)
        length = min(int(0.16 * sample_rate), n - start)
        seg_t = np.linspace(0, 0.16, length, False)
        wobble = np.sin(2 * np.pi * 10 * seg_t) * 70
        sweep = base_hz + wobble + 190 * np.exp(-seg_t * 18)
        phase = 2 * np.pi * np.cumsum(sweep) / sample_rate
        env = np.exp(-seg_t * 9)
        sound[start:start + length] += np.sin(phase) * env

    sound += 0.35 * np.sin(2 * np.pi * 880 * t) * np.exp(-t * 14)
    sound = np.tanh(sound * 1.6)
    sound = (sound / (np.max(np.abs(sound)) + 1e-6) * 16000).astype(np.int16)
    return np.column_stack([sound, sound])

def make_edm_music(sample_rate=44100, duration=8.0):
    n   = int(sample_rate * duration)
    t   = np.linspace(0, duration, n, False)
    bpm = 128
    beat = 60.0 / bpm

    kick = np.zeros(n)
    for i in range(int(duration / beat)):
        start = int(i * beat * sample_rate)
        env_t = np.linspace(0, 0.15, min(int(0.15 * sample_rate), n - start))
        env   = np.exp(-env_t * 30) * np.sin(2 * np.pi * (150 - 130 * env_t / 0.15) * env_t)
        kick[start:start + len(env)] += env * 0.6

    bass_notes = [55, 55, 65, 73, 55, 55, 65, 73]
    bass = np.zeros(n)
    for i, hz in enumerate(bass_notes):
        start = int(i * beat * sample_rate)
        end   = min(int((i + 0.9) * beat * sample_rate), n)
        seg_t = np.linspace(0, beat * 0.9, end - start)
        bass[start:end] += np.sin(2 * np.pi * hz * seg_t) * np.exp(-seg_t * 4) * 0.4

    arp_notes = [220, 277, 330, 415, 440, 415, 330, 277,
                 220, 277, 330, 415, 440, 493, 554, 493]
    arp  = np.zeros(n)
    step = beat / 2
    for i, hz in enumerate(arp_notes):
        start = int(i * step * sample_rate)
        end   = min(int((i + 0.45) * step * sample_rate), n)
        seg_t = np.linspace(0, step * 0.45, end - start)
        wave  = 0.5 * np.sin(2 * np.pi * hz * seg_t) + \
                0.3 * np.sin(2 * np.pi * hz * 2 * seg_t)
        arp[start:end] += wave * np.exp(-seg_t * 6) * 0.25

    mix    = kick + bass + arp
    mx     = np.max(np.abs(mix)) + 1e-6
    mix    = (mix / mx * 28000).astype(np.int16)
    stereo = np.column_stack([mix, mix])
    return stereo

# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════
def main():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Squat Detector")
    clock  = pygame.time.Clock()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        sys.exit(1)

    print("EDM 음악 합성 중...")
    edm_arr   = make_edm_music(duration=8.0)
    edm_sound = pygame.sndarray.make_sound(edm_arr)
    edm_sound.set_volume(0.35)
    edm_sound.play(-1)

    boing_arr = make_boing_sound()
    boing_sound = pygame.sndarray.make_sound(boing_arr)
    boing_sound.set_volume(0.65)

    # ── 게임 상태 ────────────────────────────────────
    screen_mode  = "MANUAL"
    manual_start_t = time.time()
    quote        = random.choice(MOTIVATIONAL_QUOTES)

    squat_count  = 0
    state        = "UP"
    went_down    = False
    angle_knee   = 180.0
    feedback     = "준비하세요"
    feedback_color = (255, 255, 255)

    # 자세 점수 상태
    form_score = 100
    form_score_smooth = 100.0
    form_feedback = "자세 분석 대기"
    form_metrics = {
        "hip_depth_ok": True,
        "count_depth_ok": False,
        "torso_lean": 0.0,
        "balance_diff": 0.0,
    }
    rep_depth_ok = False
    rep_form_score = 100
    last_rep_form_score = 100
    rep_deepest_angle = 180.0
    last_bonus_points = 0

    combo        = 0
    max_combo    = 0
    points       = 0
    theme_idx    = 0
    goal_idx     = 0
    target_goal  = 10
    goal_input_text = "10"
    goal_reached = False
    particles    = []
    last_squat_t = 0.0
    COMBO_TIMEOUT = 3.0

    gifticon_shown = False

    # 제스처 쿨다운 제거됨 — 키보드로만 조작

    def play_transition_sound():
        boing_sound.play()

    def open_goal_input():
        nonlocal screen_mode, goal_input_text

        play_transition_sound()
        screen_mode = "GOAL_INPUT"
        goal_input_text = str(target_goal)

    def start_squat_mode(start_feedback="시작!"):
        nonlocal screen_mode, squat_count, state, went_down
        nonlocal combo, points, theme_idx, goal_idx
        nonlocal goal_reached
        nonlocal gifticon_shown
        nonlocal form_score, form_score_smooth, form_feedback
        nonlocal rep_depth_ok, rep_form_score, last_rep_form_score
        nonlocal rep_deepest_angle, last_bonus_points
        nonlocal feedback, feedback_color

        play_transition_sound()
        screen_mode       = "SQUAT"
        squat_count       = 0
        state             = "UP"
        went_down         = False
        combo             = 0
        points            = 0
        theme_idx         = 0
        goal_idx          = 0
        goal_reached      = False
        gifticon_shown    = False
        form_score        = 100
        form_score_smooth = 100.0
        form_feedback     = "자세 분석 대기"
        rep_depth_ok      = False
        rep_form_score    = 100
        last_rep_form_score = 100
        rep_deepest_angle = 180.0
        last_bonus_points = 0
        feedback          = start_feedback
        feedback_color    = (0, 220, 220)

    running = True
    while running:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and screen_mode == "GOAL_INPUT":
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    target_goal = max(1, min(999, int(goal_input_text or "1")))
                    goal_input_text = str(target_goal)
                    start_squat_mode("시작!")
                elif event.key == pygame.K_BACKSPACE:
                    goal_input_text = goal_input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    play_transition_sound()
                    screen_mode = "START"
                    quote = random.choice(MOTIVATIONAL_QUOTES)
                elif event.unicode.isdigit() and len(goal_input_text) < 3:
                    if goal_input_text == "0":
                        goal_input_text = event.unicode
                    else:
                        goal_input_text += event.unicode
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if screen_mode in ("MANUAL", "START"):
                    open_goal_input()
            
            # START 버튼 마우스 클릭 → 스쿼트 화면으로 이동
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if screen_mode == "MANUAL":
                    mx, my = event.pos
                    _card_rect = pygame.Rect(WINDOW_W // 2 - 360, 56, 720, 608)
                    _btn_rect  = pygame.Rect(_card_rect.x + 74, _card_rect.bottom - 94,
                                             _card_rect.width - 148, 58)
                    if _btn_rect.collidepoint(mx, my):
                        open_goal_input()

        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임 읽기 실패")
            time.sleep(0.1)
            continue
        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_px, w_px = frame.shape[:2]

        # ── Pose 분석 ─────────────────────────────────
        pose_results  = pose.process(frame_rgb)

        detected = pose_results.pose_landmarks is not None

       
        # ── 설명서 5초 후 자동으로 시작화면으로 ──────
        if screen_mode == "MANUAL" and time.time() - manual_start_t >= 5.0:
            play_transition_sound()
            screen_mode = "START"

        theme   = THEME_STAGES[theme_idx % len(THEME_STAGES)]
        PRIMARY = theme["primary"]
        ACCENT  = theme["accent"]
        BG      = theme["bg"]

        # ── 웹캠 Pygame 변환 ──────────────────────────
        cam_surface = cv2.resize(frame, (CAM_W, CAM_H))
        pg_surface  = pygame.surfarray.make_surface(
            cv2.cvtColor(cam_surface, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )

        if screen_mode == "MANUAL":
            ARCADE_BG    = (10, 10, 26)
            ARCADE_PANEL = (20, 20, 38)
            ARCADE_BOX   = (12, 14, 28)
            SUN_GLARE    = (237, 255, 0)
            ARCADE_CYAN  = (0, 229, 229)
            ARCADE_WHITE = (238, 238, 238)
            ARCADE_MUTED = (112, 112, 128)

            screen.fill(ARCADE_BG)

            # 전체 프레임과 메인 카드
            draw_pixel_rect(screen, pygame.Rect(10, 10, WINDOW_W - 20, WINDOW_H - 20),
                            SUN_GLARE, width=3, bg=ARCADE_BG)
            pygame.draw.rect(screen, ARCADE_CYAN, (14, 14, WINDOW_W - 28, WINDOW_H - 28), 1)

            card_rect = pygame.Rect(WINDOW_W // 2 - 360, 56, 720, 608)
            draw_pixel_rect(screen, card_rect, SUN_GLARE, fill=ARCADE_PANEL, width=3, bg=ARCADE_BG)

            # 타이틀
            draw_korean_text(screen, "READY",
                             (WINDOW_W // 2, card_rect.y + 44), 42, ARCADE_CYAN, center=True)
            draw_korean_text(screen, "스쿼트 시작 전 확인",
                             (WINDOW_W // 2, card_rect.y + 92), 28, ARCADE_WHITE, center=True)
            pygame.draw.line(screen, SUN_GLARE,
                             (card_rect.x + 54, card_rect.y + 126),
                             (card_rect.right - 54, card_rect.y + 126), 3)

            steps = [
                ("01", "시작 자세", "화면 중앙에 서서 카메라를 바라보세요"),
                ("02", "준비 자세", "다리는 어깨너비, 복부에 힘을 주세요"),
                ("03", "카운트 기준", "엉덩이가 무릎 높이까지 내려가야 인정됩니다"),
            ]

            step_y = card_rect.y + 158
            for num, label, body in steps:
                step_rect = pygame.Rect(card_rect.x + 54, step_y, card_rect.width - 108, 78)
                draw_pixel_rect(screen, step_rect, ARCADE_CYAN, fill=ARCADE_BOX, width=2, bg=ARCADE_PANEL)

                num_rect = pygame.Rect(step_rect.x + 18, step_rect.y + 15, 78, 48)
                pygame.draw.rect(screen, SUN_GLARE, num_rect)
                pygame.draw.rect(screen, (95, 100, 30), num_rect, 2)
                draw_korean_text(screen, num,
                                 (num_rect.centerx, num_rect.centery), 25, ARCADE_BG, center=True)

                draw_korean_text(screen, label,
                                 (step_rect.x + 118, step_rect.y + 15), 22, SUN_GLARE)
                draw_korean_text(screen, body,
                                 (step_rect.x + 118, step_rect.y + 44), 18, ARCADE_WHITE)
                step_y += 96

            # START 버튼
            btn_rect = pygame.Rect(card_rect.x + 74, card_rect.bottom - 94,
                                   card_rect.width - 148, 58)
            mouse_pos = pygame.mouse.get_pos()
            btn_hovered = btn_rect.collidepoint(mouse_pos)
            blink = int(now * 2) % 2 == 0
            if btn_hovered:
                btn_col = (255, 255, 80)
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            else:
                btn_col = SUN_GLARE if blink else (180, 200, 0)
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            pygame.draw.rect(screen, btn_col, btn_rect)
            pygame.draw.rect(screen, ARCADE_CYAN, btn_rect, 2)
            draw_korean_text(screen, "PRESS SPACE  /  START",
                             (btn_rect.centerx, btn_rect.centery), 24, ARCADE_BG, center=True)

            remain = max(0, 5 - int(time.time() - manual_start_t))
            draw_korean_text(screen, f"{remain}초 후 시작 화면으로 이동   Q KEY EXIT",
                             (WINDOW_W // 2, WINDOW_H - 32), 16, ARCADE_MUTED, center=True)

            draw_scanlines(screen, alpha=24)
            pygame.display.flip()
            clock.tick(30)
            continue

        # ══════════════════════════════════════════════
        # 목표 횟수 입력 화면
        # ══════════════════════════════════════════════
        if screen_mode == "GOAL_INPUT":
            ARCADE_BG    = (10, 10, 26)
            ARCADE_PANEL = (20, 20, 38)
            ARCADE_BOX   = (12, 14, 28)
            SUN_GLARE    = (237, 255, 0)
            ARCADE_CYAN  = (0, 229, 229)
            ARCADE_WHITE = (238, 238, 238)
            ARCADE_MUTED = (112, 112, 128)

            screen.fill(ARCADE_BG)
            draw_pixel_rect(screen, pygame.Rect(10, 10, WINDOW_W - 20, WINDOW_H - 20),
                            SUN_GLARE, width=3, bg=ARCADE_BG)
            pygame.draw.rect(screen, ARCADE_CYAN, (14, 14, WINDOW_W - 28, WINDOW_H - 28), 1)

            card_rect = pygame.Rect(WINDOW_W // 2 - 360, 86, 720, 548)
            draw_pixel_rect(screen, card_rect, SUN_GLARE, fill=ARCADE_PANEL, width=3, bg=ARCADE_BG)

            draw_korean_text(screen, "SET GOAL",
                             (WINDOW_W // 2, card_rect.y + 54), 48, ARCADE_CYAN, center=True)
            draw_korean_text(screen, "목표 스쿼트 횟수를 입력하세요",
                             (WINDOW_W // 2, card_rect.y + 112), 28, ARCADE_WHITE, center=True)
            pygame.draw.line(screen, SUN_GLARE,
                             (card_rect.x + 64, card_rect.y + 150),
                             (card_rect.right - 64, card_rect.y + 150), 3)

            input_rect = pygame.Rect(card_rect.x + 120, card_rect.y + 205, card_rect.width - 240, 130)
            draw_pixel_rect(screen, input_rect, ARCADE_CYAN, fill=ARCADE_BOX, width=2, bg=ARCADE_PANEL)
            shown_goal = goal_input_text if goal_input_text else "0"
            caret = "_" if int(now * 2) % 2 == 0 else " "
            draw_korean_text(screen, f"{shown_goal}{caret}",
                             (input_rect.centerx, input_rect.centery - 8), 72, SUN_GLARE, center=True)
            draw_korean_text(screen, "REPS",
                             (input_rect.right - 70, input_rect.bottom - 34), 20, ARCADE_MUTED, center=True)

            help_rect = pygame.Rect(card_rect.x + 92, card_rect.y + 370, card_rect.width - 184, 92)
            draw_pixel_rect(screen, help_rect, SUN_GLARE, fill=(26, 26, 46), width=2, bg=ARCADE_PANEL)
            draw_korean_text(screen, "ENTER : 시작",
                             (help_rect.x + 34, help_rect.y + 20), 22, SUN_GLARE)
            draw_korean_text(screen, "BACKSPACE : 수정     ESC : 뒤로",
                             (help_rect.x + 34, help_rect.y + 54), 19, ARCADE_WHITE)

            draw_korean_text(screen, "1~999회까지 입력할 수 있습니다",
                             (WINDOW_W // 2, card_rect.bottom - 36), 18, ARCADE_MUTED, center=True)

            draw_scanlines(screen, alpha=24)
            pygame.display.flip()
            clock.tick(30)
            continue

        # ══════════════════════════════════════════════
        # 시작 화면 (레트로 아케이드 스타일)
        # ══════════════════════════════════════════════
        if screen_mode == "START":
            # ── 배경: 딥 네이비 (#0a0a1a) ────────────────
            ARCADE_BG    = (10, 10, 26)
            ARCADE_YLW   = (237, 255, 0)    # #EDFF00  Sun Glare
            ARCADE_RED   = (204, 51, 51)    # #cc3333
            ARCADE_BLUE  = (68, 68, 204)    # #4444cc
            ARCADE_DBLU  = (26, 26, 46)     # #1a1a2e  HUD 박스 배경
            CX = WINDOW_W // 2

            screen.fill(ARCADE_BG)

            # ── 격자 배경 ─────────────────────────────────
            grid_surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            grid_col  = (*ARCADE_YLW, 18)
            grid_size = 48
            for gx_ in range(0, WINDOW_W, grid_size):
                pygame.draw.line(grid_surf, grid_col, (gx_, 0), (gx_, WINDOW_H), 1)
            for gy_ in range(0, WINDOW_H, grid_size):
                pygame.draw.line(grid_surf, grid_col, (0, gy_), (WINDOW_W, gy_), 1)
            dot_col = (*ARCADE_YLW, 38)
            for gx_ in range(0, WINDOW_W, grid_size):
                for gy_ in range(0, WINDOW_H, grid_size):
                    pygame.draw.circle(grid_surf, dot_col, (gx_, gy_), 2)
            screen.blit(grid_surf, (0, 0))

            # ── 외곽 네온 테두리 ─────────────────────────
            border_surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            pygame.draw.rect(border_surf, (*ARCADE_YLW, 180),
                             (0, 0, WINDOW_W, WINDOW_H), 3)
            pygame.draw.rect(border_surf, (*ARCADE_YLW, 60),
                             (5, 5, WINDOW_W - 10, WINDOW_H - 10), 1)
            pygame.draw.rect(border_surf, (*ARCADE_YLW, 25),
                             (9, 9, WINDOW_W - 18, WINDOW_H - 18), 1)
            screen.blit(border_surf, (0, 0))

            # ── 코너 픽셀 장식 ────────────────────────────
            corner_surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            c_positions = [
                (14, 14, 1, 1), (WINDOW_W - 30, 14, -1, 1),
                (14, WINDOW_H - 30, 1, -1), (WINDOW_W - 30, WINDOW_H - 30, -1, -1)
            ]
            for cx_, cy_, dx_, dy_ in c_positions:
                for i in range(5):
                    a = max(0, 220 - i * 44)
                    sz = max(1, 5 - i)
                    pygame.draw.rect(corner_surf, (*ARCADE_YLW, a),
                                     (cx_ + dx_ * i * 5, cy_, sz, sz))
                    pygame.draw.rect(corner_surf, (*ARCADE_YLW, a),
                                     (cx_, cy_ + dy_ * i * 5, sz, sz))
            screen.blit(corner_surf, (0, 0))

            # ── 별 배경 ───────────────────────────────────
            STARS = [
                (int(60  / 680 * WINDOW_W), int(40  / 520 * WINDOW_H), 2, 0.7),
                (int(130 / 680 * WINDOW_W), int(80  / 520 * WINDOW_H), 1, 0.5),
                (int(200 / 680 * WINDOW_W), int(25  / 520 * WINDOW_H), 2, 0.8),
                (int(300 / 680 * WINDOW_W), int(60  / 520 * WINDOW_H), 1, 0.6),
                (int(420 / 680 * WINDOW_W), int(30  / 520 * WINDOW_H), 2, 0.7),
                (int(500 / 680 * WINDOW_W), int(70  / 520 * WINDOW_H), 1, 0.5),
                (int(580 / 680 * WINDOW_W), int(20  / 520 * WINDOW_H), 2, 0.8),
                (int(640 / 680 * WINDOW_W), int(90  / 520 * WINDOW_H), 1, 0.6),
                (int(80  / 680 * WINDOW_W), int(150 / 520 * WINDOW_H), 1, 0.4),
                (int(360 / 680 * WINDOW_W), int(110 / 520 * WINDOW_H), 2, 0.6),
                (int(460 / 680 * WINDOW_W), int(140 / 520 * WINDOW_H), 1, 0.5),
                (int(600 / 680 * WINDOW_W), int(130 / 520 * WINDOW_H), 2, 0.7),
                (int(150 / 680 * WINDOW_W), int(200 / 520 * WINDOW_H), 1, 0.4),
                (int(520 / 680 * WINDOW_W), int(200 / 520 * WINDOW_H), 1, 0.5),
                (int(40  / 680 * WINDOW_W), int(280 / 520 * WINDOW_H), 2, 0.6),
                (int(640 / 680 * WINDOW_W), int(300 / 520 * WINDOW_H), 1, 0.4),
                (int(250 / 680 * WINDOW_W), int(350 / 520 * WINDOW_H), 1, 0.5),
                (int(480 / 680 * WINDOW_W), int(380 / 520 * WINDOW_H), 2, 0.6),
                (int(100 / 680 * WINDOW_W), int(420 / 520 * WINDOW_H), 1, 0.4),
                (int(580 / 680 * WINDOW_W), int(440 / 520 * WINDOW_H), 1, 0.5),
            ]
            star_surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            for sx, sy, sr, sa in STARS:
                pygame.draw.circle(star_surf, (255, 255, 255, int(sa * 255)), (sx, sy), sr)
            screen.blit(star_surf, (0, 0))

            # ── 네온 픽셀 러닝 캐릭터 ─────────────────────
            run_phase = now * 8.0
            runner_y = int(210 / 520 * WINDOW_H)

            draw_runner_sprite(
                screen,
                int(95 / 680 * WINDOW_W),
                runner_y,
                run_phase,
                facing=1,
                height=170
            )

            draw_runner_sprite(
                screen,
                int(585 / 680 * WINDOW_W),
                runner_y,
                run_phase + math.pi,
                facing=-1,
                height=170
            )

            # ── 상단 HUD 바 ──────────────────────────────
            hud_y = int(18 / 520 * WINDOW_H)
            hud_h = int(22 / 520 * WINDOW_H)

            # 라이프 박스 (왼쪽)
            life_rect = pygame.Rect(int(30/680*WINDOW_W), hud_y, int(44/680*WINDOW_W), hud_h)
            pygame.draw.rect(screen, ARCADE_DBLU, life_rect, border_radius=3)
            pygame.draw.rect(screen, ARCADE_YLW,  life_rect, 1, border_radius=3)
            draw_korean_text(screen, "★ 3",
                             (life_rect.centerx, life_rect.centery), 11, ARCADE_YLW, center=True)

            # 점수 바 (라이프 오른쪽)
            score_outer = pygame.Rect(int(84/680*WINDOW_W), hud_y, int(120/680*WINDOW_W), hud_h)
            pygame.draw.rect(screen, ARCADE_DBLU, score_outer, border_radius=3)
            pygame.draw.rect(screen, ARCADE_YLW,  score_outer, 1, border_radius=3)
            score_fill  = pygame.Rect(score_outer.x+2, score_outer.y+2,
                                      int(80/680*WINDOW_W), score_outer.height-4)
            pygame.draw.rect(screen, ARCADE_YLW, score_fill, border_radius=2)
            draw_korean_text(screen, str(points).zfill(10),
                             (score_outer.x + int(score_outer.width * 0.65),
                              score_outer.centery), 9, ARCADE_BG, center=True)

            # 하트 박스 (오른쪽)
            heart_rect = pygame.Rect(int(560/680*WINDOW_W), hud_y, int(50/680*WINDOW_W), hud_h)
            pygame.draw.rect(screen, ARCADE_DBLU, heart_rect, border_radius=3)
            pygame.draw.rect(screen, ARCADE_RED,  heart_rect, 1, border_radius=3)
            draw_korean_text(screen, "♥ x3",
                             (heart_rect.centerx, heart_rect.centery), 11, ARCADE_RED, center=True)

            # ── 타이틀 "SQUAT" ───────────────────────────
            title_y = int(105 / 520 * WINDOW_H)
            # 그림자 (검정 외곽선)
            draw_korean_text(screen, "START",
                             (CX+4, title_y+4), 52, (10, 10, 26), center=True)
            draw_korean_text(screen, "START",
                             (CX-4, title_y-4), 52, (10, 10, 26), center=True)
            draw_korean_text(screen, "START",
                             (CX+4, title_y-4), 52, (10, 10, 26), center=True)
            draw_korean_text(screen, "START",
                             (CX-4, title_y+4), 52, (10, 10, 26), center=True)
            # 메인 타이틀 (Sun Glare 노랑)
            draw_korean_text(screen, "SQUAT", (CX, title_y), 52, ARCADE_YLW, center=True)

            # ── START 박스 ───────────────────────────────
            box_x  = int(185 / 680 * WINDOW_W)
            box_y  = int(155 / 520 * WINDOW_H)
            box_w  = int(310 / 680 * WINDOW_W)
            box_h  = int(62  / 520 * WINDOW_H)
            outer_box = pygame.Rect(box_x, box_y, box_w, box_h)
            pygame.draw.rect(screen, ARCADE_DBLU, outer_box, border_radius=4)
            pygame.draw.rect(screen, ARCADE_YLW,  outer_box, 2, border_radius=4)

            inner_box = pygame.Rect(box_x+5, box_y+5, box_w-10, box_h-10)
            inner_surf = pygame.Surface((inner_box.width, inner_box.height), pygame.SRCALPHA)
            inner_surf.fill((15, 26, 15, 153))
            pygame.draw.rect(inner_surf, (*ARCADE_YLW, 60),
                             (0, 0, inner_box.width, inner_box.height), 1, border_radius=2)
            screen.blit(inner_surf, (inner_box.x, inner_box.y))

            # START 박스 안 텍스트 (깜박)
            blink = int(now * 2) % 2 == 0
            start_text_color = ARCADE_YLW if blink else (180, 200, 0)
            draw_korean_text(screen, "START",
                             (CX, box_y + box_h // 2), 28, start_text_color, center=True)

            # 점선 테두리 (깜박 효과)
            if blink:
                dash_surf = pygame.Surface((box_w + 8, box_h + 8), pygame.SRCALPHA)
                dash_rect = pygame.Rect(0, 0, box_w+8, box_h+8)
                # 점선 테두리 시뮬레이션
                for i in range(0, box_w+8, 8):
                    if (i // 8) % 2 == 0:
                        pygame.draw.rect(dash_surf, (255, 255, 255, 102),
                                         (i, 0, 4, 1))
                        pygame.draw.rect(dash_surf, (255, 255, 255, 102),
                                         (i, box_h+7, 4, 1))
                for i in range(0, box_h+8, 8):
                    if (i // 8) % 2 == 0:
                        pygame.draw.rect(dash_surf, (255, 255, 255, 102),
                                         (0, i, 1, 4))
                        pygame.draw.rect(dash_surf, (255, 255, 255, 102),
                                         (box_w+7, i, 1, 4))
                screen.blit(dash_surf, (box_x-4, box_y-4))

            # ── 안내 텍스트 ──────────────────────────────
            guide_y = int(280 / 520 * WINDOW_H)
            draw_korean_text(screen, "▶ 스페이스바를 누르면 목표 횟수를 입력합니다.",
                             (CX, guide_y), 18, ARCADE_YLW, center=True)

            # ── 구분선 ───────────────────────────────────
            sep_y = int(300 / 520 * WINDOW_H)
            sep_surf = pygame.Surface((WINDOW_W, 1), pygame.SRCALPHA)
            pygame.draw.line(sep_surf, (*ARCADE_YLW, 102),
                             (int(180/680*WINDOW_W), 0), (int(500/680*WINDOW_W), 0), 1)
            screen.blit(sep_surf, (0, sep_y))

            # ── 동기부여 글귀 ─────────────────────────────
            quote_y = int(335 / 520 * WINDOW_H)
            q_surf  = pygame.Surface((WINDOW_W, 30), pygame.SRCALPHA)
            screen.blit(q_surf, (0, quote_y))
            draw_korean_text(screen, quote,
                             (CX, quote_y), 16, (180, 180, 180), center=True)

            # ── UFO 장식 (좌우) ──────────────────────────
            def draw_ufo(surf, cx_, cy_):
                # 본체
                ufo_s = pygame.Surface((40, 20), pygame.SRCALPHA)
                pygame.draw.ellipse(ufo_s, (51, 51, 170), (0, 5, 40, 18))
                pygame.draw.ellipse(ufo_s, (85, 85, 204), (10, 0, 20, 14))
                surf.blit(ufo_s, (cx_ - 20, cy_ - 9))
                # 발광점 3개
                for dx in [-6, 0, 6]:
                    pygame.draw.circle(surf, ARCADE_YLW, (cx_ + dx, cy_ + 7), 3)

            ufo_y = int(330 / 520 * WINDOW_H)
            draw_ufo(screen, int(180 / 680 * WINDOW_W), ufo_y)
            draw_ufo(screen, int(500 / 680 * WINDOW_W), ufo_y)

            # ── 하단 픽셀 장식 ───────────────────────────
            pix_y = int(380 / 520 * WINDOW_H)
            pix_sizes = [(0.8, 140), (0.5, 156), (0.3, 172)]
            for alpha, px_norm in pix_sizes:
                px = int(px_norm / 680 * WINDOW_W)
                ps = pygame.Surface((12, 12), pygame.SRCALPHA)
                ps.fill((*ARCADE_YLW, int(alpha * 255)))
                screen.blit(ps, (px, pix_y))
            for alpha, px_norm in [(0.3, 516), (0.5, 532), (0.8, 548)]:
                px = int(px_norm / 680 * WINDOW_W)
                ps = pygame.Surface((12, 12), pygame.SRCALPHA)
                ps.fill((*ARCADE_YLW, int(alpha * 255)))
                screen.blit(ps, (px, pix_y))

            # ── Q KEY : EXIT ─────────────────────────────
            draw_korean_text(screen, "Q KEY : EXIT",
                             (CX, int(480 / 520 * WINDOW_H)), 11, (85, 85, 102), center=True)

            draw_scanlines(screen, alpha=18)

            pygame.display.flip()
            clock.tick(30)
            continue

        # ══════════════════════════════════════════════
        # 스쿼트 화면
        # ══════════════════════════════════════════════
        if detected:
            landmarks = pose_results.pose_landmarks.landmark

            hip   = get_px(landmarks, mp_pose.PoseLandmark.RIGHT_HIP,   w_px, h_px)
            knee  = get_px(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE,  w_px, h_px)
            ankle = get_px(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, w_px, h_px)
            l_hip   = get_px(landmarks, mp_pose.PoseLandmark.LEFT_HIP,   w_px, h_px)
            l_knee  = get_px(landmarks, mp_pose.PoseLandmark.LEFT_KNEE,  w_px, h_px)
            l_ankle = get_px(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, w_px, h_px)

            angle_r    = calc_angle(hip,   knee,  ankle)
            angle_l    = calc_angle(l_hip, l_knee, l_ankle)
            angle_knee = min(angle_r, angle_l)

            # ── 자세 점수 알고리즘 평가 ─────────────────
            form_metrics = evaluate_squat_form(landmarks, angle_l, angle_r)
            form_score = form_metrics["score"]
            form_score_smooth = form_score_smooth * 0.80 + form_score * 0.20
            form_feedback = form_metrics["feedback"]

            # 한 번의 스쿼트 중 한 번이라도 엉덩이가 무릎 높이와 같거나 더 낮아졌는지 기록합니다.
            # 이 값이 True가 되어야 올라올 때 스쿼트 1회로 카운트합니다.
            if angle_knee < ANGLE_UP and form_metrics.get("count_depth_ok", False):
                rep_depth_ok = True

            # 한 번의 스쿼트 중 가장 깊게 내려간 지점의 자세 점수를 대표 점수로 저장
            if angle_knee < ANGLE_UP and angle_knee < rep_deepest_angle:
                rep_deepest_angle = angle_knee
                rep_form_score = form_score

            # 상태 판정
            if state == "UP" and angle_knee < ANGLE_DOWN:
                state      = "DOWN"
                went_down  = True
                feedback   = form_feedback if form_score < FORM_BONUS_SCORE else "좋아요! 올라오세요"
                feedback_color = (50, 220, 50) if form_score >= FORM_BONUS_SCORE else (255, 215, 0)

            elif state == "DOWN" and angle_knee > ANGLE_UP:
                if went_down and rep_depth_ok:
                    squat_count += 1
                    if now - last_squat_t < COMBO_TIMEOUT:
                        combo += 1
                    else:
                        combo = 1
                    last_squat_t = now
                    max_combo    = max(max_combo, combo)

                    last_rep_form_score = int(rep_form_score)
                    last_bonus_points = FORM_BONUS_POINTS if last_rep_form_score >= FORM_BONUS_SCORE else 0
                    points      += 10 * combo + last_bonus_points

                    # 파티클
                    for _ in range(20 + combo * 3):
                        particles.append(Particle(
                            random.randint(100, CAM_W - 100),
                            random.randint(200, 500),
                            PRIMARY
                        ))

                    # 테마 단계
                    if squat_count % 10 == 0:
                        theme_idx += 1

                    if squat_count >= target_goal and not goal_reached:
                        goal_reached = True
                        feedback = f"목표 {target_goal}회 달성!"
                        feedback_color = (0, 220, 220)
                    elif last_bonus_points > 0:
                        feedback = f"{squat_count}회 완료! 자세 {last_rep_form_score}점 +{last_bonus_points}"
                        feedback_color = (0, 220, 220)
                    else:
                        feedback = f"{squat_count}회 완료! 자세 {last_rep_form_score}점"
                        feedback_color = (255, 215, 0)

                elif went_down and not rep_depth_ok:
                    # 무릎 각도는 내려갔지만 엉덩이가 무릎 높이까지 내려가지 않은 경우
                    # 스쿼트 1회로 인정하지 않습니다.
                    combo = 0
                    last_bonus_points = 0
                    feedback = "깊이가 부족해서 카운트 안 됨"
                    feedback_color = (220, 50, 50)
                    form_feedback = "엉덩이를 무릎 높이까지 낮춰주세요"

                state     = "UP"
                went_down = False
                rep_depth_ok = False
                rep_form_score = 100
                rep_deepest_angle = 180.0

            elif state == "UP" and angle_knee < ANGLE_UP:
                feedback       = form_feedback if form_score < FORM_BONUS_SCORE else "더 내려가세요"
                feedback_color = (255, 215, 0)

            if now - last_squat_t > COMBO_TIMEOUT and combo > 0:
                combo = 0

            # 스켈레톤 (얼굴 제외)
            draw_body_skeleton(frame, landmarks, w_px, h_px, PRIMARY)

            cv2.putText(frame, f"{int(angle_knee)}",
                        (knee[0] + 10, knee[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
            cv2.line(frame, hip,   knee,  (0, 200, 255), 3)
            cv2.line(frame, knee,  ankle, (0, 200, 255), 3)
            cv2.circle(frame, knee,  10, (255, 100,  0), -1)
            cv2.circle(frame, hip,    7, (  0, 200, 255), -1)
            cv2.circle(frame, ankle,  7, (  0, 200, 255), -1)

        # ── 캠 → Pygame (스쿼트 화면용 재생성) ──────
        cam_surface = cv2.resize(frame, (CAM_W, CAM_H))
        pg_surface  = pygame.surfarray.make_surface(
            cv2.cvtColor(cam_surface, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )

        ARCADE_BG    = (10, 10, 26)
        ARCADE_PANEL = (20, 20, 38)
        ARCADE_BOX   = (12, 14, 28)
        ARCADE_DBOX  = (26, 26, 46)
        SUN_GLARE    = (237, 255, 0)
        ARCADE_CYAN  = (0, 229, 229)
        ARCADE_RED   = (255, 59, 59)
        ARCADE_GREEN = (50, 230, 50)
        ARCADE_WHITE = (238, 238, 238)
        ARCADE_MUTED = (112, 112, 128)

        screen.fill(ARCADE_BG)
        screen.blit(pg_surface, (0, 0))

        # 파티클
        particles = [p for p in particles if p.life > 0]
        for p in particles:
            p.update()
            p.draw(screen)

        draw_webcam_hud(screen, detected, feedback, feedback_color)

        # 오른쪽 레트로 HUD 패널
        panel_rect = pygame.Rect(CAM_W + 8, 48, PANEL_W - 20, WINDOW_H - 72)
        draw_pixel_rect(screen, panel_rect, SUN_GLARE, fill=ARCADE_PANEL, width=3, bg=ARCADE_BG)

        cx_panel = panel_rect.centerx
        draw_korean_text(screen, "SQUAT",
                         (cx_panel, panel_rect.y + 30), 46, ARCADE_CYAN, center=True)
        draw_korean_text(screen, "ARCADE",
                         (cx_panel, panel_rect.y + 78), 28, ARCADE_WHITE, center=True)
        pygame.draw.line(screen, SUN_GLARE,
                         (panel_rect.x + 28, panel_rect.y + 102),
                         (panel_rect.right - 28, panel_rect.y + 102), 3)

        # COUNT 박스
        count_rect = pygame.Rect(panel_rect.x + 28, panel_rect.y + 122, panel_rect.width - 56, 120)
        draw_pixel_rect(screen, count_rect, ARCADE_CYAN, fill=ARCADE_BOX, width=2, bg=ARCADE_PANEL)
        draw_korean_text(screen, "COUNT", (count_rect.x + 18, count_rect.y + 18), 18, SUN_GLARE)
        draw_korean_text(screen, f"LV {goal_idx + 1:02d}",
                         (count_rect.right - 62, count_rect.y + 18), 16, ARCADE_MUTED)
        draw_korean_text(screen, str(squat_count).zfill(2),
                         (count_rect.centerx, count_rect.y + 55), 62, ARCADE_WHITE, center=True)

        # 목표 진행
        current_goal = target_goal
        bar_ratio    = np.clip(squat_count / max(current_goal, 1), 0, 1)
        mission_y = count_rect.bottom + 30
        draw_korean_text(screen, f"MISSION {current_goal} REPS",
                         (panel_rect.x + 48, mission_y), 17, SUN_GLARE)
        seg_x = panel_rect.x + 48
        seg_y = mission_y + 30
        seg_w = panel_rect.width - 96
        draw_segment_bar(screen, seg_x, seg_y, seg_w, 24, 10, int(math.ceil(bar_ratio * 10)),
                         SUN_GLARE, (52, 55, 34), (95, 100, 30))
        draw_korean_text(screen, f"{squat_count}/{current_goal}",
                         (cx_panel, seg_y + 34), 17, ARCADE_WHITE, center=True)

        # SCORE / POSE 박스
        box_y = seg_y + 62
        score_rect = pygame.Rect(panel_rect.x + 28, box_y, 154, 98)
        pose_rect = pygame.Rect(panel_rect.x + 206, box_y, panel_rect.right - panel_rect.x - 234, 98)
        draw_pixel_rect(screen, score_rect, SUN_GLARE, fill=ARCADE_DBOX, width=2, bg=ARCADE_PANEL)
        draw_pixel_rect(screen, pose_rect, SUN_GLARE, fill=ARCADE_DBOX, width=2, bg=ARCADE_PANEL)

        draw_korean_text(screen, "SCORE", (score_rect.x + 20, score_rect.y + 18), 16, SUN_GLARE)
        draw_korean_text(screen, str(points).zfill(4),
                         (score_rect.centerx, score_rect.y + 56), 27, ARCADE_WHITE, center=True)
        if combo > 1:
            blink_col = SUN_GLARE if int(now * 4) % 2 == 0 else ARCADE_CYAN
            draw_korean_text(screen, f"x{combo}", (score_rect.right - 24, score_rect.bottom - 18),
                             15, blink_col, center=True)

        draw_korean_text(screen, "POSE", (pose_rect.x + 20, pose_rect.y + 18), 16, SUN_GLARE)
        state_color = ARCADE_GREEN if state == "DOWN" else ARCADE_WHITE
        angle_color = ARCADE_GREEN if angle_knee < ANGLE_DOWN else ARCADE_RED
        draw_korean_text(screen, state,
                         (pose_rect.x + 65, pose_rect.y + 58), 34, state_color, center=True)
        draw_korean_text(screen, f"{int(angle_knee)}°",
                         (pose_rect.right - 42, pose_rect.y + 58), 27, angle_color, center=True)

        # 자세 점수 박스
        form_display_score = int(form_score_smooth)
        if form_display_score >= 85:
            form_color = ARCADE_GREEN
        elif form_display_score >= 70:
            form_color = SUN_GLARE
        else:
            form_color = ARCADE_RED

        form_rect = pygame.Rect(panel_rect.x + 28, box_y + 122, panel_rect.width - 56, 98)
        draw_pixel_rect(screen, form_rect, form_color, fill=(10, 28, 20), width=2, bg=ARCADE_PANEL)
        draw_korean_text(screen, "FORM SCORE", (form_rect.x + 20, form_rect.y + 18), 17, form_color)
        draw_korean_text(screen, str(form_display_score),
                         (form_rect.right - 44, form_rect.y + 18), 33, form_color, center=True)
        draw_segment_bar(screen, form_rect.x + 20, form_rect.y + 60, form_rect.width - 40, 18,
                         20, int(form_display_score / 5), form_color, (24, 45, 28), (40, 90, 50))

        # 피드백 / 깊이 상태
        depth_ok_now = form_metrics.get("count_depth_ok", False)
        depth_text = "[OK] DEPTH CLEAR" if depth_ok_now else "[!] DEPTH LOW"
        depth_color = ARCADE_GREEN if depth_ok_now else ARCADE_RED
        alert_y = form_rect.bottom + 18
        draw_korean_text(screen, depth_text, (cx_panel, alert_y), 21, depth_color, center=True)
        draw_korean_text(screen, form_feedback, (cx_panel, alert_y + 24), 15, form_color, center=True)

        if goal_reached and int(now * 2) % 2 == 0:
            draw_korean_text(screen, "MISSION CLEAR!",
                             (cx_panel, alert_y + 44), 16, (255, 130, 50), center=True)
        elif not detected:
            draw_korean_text(screen, "CAMERA CHECK",
                             (cx_panel, alert_y + 44), 16, ARCADE_RED, center=True)
        else:
            draw_korean_text(screen, "SPACE START   Q EXIT",
                             (cx_panel, alert_y + 44), 14, ARCADE_MUTED, center=True)

        draw_scanlines(screen, alpha=30)

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pose.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()