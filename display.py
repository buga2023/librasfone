import cv2

# Cores no formato BGR, usado pelo OpenCV.
GREEN = (0, 220, 0)
YELLOW = (0, 230, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (170, 170, 170)
SUBTITLE_WORDS = 10  # a legenda mostra só as últimas palavras

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(frame, text, origin, scale=0.7, color=WHITE, thickness=2) -> None:
    cv2.putText(frame, text, origin, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def put_centered_text(frame, text, y, scale, color, thickness) -> None:
    (width, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    put_text(frame, text, ((frame.shape[1] - width) // 2, y), scale, color, thickness)


def draw_subtitle(frame, words: list[str], color=WHITE, progress: float | None = None) -> None:
    if not words:
        return
    text = " ".join(words[-SUBTITLE_WORDS:]).upper()
    height, width = frame.shape[:2]
    # Diminui a fonte até a frase caber na largura (com limite mínimo).
    scale = 1.0
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    while tw > width - 40 and scale > 0.5:
        scale -= 0.1
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    y = height - 50
    cv2.rectangle(frame, (0, y - th - 18), (width, y + 14), (0, 0, 0), -1)
    cv2.putText(frame, text, ((width - tw) // 2, y), FONT, scale, color, 2, cv2.LINE_AA)
    if progress is not None:
        cv2.rectangle(frame, (0, y + 10), (int(width * min(progress, 1.0)), y + 14), GREEN, -1)
