import shutil
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from camera_io import video_timestamp
from display import GREEN, put_text

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

CAMERA_INDEX = 0
MAX_HANDS = 2
WINDOW_NAME = "LibrasFone - Landmarks (Q para sair)"


def ensure_model() -> Path:
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size:
        return MODEL_PATH
    print(f"Baixando o modelo para {MODEL_PATH} ...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Baixa em um arquivo temporário e só renomeia no fim: se o download for
    # interrompido, não fica um modelo pela metade no lugar do verdadeiro.
    partial = MODEL_PATH.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not partial.stat().st_size:
            raise OSError("O download do modelo do MediaPipe veio vazio")
        partial.replace(MODEL_PATH)
    finally:
        partial.unlink(missing_ok=True)
    return MODEL_PATH


def create_landmarker(num_hands: int = MAX_HANDS) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=ensure_model().read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker: vision.HandLandmarker, frame, timestamp_ms: int) -> vision.HandLandmarkerResult:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)


def draw_hands(frame, result: vision.HandLandmarkerResult) -> None:
    for hand_landmarks in result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )


def main() -> int:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        print("Erro: não foi possível abrir a webcam.", file=sys.stderr)
        return 1

    start = time.monotonic()
    last_timestamp_ms = -1

    try:
        with create_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Erro: falha ao ler um frame da webcam.", file=sys.stderr)
                    return 1

                timestamp_ms = video_timestamp(time.monotonic() - start, last_timestamp_ms)
                last_timestamp_ms = timestamp_ms
                result = detect_hands(landmarker, frame, timestamp_ms)

                draw_hands(frame, result)
                # Espelha só na hora de mostrar, para a imagem parecer um espelho.
                frame = cv2.flip(frame, 1)
                put_text(frame, f"Mãos detectadas: {len(result.hand_landmarks)}", (10, 30), 0.8, GREEN)

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
