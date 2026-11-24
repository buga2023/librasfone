import argparse
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from camera_io import video_timestamp
from display import GREEN, RED, WHITE, YELLOW, put_centered_text, put_text
from landmark_viewer import CAMERA_INDEX, create_landmarker, detect_hands, draw_hands
from landmarks import HAND_SLOT_NAMES, HAND_SLOTS, SAMPLE_SHAPE, SEQUENCE_LENGTH, extract_hands

# Identifica quem gravou cada amostra. Pode ser definido aqui ou com --coletor.
# Ele entra no nome do arquivo e é usado pelo train.py como "fonte" para testar
# o modelo com pessoas que ficaram fora do treino.
COLLECTOR_ID = ""
COLLECTOR_ID_PATTERN = re.compile(r"[a-z0-9]+")

SIGNS = [
    "como", "voce", "eu", "nome", "qual", "onde", "quando", "quem", "o_que",
    "quero", "preciso", "pode", "nao_pode", "esperar", "repetir", "falar",
    "ouvir", "entender", "nao_entendi", "pergunta", "resposta", "certo",
    "errado", "sim", "nao", "agora", "depois", "por_favor", "obrigado",
]

COUNTDOWN_SECONDS = 3
MIN_FRAMES_WITH_HAND = 10  # amostras com mãos em menos frames que isso são descartadas
MESSAGE_SECONDS = 2.5

DATASET_DIR = Path(__file__).parent / "dataset"

WINDOW_NAME = "LibrasFone - Coleta de dados (Q para sair)"
KEY_SPACE = 32
KEY_LEFT = 0x250000
KEY_RIGHT = 0x270000


def collector_id_problem() -> str | None:
    if not COLLECTOR_ID:
        return "COLLECTOR_ID vazio"
    if not COLLECTOR_ID_PATTERN.fullmatch(COLLECTOR_ID):
        return f"COLLECTOR_ID inválido: {COLLECTOR_ID!r}"
    return None


def report_collector_id_problem(problem: str) -> None:
    print(
        f"\nAVISO: {problem}.\n"
        'Rode com --coletor seunome ou defina COLLECTOR_ID no topo de collect_data.py, ex.: COLLECTOR_ID = "etse"\n'
        "(apenas letras minúsculas e números, sem espaços nem acentos).\n"
        "A gravação fica BLOQUEADA até isso ser feito.\n",
        file=sys.stderr,
    )


def sample_files(sign: str) -> list[Path]:
    return sorted((DATASET_DIR / sign).glob(f"{sign}_*.npy"))


def own_sample_files(sign: str) -> list[Path]:
    return sorted((DATASET_DIR / sign).glob(f"{sign}_{COLLECTOR_ID}_*.npy"))


def find_incompatible_samples() -> list[tuple[Path, str]]:
    incompatible = []
    for path in sorted(DATASET_DIR.rglob("*.npy")):
        try:
            # mmap_mode="r" lê só o cabeçalho, sem carregar o arquivo inteiro.
            shape = np.load(path, mmap_mode="r").shape
        except Exception:
            incompatible.append((path, "ilegível"))
            continue
        if shape != SAMPLE_SHAPE:
            incompatible.append((path, f"formato {shape}"))
    return incompatible


def report_incompatible_samples(incompatible: list[tuple[Path, str]]) -> None:
    print(
        f"\nAVISO: {len(incompatible)} amostra(s) em formato incompatível com o atual {SAMPLE_SHAPE}.\n"
        "O formato mudou de 63 valores por frame (uma mão) para 126 (duas mãos);\n"
        "amostras antigas não podem ser misturadas com as novas.",
        file=sys.stderr,
    )
    for path, reason in incompatible[:10]:
        print(f"  {path} ({reason})", file=sys.stderr)
    if len(incompatible) > 10:
        print(f"  ... e mais {len(incompatible) - 10}", file=sys.stderr)
    print(
        "A gravação fica BLOQUEADA até você mover ou apagar essas amostras, por exemplo\n"
        "renomeando dataset/ para dataset_antigo/.\n",
        file=sys.stderr,
    )


def next_sample_index(sign: str) -> int:
    pattern = re.compile(rf"{re.escape(sign)}_{re.escape(COLLECTOR_ID)}_(\d+)\.npy")
    used = [0]
    for path in (DATASET_DIR / sign).glob(f"{sign}_{COLLECTOR_ID}_*.npy"):
        match = pattern.fullmatch(path.name)
        if match:
            used.append(int(match.group(1)))
    return max(used) + 1


def save_sample(sign: str, sample: np.ndarray) -> Path:
    problem = collector_id_problem()
    if problem:
        raise ValueError(problem)
    path = DATASET_DIR / sign / f"{sign}_{COLLECTOR_ID}_{next_sample_index(sign):04d}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    # O modo "xb" falha se o arquivo já existir, então nunca sobrescreve uma amostra.
    with open(path, "xb") as f:
        np.save(f, np.asarray(sample, dtype=np.float32))
    return path


@dataclass
class RecordingSession:

    state: str = "idle"
    countdown_end: float = 0.0
    frames: list = field(default_factory=list)
    frames_with_hand: int = 0

    def start(self, now: float) -> None:
        self.state = "countdown"
        self.countdown_end = now + COUNTDOWN_SECONDS

    def update(self, hands: np.ndarray, has_hand: bool, now: float) -> np.ndarray | None:
        if self.state == "countdown" and now >= self.countdown_end:
            self.state = "recording"
            self.frames, self.frames_with_hand = [], 0
        if self.state != "recording":
            return None
        self.frames.append(hands.reshape(-1).copy())
        self.frames_with_hand += bool(has_hand)
        if len(self.frames) == SEQUENCE_LENGTH:
            self.state = "idle"
            return np.stack(self.frames)
        return None


def main(argv=None) -> int:
    global COLLECTOR_ID
    parser = argparse.ArgumentParser(description="Coleta de amostras do LibrasFone pela webcam.")
    parser.add_argument("--coletor", help="nome do coletor; alternativa a editar COLLECTOR_ID")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    args = parser.parse_args(argv)
    if args.coletor is not None:
        if not COLLECTOR_ID_PATTERN.fullmatch(args.coletor):
            parser.error("--coletor deve conter apenas letras minúsculas e números")
        COLLECTOR_ID = args.coletor
    if args.camera < 0:
        parser.error("--camera deve ser um índice não negativo")
    # Aqui a câmera é lida diretamente, sem o CameraReader: a amostra precisa
    # de SEQUENCE_LENGTH frames consecutivos, sem pular nenhum.
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        print("Erro: não foi possível abrir a webcam.", file=sys.stderr)
        return 1

    incompatible = find_incompatible_samples()
    if incompatible:
        report_incompatible_samples(incompatible)

    collector_problem = collector_id_problem()
    if collector_problem:
        report_collector_id_problem(collector_problem)

    counts = {sign: 0 if collector_problem else len(own_sample_files(sign)) for sign in SIGNS}
    total_counts = {sign: len(sample_files(sign)) for sign in SIGNS}
    sign_index = 0

    recording = RecordingSession()

    message, message_color, message_until = "", GREEN, 0.0

    start = time.monotonic()
    last_timestamp_ms = -1

    try:
        with create_landmarker(num_hands=len(HAND_SLOTS)) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Erro: falha ao ler um frame da webcam.", file=sys.stderr)
                    return 1

                now = time.monotonic()
                timestamp_ms = video_timestamp(now - start, last_timestamp_ms)
                last_timestamp_ms = timestamp_ms

                result = detect_hands(landmarker, frame, timestamp_ms)
                hands, hands_present = extract_hands(result, frame.shape[1] / frame.shape[0])
                has_hand = any(hands_present)

                sign = SIGNS[sign_index]

                sample = recording.update(hands, has_hand, now)
                if sample is not None:
                    message_until = now + MESSAGE_SECONDS
                    if recording.frames_with_hand < MIN_FRAMES_WITH_HAND:
                        message = f"DESCARTADA: mãos em só {recording.frames_with_hand}/{SEQUENCE_LENGTH} frames"
                        message_color = RED
                    else:
                        path = save_sample(sign, sample)
                        counts[sign] += 1
                        total_counts[sign] += 1
                        message = f"Salva: {path.name} (mãos em {recording.frames_with_hand}/{SEQUENCE_LENGTH} frames)"
                        message_color = GREEN
                        print(f"Amostra salva em {path}")

                # ---- desenho da tela
                draw_hands(frame, result)
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                put_text(frame, f"Sinal [{sign_index + 1}/{len(SIGNS)}]: {sign}", (10, 30), 0.9, YELLOW)
                put_text(
                    frame,
                    f"Amostras gravadas: {counts[sign]} suas / {total_counts[sign]} no total",
                    (10, 62),
                )
                if has_hand:
                    sides = "+".join(name for name, here in zip(HAND_SLOT_NAMES, hands_present) if here)
                    put_text(frame, f"Mãos detectadas: {sum(hands_present)} ({sides})", (10, 92), 0.6, GREEN)
                else:
                    put_text(frame, "Mãos detectadas: 0", (10, 92), 0.6, RED)

                if incompatible:
                    put_text(frame, f"FORMATO ANTIGO: {len(incompatible)} amostra(s) incompatíveis", (10, 126), 0.6, RED)
                    put_text(frame, "Gravação bloqueada: mova ou apague as amostras antigas de dataset/", (10, 150), 0.5, RED)
                    put_text(frame, "(lista no terminal) e abra o programa novamente", (10, 172), 0.5, RED)
                elif collector_problem:
                    put_text(frame, f"{collector_problem}: gravação bloqueada", (10, 126), 0.6, RED)
                    put_text(frame, "Rode com --coletor seunome (ou defina COLLECTOR_ID)", (10, 150), 0.5, RED)
                    put_text(frame, "(minúsculas, sem espaços nem acentos) e abra o programa novamente", (10, 172), 0.5, RED)

                if recording.state == "countdown":
                    remaining = max(1, math.ceil(recording.countdown_end - now))
                    put_centered_text(frame, str(remaining), height // 2 + 40, 5.0, YELLOW, 10)
                    put_centered_text(frame, "Prepare o sinal...", height // 2 + 90, 0.8, YELLOW, 2)
                elif recording.state == "recording":
                    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), RED, 8)
                    put_centered_text(
                        frame, f"GRAVANDO {len(recording.frames)}/{SEQUENCE_LENGTH}", height - 50, 0.9, RED, 2
                    )
                    progress = int((width - 40) * len(recording.frames) / SEQUENCE_LENGTH)
                    cv2.rectangle(frame, (20, height - 35), (width - 20, height - 20), WHITE, 1)
                    cv2.rectangle(frame, (20, height - 35), (20 + progress, height - 20), RED, -1)
                else:
                    if now < message_until:
                        put_text(frame, message, (10, height - 45), 0.6, message_color)
                    put_text(frame, "ESPAÇO: gravar   N/P ou setas: trocar sinal   Q: sair", (10, height - 15), 0.5)

                cv2.imshow(WINDOW_NAME, frame)

                # ---- teclado (waitKeyEx devolve códigos maiores para as setas)
                key = cv2.waitKeyEx(1)
                char = chr(key & 0xFF).lower() if 0 <= key < 256 else ""
                if char == "q":
                    break
                if recording.state == "idle":
                    if char == "n" or key == KEY_RIGHT:
                        sign_index = (sign_index + 1) % len(SIGNS)
                    elif char == "p" or key == KEY_LEFT:
                        sign_index = (sign_index - 1) % len(SIGNS)
                    elif key == KEY_SPACE and incompatible:
                        message = "Bloqueado: há amostras em formato antigo (veja o terminal)"
                        message_color, message_until = RED, now + MESSAGE_SECONDS
                    elif key == KEY_SPACE and collector_problem:
                        message = "Bloqueado: rode com --coletor seunome"
                        message_color, message_until = RED, now + MESSAGE_SECONDS
                    elif key == KEY_SPACE:
                        recording.start(now)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
