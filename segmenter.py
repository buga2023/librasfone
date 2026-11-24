from collections import deque
from dataclasses import dataclass

import numpy as np

from landmarks import hand_size

# Tempos em segundos.
START_TIME = 0.06      # mãos acima da linha por esse tempo -> começa a captar
END_TIME = 0.2         # mãos abaixo da linha (ou fora da imagem) por esse tempo -> termina
MIN_SIGN_TIME = 0.3    # trechos mais curtos são descartados (movimento acidental)
MAX_SIGN_TIME = 6.0    # força o fim da captura se as mãos não descerem
PRE_TIME = 0.15        # guarda um pouco antes do início, para não cortar o começo do movimento
TAIL_TIME = 0.07       # guarda um pouco depois de as mãos descerem

# A linha fica RAISE_HAND_SIZES "tamanhos de mão" acima da posição de descanso.
RAISE_HAND_SIZES = 1.3
# Linha usada quando, no descanso, as mãos ficam fora da imagem.
ABSENT_REST_LINE = 0.95

# Detecção de "mãos paradas acima da linha" (veja SignSegmenter._resting).
REST_CHECK_TIME = 1.2  # janela de tempo analisada, em segundos
REST_SPREAD = 0.25     # movimento máximo, em tamanhos de mão, para considerar as mãos paradas
REST_DROP = 2.0        # quanto as mãos precisam ter descido, em tamanhos de mão


def top_of_hands(hands: np.ndarray, present) -> float | None:
    ys = [hands[s, :, 1].min() for s in range(2) if present[s]]
    return min(ys) if ys else None


def hand_spread(frames: list[np.ndarray], present: list) -> float | None:
    best = None
    for s in range(2):
        seen = [h for h, p in zip(frames, present) if p[s]]
        if len(seen) < 3 or len(seen) < 0.8 * len(frames):
            continue
        pts = np.stack([h[s, :, :2] for h in seen])
        size = max(float(np.median([hand_size(h[s]) for h in seen])), 1e-3)
        dev = float(np.linalg.norm(pts - pts.mean(axis=0), axis=2).max(axis=0).mean()) / size
        best = dev if best is None else max(best, dev)
    return best


def rest_line_from(frames: list[np.ndarray], present: list) -> float | None:
    tops, sizes = [], []
    for hands, pres in zip(frames, present):
        top = top_of_hands(hands, pres)
        if top is not None:
            tops.append(top)
            sizes += [hand_size(hands[s]) for s in range(2) if pres[s]]
    if not tops:
        return None
    rest = float(np.percentile(tops, 90))
    margin = max(0.06, RAISE_HAND_SIZES * float(np.median(sizes)))
    return float(np.clip(rest - margin, 0.2, ABSENT_REST_LINE))


def calibrate_rest_line(frames: list) -> tuple[float, str]:
    with_hands = [(h, p) for h, p in frames if any(p)]
    if not with_hands or len(with_hands) < 0.5 * len(frames):
        return ABSENT_REST_LINE, "mãos fora da câmera no descanso"
    line = rest_line_from([h for h, _ in with_hands], [p for _, p in with_hands])
    return line, "mãos em descanso visíveis na câmera"


@dataclass(eq=False)
class Segment:

    frames: np.ndarray
    start: float
    end: float
    lowered_at: float

    def __len__(self) -> int:
        return len(self.frames)


class SignSegmenter:

    def __init__(self, rest_line: float):
        self.rest_line = rest_line
        self.capturing = False
        self.pre = deque()  # últimos frames antes do início da captura (PRE_TIME)
        self.frames: list[tuple[float, np.ndarray, list]] = []  # (tempo, mãos, presença)
        self.active_since: float | None = None
        self.inactive_since: float | None = None
        self.last_rest_check = 0.0
        self.messages: list[str] = []  # avisos para o programa principal mostrar no terminal

    def update(self, hands: np.ndarray, present, t: float) -> Segment | None:
        present = list(present)
        top = top_of_hands(hands, present)
        active = top is not None and top < self.rest_line

        if not self.capturing:
            self._wait_for_start(hands, present, t, active)
            return None

        self.frames.append((t, hands, present))
        duration = t - self.frames[0][0]

        if not active:
            self.inactive_since = t if self.inactive_since is None else self.inactive_since
            if t - self.inactive_since >= END_TIME:
                return self._finish(t, self.inactive_since + TAIL_TIME, self.inactive_since)
            return None
        self.inactive_since = None

        # A verificação de mãos paradas é mais cara, então roda no máximo a cada 0,25 s.
        if duration >= REST_CHECK_TIME and t - self.last_rest_check >= 0.25:
            self.last_rest_check = t
            if self._resting(t):
                return None
        if duration >= MAX_SIGN_TIME:
            return self._finish(t, t, t)
        return None

    def _wait_for_start(self, hands: np.ndarray, present: list, t: float, active: bool) -> None:
        self.pre.append((t, hands, present))
        while self.pre and t - self.pre[0][0] > PRE_TIME:
            self.pre.popleft()
        if not active:
            self.active_since = None
            return
        self.active_since = t if self.active_since is None else self.active_since
        if t - self.active_since >= START_TIME:
            self.capturing = True
            self.frames = list(self.pre)
            self.inactive_since = None

    def _resting(self, t: float) -> bool:
        window = [f for f in self.frames if t - f[0] <= REST_CHECK_TIME]
        frames = [h for _, h, _ in window]
        present = [p for _, _, p in window]
        if len(frames) < 3:
            return False
        spread = hand_spread(frames, present)
        if spread is None or spread >= REST_SPREAD:
            return False
        tops_now = [top_of_hands(h, p) for h, p in zip(frames, present) if any(p)]
        tops_all = [top_of_hands(h, p) for _, h, p in self.frames if any(p)]
        sizes = [hand_size(h[s]) for h, p in zip(frames, present) for s in range(2) if p[s]]
        still_since_start = len(window) >= 0.9 * len(self.frames)
        # y cresce para baixo: valor positivo significa que as mãos estão mais
        # baixas agora do que nas posições altas (percentil 25) da captura.
        dropped = float(np.median(tops_now)) - float(np.percentile(tops_all, 25))
        if not (still_since_start or dropped >= REST_DROP * float(np.median(sizes))):
            return False
        new_line = rest_line_from(frames, present)
        # Só aceita mover a linha para cima (y menor); descer a linha pioraria o problema.
        if new_line is None or new_line >= self.rest_line:
            return False
        self.messages.append(f"Mãos paradas acima da linha: linha recalibrada automaticamente para {new_line:.2f}")
        self.rest_line = new_line
        self._reset()
        return True

    def _reset(self) -> None:
        self.capturing = False
        self.frames = []
        self.pre.clear()
        self.active_since = None
        self.inactive_since = None

    def _finish(self, t: float, cut: float, lowered_at: float) -> Segment | None:
        frames = [h for ft, h, _ in self.frames if ft <= cut]
        start = self.frames[0][0]
        self._reset()
        if len(frames) < 3 or cut - start < MIN_SIGN_TIME:
            return None
        return Segment(np.stack(frames), start, t, lowered_at)
