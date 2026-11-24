import numpy as np

from features import to_hands
from landmarks import FRAME_FEATURES, SEQUENCE_LENGTH, WRIST, hand_size, presence, sample_at

# Intensidade das variações. NORMAL é usada no treino; STRONG serve para
# testar se o modelo aguenta variações maiores do que as que viu no treino.
#   crop:  fração da duração do sinal que é mantida (mínimo, máximo)
#   warp:  quanto a velocidade pode variar ao longo do sinal
#   roll:  rotação máxima no plano da imagem, em graus
#   yaw:   rotação máxima em torno do eixo vertical, em graus
#   amp:   fator que aumenta ou diminui a amplitude do movimento
#   noise: tremor, em fração do tamanho da mão
#   drop:  probabilidade de "perder" a mão em um frame
#   mirror: probabilidade de espelhar (trocar mão direita e esquerda)
NORMAL = dict(
    crop=(0.55, 1.0), warp=0.35, roll=12, yaw=25,
    amp=(0.7, 1.35), noise=0.04, drop=0.08, mirror=0.5,
)
STRONG = dict(
    crop=(0.45, 0.85), warp=0.5, roll=20, yaw=35,
    amp=(0.5, 1.6), noise=0.08, drop=0.15, mirror=0.5,
)


def _rotate(points: np.ndarray, center: np.ndarray, angle_deg: float, axes: tuple[int, int]) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    i, j = axes
    p = points.copy()
    u = points[..., i] - center[i]
    v = points[..., j] - center[j]
    p[..., i] = center[i] + c * u - s * v
    p[..., j] = center[j] + s * u + c * v
    return p


def _temporal_crop(hands: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    length = len(hands)
    span = rng.uniform(*cfg["crop"]) * (length - 1)
    start = rng.uniform(0, (length - 1) - span)
    gamma = np.exp(rng.uniform(-cfg["warp"], cfg["warp"]))
    t = np.linspace(0, 1, SEQUENCE_LENGTH)
    return sample_at(hands, start + span * t**gamma)


def _spatial_variation(hands, present, center, size, rng, cfg):
    # Afasta ou aproxima cada mão do centro, mantendo o formato da mão.
    k = rng.uniform(*cfg["amp"])
    offset = (hands[:, :, WRIST : WRIST + 1, :2] - center[:2]) * (k - 1)
    hands[..., :2] = np.where(present[..., None, None], hands[..., :2] + offset, 0)

    roll = rng.uniform(-cfg["roll"], cfg["roll"])
    yaw = rng.uniform(-cfg["yaw"], cfg["yaw"])
    rotated = _rotate(hands, center, roll, (0, 1))
    rotated = _rotate(rotated, center, yaw, (0, 2))
    hands = np.where(present[..., None, None], rotated, 0).astype(np.float32)

    noise = rng.normal(0, cfg["noise"] * size, hands.shape).astype(np.float32)
    hands = np.where(present[..., None, None], hands + noise, 0)
    return hands


def _drop_and_mirror(hands, present, center, rng, cfg):
    drop = rng.random(present.shape) < cfg["drop"]
    hands[drop] = 0

    if rng.random() < cfg["mirror"]:
        present = presence(hands)
        hands = hands[:, ::-1].copy()
        present = present[:, ::-1]
        hands[..., 0] = np.where(present[..., None], 2 * center[0] - hands[..., 0], 0)
    return hands


def augment(sample: np.ndarray, rng: np.random.Generator, cfg: dict = NORMAL) -> np.ndarray:
    hands = to_hands(sample)
    if not presence(hands).any():
        return sample.copy()

    hands = _temporal_crop(hands, rng, cfg)
    present = presence(hands)
    if not present.any():
        return sample.copy()

    pts = hands[present]
    center = pts[:, WRIST].mean(axis=0)
    sizes = [hand_size(h) for h in pts]
    size = float(np.median(sizes)) if sizes else 0.1

    hands = _spatial_variation(hands, present, center, size, rng, cfg)
    hands = _drop_and_mirror(hands, present, center, rng, cfg)

    return hands.reshape(SEQUENCE_LENGTH, FRAME_FEATURES).astype(np.float32)
