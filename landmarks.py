import numpy as np

SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21
HAND_FEATURES = NUM_LANDMARKS * 3
HAND_SLOTS = ("Right", "Left")
HAND_SLOT_NAMES = ("D", "E")
FRAME_FEATURES = len(HAND_SLOTS) * HAND_FEATURES
SAMPLE_SHAPE = (SEQUENCE_LENGTH, FRAME_FEATURES)

# O MediaPipe classifica a mão supondo uma imagem espelhada (como uma selfie).
# A imagem da webcam chega sem espelhamento, então os rótulos vêm invertidos.
SWAP_HANDEDNESS = True

# Índices dos pontos do MediaPipe usados como referência.
WRIST = 0
MIDDLE_MCP = 9  # base do dedo médio


def presence(frames: np.ndarray) -> np.ndarray:
    return np.any(frames != 0, axis=(2, 3))


def extract_hands(result, aspect: float) -> tuple[np.ndarray, list[bool]]:
    hands_xyz = np.zeros((len(HAND_SLOTS), NUM_LANDMARKS, 3), dtype=np.float32)
    present = [False] * len(HAND_SLOTS)

    hands = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if not handedness or len(landmarks) != NUM_LANDMARKS:
            continue
        category = handedness[0]
        label = category.category_name
        if SWAP_HANDEDNESS:
            label = "Left" if label == "Right" else "Right"
        hands.append((category.score, label, landmarks))

    # As mãos com maior confiança escolhem a vaga primeiro. Se o MediaPipe disser
    # que as duas mãos são "direitas", a segunda vai para a vaga que sobrou.
    for _, label, landmarks in sorted(hands, key=lambda h: h[0], reverse=True):
        slot = HAND_SLOTS.index(label) if label in HAND_SLOTS else 0
        if present[slot]:
            slot = 1 - slot
        if present[slot]:
            continue
        points = np.array([[lm.x * aspect, lm.y, lm.z * aspect] for lm in landmarks], dtype=np.float32)
        hands_xyz[slot] = points
        present[slot] = True

    return hands_xyz, present


def sample_at(frames: np.ndarray, positions: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float64)
    if frames.size and (frames.ndim != 4 or frames.shape[1:] != (2, NUM_LANDMARKS, 3)):
        raise ValueError(f"Os frames devem ter formato (T, 2, 21, 3), mas têm {frames.shape}")
    positions = np.clip(positions, 0, max(len(frames) - 1, 0))
    out = np.zeros((len(positions), len(HAND_SLOTS), NUM_LANDMARKS, 3), dtype=np.float32)
    if len(frames) == 0:
        return out

    present = presence(frames)
    nearest = np.round(positions).astype(int)
    for slot in range(frames.shape[1]):
        idx = np.flatnonzero(present[:, slot])
        if len(idx) == 0:
            continue
        # np.interp trabalha com uma série de cada vez: interpolamos cada uma
        # das 63 coordenadas (21 pontos x 3) separadamente.
        flat = frames[idx, slot].reshape(len(idx), -1)
        values = np.stack([np.interp(positions, idx, flat[:, k]) for k in range(flat.shape[1])], axis=1)
        keep = present[nearest, slot]
        out[keep, slot] = values[keep].reshape(-1, NUM_LANDMARKS, 3)
    return out


def resample_sequence(frames: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    return sample_at(frames, np.linspace(0, max(len(frames) - 1, 0), length))


def to_sample(frames: np.ndarray) -> np.ndarray:
    return resample_sequence(frames).reshape(SEQUENCE_LENGTH, FRAME_FEATURES)


def hand_size(hand: np.ndarray) -> float:
    return float(np.linalg.norm(hand[MIDDLE_MCP, :2] - hand[WRIST, :2]))
