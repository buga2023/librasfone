import numpy as np

from landmarks import FRAME_FEATURES, MIDDLE_MCP, NUM_LANDMARKS, SEQUENCE_LENGTH, WRIST, presence

SHAPE_FEATURES = NUM_LANDMARKS * 3
TRAJECTORY_FEATURES = 2
MIN_HAND_SIZE = 1e-3  # evita divisão por zero quando a mão aparece "achatada"
PER_HAND = SHAPE_FEATURES + TRAJECTORY_FEATURES + 1
FEATURES_PER_FRAME = 2 * PER_HAND
FEATURE_SIZE = SEQUENCE_LENGTH * FEATURES_PER_FRAME


def to_hands(sample: np.ndarray) -> np.ndarray:
    return np.asarray(sample, dtype=np.float32).reshape(len(sample), 2, NUM_LANDMARKS, 3)


def dominant_slot(hands: np.ndarray) -> int:
    present = presence(hands)
    scores = []
    for slot in range(2):
        idx = np.flatnonzero(present[:, slot])
        if len(idx) == 0:
            scores.append(0.0)
            continue
        wrist = hands[idx, slot, WRIST, :2]
        motion = np.abs(np.diff(wrist, axis=0)).sum() if len(idx) > 1 else 0.0
        height = 1.0 - float(np.median(wrist[:, 1]))
        scores.append(len(idx) * (1.0 + 20.0 * motion / max(len(idx), 1)) * (0.5 + height))
    return int(np.argmax(scores))


def canonicalize(hands: np.ndarray) -> np.ndarray:
    hands = hands.copy()
    if dominant_slot(hands) == 1:
        present = presence(hands)
        hands = hands[:, ::-1].copy()
        present = present[:, ::-1]
        hands[..., 0] = np.where(present[..., None], -hands[..., 0], 0.0)
    return hands


def sample_features(sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample)
    if sample.ndim != 2 or sample.shape[1] != FRAME_FEATURES:
        raise ValueError(f"Amostra com formato inesperado: {sample.shape}")
    hands = canonicalize(to_hands(sample))
    present = presence(hands)
    out = np.zeros((len(hands), 2, PER_HAND), dtype=np.float32)
    if not present.any():
        return out.reshape(-1)

    # Tamanho de cada mão em cada frame (NaN onde a mão não aparece).
    sizes = np.linalg.norm(hands[:, :, MIDDLE_MCP, :2] - hands[:, :, WRIST, :2], axis=-1)
    sizes = np.where(present, sizes, np.nan)
    mean_size = max(float(np.nanmedian(sizes)), MIN_HAND_SIZE)

    # A trajetória é medida a partir da posição média do pulso da mão dominante
    # (ou da outra mão, se a dominante nunca aparecer).
    ref_slot = 0 if present[:, 0].any() else 1
    ref = hands[present[:, ref_slot], ref_slot, WRIST, :2].mean(axis=0)

    for slot in range(2):
        idx = present[:, slot]
        if not idx.any():
            continue
        h = hands[idx, slot]
        frame_size = np.maximum(sizes[idx, slot], MIN_HAND_SIZE)
        shape = (h - h[:, WRIST : WRIST + 1]) / frame_size[:, None, None]
        traj = (h[:, WRIST, :2] - ref) / mean_size
        out[idx, slot, :SHAPE_FEATURES] = shape.reshape(len(h), -1)
        out[idx, slot, SHAPE_FEATURES : SHAPE_FEATURES + TRAJECTORY_FEATURES] = traj
        out[idx, slot, -1] = 1.0

    return out.reshape(-1)


def batch_features(samples) -> np.ndarray:
    return np.stack([sample_features(s) for s in samples]).astype(np.float32)
