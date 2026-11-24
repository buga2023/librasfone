import tempfile
import wave
from pathlib import Path

import numpy as np

# Trechos procurados no nome/idioma das vozes instaladas para achar uma voz em português.
PT_HINTS = ("pt-br", "pt_br", "portuguese", "portugu", "brazil", "brasil", "maria", "daniel", "francisca")
SPEECH_RATE = 170  # palavras por minuto


def _pick_voice(engine) -> str | None:
    for voice in engine.getProperty("voices"):
        languages = [str(lang) for lang in (getattr(voice, "languages", None) or [])]
        text = " ".join([voice.id or "", voice.name or ""] + languages).lower()
        if any(hint in text for hint in PT_HINTS):
            engine.setProperty("voice", voice.id)
            return voice.name
    return None


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if len(raw) != frames * channels * width:
        raise ValueError("WAV truncado: o arquivo tem menos amostras do que o cabeçalho informa")
    # Converte os inteiros do WAV para float entre -1 e 1, conforme o número de bytes por amostra.
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"WAV de {width * 8} bits não é suportado")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), rate


def render_phrases(texts: list[str]) -> tuple[dict[str, tuple[np.ndarray, int]], str]:
    import pyttsx3  # importado aqui para o programa funcionar com --sem-audio mesmo sem o pyttsx3

    out: dict[str, tuple[np.ndarray, int]] = {}
    with tempfile.TemporaryDirectory(prefix="librasfone_tts_") as tmp:
        engine = pyttsx3.init()
        try:
            voice_name = _pick_voice(engine) or "voz padrão do sistema (nenhuma voz em português encontrada)"
            engine.setProperty("rate", SPEECH_RATE)
            paths = {}
            for index, text in enumerate(sorted(set(texts))):
                path = Path(tmp) / f"{index:03d}.wav"
                engine.save_to_file(text, str(path))
                paths[text] = path
            engine.runAndWait()
            for text, path in paths.items():
                try:
                    audio, rate = read_wav(path)
                    if not len(audio):
                        raise ValueError("arquivo sem amostras")
                except (OSError, EOFError, ValueError, wave.Error) as exc:
                    print(f"[voz] não foi possível gerar {text!r}: {exc}")
                    continue
                out[text] = (_trim_silence(audio), rate)
        finally:
            try:
                engine.stop()
            except Exception:
                pass
    return out, voice_name


def _trim_silence(audio: np.ndarray, threshold: float = 0.01, pad: int = 400) -> np.ndarray:
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if len(loud) == 0:
        return audio
    return audio[max(loud[0] - pad, 0):loud[-1] + pad + 1]


def join_phrase(words: list[str], voices: dict[str, tuple[np.ndarray, int]],
                gap: float = 0.08) -> tuple[np.ndarray, int] | None:
    clips = [voices[word] for word in words if word in voices and len(voices[word][0])]
    if not clips:
        return None
    rate = clips[0][1]
    silence = np.zeros(int(gap * rate), dtype=np.float32)
    parts = []
    for audio, clip_rate in clips:
        if clip_rate != rate:
            count = max(1, int(round(len(audio) * rate / clip_rate)))
            audio = np.interp(np.linspace(0, len(audio) - 1, count), np.arange(len(audio)), audio)
        if parts:
            parts.append(silence)
        parts.append(audio.astype(np.float32))
    return np.concatenate(parts), rate
