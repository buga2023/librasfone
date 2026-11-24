import argparse
import json
import math
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import joblib

from camera_io import CameraReader, valid_fps, video_timestamp
from display import GRAY, GREEN, RED, WHITE, YELLOW, draw_subtitle, put_text
from landmark_viewer import CAMERA_INDEX, create_landmarker, detect_hands, draw_hands
from landmarks import HAND_SLOT_NAMES, extract_hands
from recognition import predict_sign, validate_bundle
from segmenter import SignSegmenter, calibrate_rest_line
from translation import Translator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "librasfone_model.joblib"
FALAS_PATH = ROOT / "falas.json"

WINDOW_NAME = "LibrasFone (Q para sair)"

DEFAULT_THRESHOLD = 0.7
DEFAULT_REST_LINE = 0.85
CALIBRATION_TIME = 1.5  # segundos com as mãos em descanso no início
DEFAULT_PHRASE_PAUSE = 3.0

# Códigos de cv2.waitKeyEx no Windows.
KEY_UP = 0x260000
KEY_DOWN = 0x280000
KEY_ENTER = (13, 10)
KEY_SPACE = 32
KEY_BACKSPACE = 8


def load_falas() -> dict[str, str]:
    if not FALAS_PATH.exists():
        return {}
    data = json.loads(FALAS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("falas.json deve conter um objeto com o texto de cada palavra")
    words = {k: v for k, v in data.items() if not k.startswith("_")}
    if any(not isinstance(v, str) for v in words.values()):
        raise ValueError("falas.json deve conter um texto para cada palavra")
    return words


def spoken_text(label: str, falas: dict[str, str]) -> str:
    return falas.get(label, label.replace("_", " "))


def load_model():
    if not MODEL_PATH.exists():
        print(f"Modelo não encontrado em {MODEL_PATH}. Rode antes: python train.py", file=sys.stderr)
        return None
    try:
        return validate_bundle(joblib.load(MODEL_PATH))
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        print(f"Modelo inválido: {exc}", file=sys.stderr)
        return None


def test_audio(args) -> int:
    from audio_out import AudioOut, find_cable_device, list_output_devices
    from tts import render_phrases

    print("Dispositivos de saída de áudio:")
    for index, name, api in list_output_devices():
        print(f"  [{index}] {name}  ({api})")
    cable = find_cable_device()
    if cable is None:
        print("\nVB-Cable NÃO encontrado. Instale em https://vb-audio.com/Cable/ e reinicie o computador.")
    else:
        print(f"\nVB-Cable encontrado: dispositivo [{cable}]")
    phrase = "Teste do LibrasFone. Se você está ouvindo, o áudio está funcionando."
    audio, voice = render_phrases([phrase])
    print(f"Voz: {voice}")
    if phrase not in audio:
        print("Não foi possível gerar a voz (pyttsx3).")
        return 1
    out = AudioOut(use_cable=True, use_monitor=not args.sem_monitor, keep_open=not args.audio_simples)
    try:
        print(f"Tocando em: {out.describe()}")
        playback = out.play(*audio[phrase])
        if playback is None:
            print("Nenhuma saída de áudio disponível para o teste.", file=sys.stderr)
            return 1
        time.sleep(0.3)
        while out.busy():
            time.sleep(0.05)
        if playback.started is None:
            print("O dispositivo não iniciou a reprodução.", file=sys.stderr)
            return 1
        print(f"O som começou {1000 * (playback.started - playback.queued_at):.0f} ms depois do pedido.")
    finally:
        out.close()
    print("Pronto. Para conferir o microfone virtual, abra Configurações de som > Entrada >")
    print("'CABLE Output' e veja se a barra de volume se move enquanto o teste toca.")
    return 0


CAMERA_BUSY_HELP = """Erro: não foi possível usar a webcam.
A webcam só funciona em um programa por vez. Se o Meet (ou outro app) estiver usando ela:
  1. No Meet, troque a câmera para "OBS Virtual Camera" (ou desligue a câmera).
  2. Rode o librasfone com --camera-virtual. Ele lê a webcam e envia a imagem,
     com a legenda, para a OBS Virtual Camera que o Meet está usando."""


def open_virtual_camera(width: int, height: int, fps: float):
    try:
        import pyvirtualcam
    except ImportError:
        print("[câmera virtual] instale com: pip install pyvirtualcam  (e tenha o OBS Studio instalado)")
        return None
    try:
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"[câmera virtual] ligada: {cam.device} — no Meet, escolha essa câmera")
        return cam
    except Exception as exc:
        print(f"[câmera virtual] não foi possível ligar ({exc}). O OBS Studio está instalado?")
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LibrasFone: tradutor de Libras ao vivo.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="índice da webcam (padrão 0)")
    parser.add_argument("--video", type=Path, help="usa um arquivo de vídeo no lugar da câmera")
    parser.add_argument("--modo", choices=("frase", "palavra"), default="frase",
                        help="frase: junta os sinais e fala depois da pausa (padrão); palavra: fala cada sinal na hora")
    parser.add_argument("--pausa-frase", type=float, default=DEFAULT_PHRASE_PAUSE,
                        help="segundos com as mãos em descanso para falar a frase (padrão 3)")
    parser.add_argument("--limiar", type=float, default=DEFAULT_THRESHOLD, help="confiança mínima (0-1)")
    parser.add_argument("--linha", type=float, default=None,
                        help="linha de descanso fixa (0 = topo, 1 = base). Sem ela, a linha é calibrada automaticamente")
    parser.add_argument("--sem-audio", action="store_true", help="não fala nada, só mostra o texto")
    parser.add_argument("--sem-monitor", action="store_true", help="não toca no alto-falante, só no VB-Cable")
    parser.add_argument("--audio-simples", action="store_true", help="abre o dispositivo de áudio a cada fala")
    parser.add_argument("--testar-audio", action="store_true", help="testa a voz e o VB-Cable e sai")
    parser.add_argument("--camera-virtual", action="store_true", help="envia a imagem para a câmera virtual do OBS")
    parser.add_argument("--sem-janela", action="store_true", help="não abre janela (só mostra no terminal)")
    parser.add_argument("--registrar", action="store_true",
                        help="grava eventos, latências e uso de CPU em testes/sessao_<data>/")
    parser.add_argument("--roteiro", action="store_true",
                        help="teste de acerto: pede cada palavra e anota o que foi reconhecido (liga --registrar)")
    parser.add_argument("--repeticoes", type=int, default=3, help="repetições de cada palavra no --roteiro (padrão 3)")
    # Usado pelo painel.py: o programa para quando recebe "parar" na entrada padrão.
    parser.add_argument("--painel", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # O argparse aceita "nan" e "inf" como float. Comparações com NaN são sempre
    # falsas, então `0 <= valor <= 1` já recusa os dois casos.
    for name in ("limiar", "linha"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"--{name} deve estar entre 0 e 1")
    # Aqui isfinite é necessário: "nan" passaria por `<= 0`, e com "inf" a frase nunca seria falada.
    if not math.isfinite(args.pausa_frase) or args.pausa_frase <= 0:
        parser.error("--pausa-frase deve ser maior que zero")
    if args.camera < 0:
        parser.error("--camera deve ser um índice não negativo")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        # ExitStack guarda as funções de liberação (câmera, áudio, câmera virtual)
        # registradas ao longo de run() e chama todas no final, mesmo com erro ou
        # com `return` no meio. Assim run() não precisa de vários try/finally aninhados.
        with ExitStack() as resources:
            resources.callback(cv2.destroyAllWindows)
            return run(args, resources)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


def handle_segment(segment, model, labels, args, falas, translator, log=None, roteiro=None) -> tuple[str, tuple]:
    prediction = predict_sign(model, labels, segment.frames, args.limiar)
    label, conf = prediction.label, prediction.confidence
    if roteiro is not None:
        roteiro.record(label, prediction.status, conf)
    # Latência entre as mãos descerem e o texto aparecer (só faz sentido ao vivo).
    delay_text, delay_ms = "", None
    if not args.video:
        delay_ms = 1000 * (time.monotonic() - segment.lowered_at)
        delay_text = f" | fim do sinal -> texto: {delay_ms:.0f} ms"
    seconds = segment.end - segment.start
    text = spoken_text(label, falas)
    if log is not None:
        kind = {"unknown": "nao_reconhecido", "uncertain": "incerto", "accepted": "palavra"}[prediction.status]
        log.event(kind, "" if prediction.status == "unknown" else text, conf, seconds, delay_ms)
    if prediction.status == "unknown":
        print(f"[{seconds:.1f}s] não reconhecido ({conf:.0%}){delay_text}")
        return "Sinal não reconhecido", GRAY
    if prediction.status == "uncertain":
        print(f"[{seconds:.1f}s] incerto: {text} ({conf:.0%}) — ignorado{delay_text}")
        return f"Incerto: {text}? ({conf:.0%})", YELLOW
    print(f"[{seconds:.1f}s] + {text} ({conf:.0%}){delay_text}")
    translator.add_word(text, segment.lowered_at)
    return f"{text.upper()} ({conf:.0%})", GREEN


def _wait_stop_command(stop_requested: threading.Event) -> None:
    for line in sys.stdin:
        if line.strip().lower() == "parar":
            break
    stop_requested.set()


def run(args, resources: ExitStack) -> int:
    if args.testar_audio:
        return test_audio(args)

    bundle = load_model()
    if bundle is None:
        return 1
    model, labels = bundle["model"], bundle["labels"]
    falas = load_falas()
    words = [label for label in labels if not label.startswith("_")]
    print("Palavras que o modelo conhece: " + ", ".join(spoken_text(w, falas) for w in words))

    # ---- áudio (opcional: se falhar, o programa continua só com legenda)
    audio_out, voices = None, {}
    if not args.sem_audio:
        try:
            from audio_out import AudioOut
            from tts import render_phrases

            voices, voice_name = render_phrases([spoken_text(w, falas) for w in words])
            audio_out = AudioOut(use_cable=True, use_monitor=not args.sem_monitor, keep_open=not args.audio_simples)
            resources.callback(audio_out.close)
            print(f"Voz: {voice_name}")
            print(f"Áudio: {audio_out.describe()}")
            if audio_out.cable is None:
                print("  -> Instale o VB-Cable (https://vb-audio.com/Cable/) para a voz ir para a chamada.")
        except Exception as exc:
            print(f"[voz desligada] {exc}")
            audio_out = None

    # ---- entrada de vídeo
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        resources.callback(cap.release)
        video_fps = valid_fps(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened():
            print("Erro: não foi possível abrir o vídeo.", file=sys.stderr)
            return 1
    else:
        cap = CameraReader(args.camera)
        resources.callback(cap.release)
        video_fps = cap.fps
        if not cap.isOpened():
            print(CAMERA_BUSY_HELP, file=sys.stderr)
            return 1

    mode_text = f"modo frase (fala depois de {args.pausa_frase:.1f}s parado)" if args.modo == "frase" else "modo palavra"
    print(f"Pronto — {mode_text}.")

    log = None
    if args.roteiro:
        args.registrar = True  # o roteiro grava o resultado na pasta da sessão
    if args.registrar:
        from medicao import SessionLog
        log = SessionLog(args, len(words))
        resources.callback(log.close)
        log.header.append(f"Áudio: {audio_out.describe() if audio_out is not None else 'desligado'}")
    roteiro = None
    if args.roteiro:
        from roteiro import Roteiro
        roteiro = Roteiro(words, falas, args.repeticoes, log.folder)
        resources.callback(roteiro.close)
        print(f"Roteiro: {len(roteiro.items)} sinais ({len(words)} palavras x {args.repeticoes}). "
              "N pula a palavra, D descarta a última tentativa.")

    translator = Translator(args, falas, voices, audio_out, log)
    segmenter = SignSegmenter(args.linha if args.linha is not None else DEFAULT_REST_LINE)
    # Sem --linha, os primeiros CALIBRATION_TIME segundos servem para medir a posição de descanso.
    calibrating = args.linha is None
    calib_buffer: list = []
    calib_start = None
    last_result, last_color, last_until = "", WHITE, 0.0
    vcam = None
    stop_requested = threading.Event()
    if args.painel:
        threading.Thread(target=_wait_stop_command, args=(stop_requested,), daemon=True).start()
    start = time.monotonic()
    last_ts = -1
    frame_index = 0
    frame_count = 0
    fps_shown = 0.0
    last_frame_time = None

    with create_landmarker() as landmarker:
        while not stop_requested.is_set():
            # ---- leitura do frame e do instante `t` (em segundos) a que ele corresponde
            if args.video:
                ok, frame = cap.read()
                if not ok:
                    print("Fim do vídeo.")
                    break
                # No vídeo, o tempo vem da posição do frame. Com janela, esperamos
                # para reproduzir na velocidade normal.
                t = start + frame_index / video_fps
                if not args.sem_janela:
                    ahead = t - time.monotonic()
                    if ahead > 0:
                        time.sleep(ahead)
                ts = video_timestamp(frame_index / video_fps, last_ts)
            else:
                ok, frame, t, frame_count = cap.read(frame_count)
                if not ok:
                    print(CAMERA_BUSY_HELP, file=sys.stderr)
                    return 1
                ts = video_timestamp(t - start, last_ts)
            last_ts = ts
            frame_index += 1
            now = time.monotonic() if not args.video else t

            # FPS exibido: média móvel exponencial, para o número não ficar pulando.
            if last_frame_time is not None and t > last_frame_time:
                current_fps = 1.0 / (t - last_frame_time)
                fps_shown = 0.9 * fps_shown + 0.1 * current_fps if fps_shown else current_fps
            last_frame_time = t
            if log is not None:
                log.fps = fps_shown

            # ---- detecção das mãos e segmentação
            height, width = frame.shape[:2]
            result = detect_hands(landmarker, frame, ts)
            hands, present = extract_hands(result, width / height)

            segments = []
            if calibrating:
                calib_start = t if calib_start is None else calib_start
                calib_buffer.append((t, hands, present))
                if t - calib_start >= CALIBRATION_TIME:
                    line, why = calibrate_rest_line([(h, p) for _, h, p in calib_buffer])
                    segmenter = SignSegmenter(line)
                    calibrating = False
                    print(f"Calibrado: {why} -> linha de descanso em {line:.2f}")
                    # Reprocessa os frames da calibração com a linha nova, para
                    # não perder um sinal feito durante a calibração.
                    segments = [s for s in (segmenter.update(h, p, ft) for ft, h, p in calib_buffer) if s is not None]
                    calib_buffer, calib_start = [], None
            else:
                segment = segmenter.update(hands, present, t)
                segments = [segment] if segment is not None else []
            for message in segmenter.messages:
                print(message)
            segmenter.messages.clear()

            # ---- reconhecimento
            for segment in segments:
                last_result, last_color = handle_segment(segment, model, labels, args, falas, translator, log, roteiro)
                last_until = now + 2.5

            progress = None if calibrating else translator.tick(now, segmenter.capturing)
            subtitle_words, subtitle_color = translator.subtitle(now)

            # ---- câmera virtual: recebe a imagem sem os desenhos de depuração, só com a legenda
            if args.camera_virtual and vcam is None:
                vcam = open_virtual_camera(width, height, video_fps)
                if vcam is not None:
                    resources.callback(vcam.close)
                else:
                    args.camera_virtual = False  # não tenta abrir de novo a cada frame
            if vcam is not None:
                clean = frame.copy()
                draw_subtitle(clean, subtitle_words, subtitle_color)
                vcam.send(clean)

            if args.sem_janela:
                continue

            # ---- janela de controle
            draw_hands(frame, result)
            frame = cv2.flip(frame, 1)  # espelha só para exibir

            line_y = int(segmenter.rest_line * height)
            cv2.line(frame, (0, line_y), (width, line_y), YELLOW if segmenter.capturing else GRAY, 1)
            put_text(frame, "linha de descanso", (width - 210, line_y - 8), 0.5, GRAY, 1)

            if calibrating:
                put_text(frame, "CALIBRANDO... deixe as mãos em descanso", (10, 32), 0.7, YELLOW)
            elif segmenter.capturing:
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), RED, 6)
                put_text(frame, "CAPTANDO SINAL... abaixe as mãos ao terminar", (10, 32), 0.7, RED)
            elif translator.phrase:
                put_text(frame, "Faça outro sinal ou fique parado para falar a frase", (10, 32), 0.7, GREEN)
            else:
                put_text(frame, "Pronto: levante as mãos, faça o sinal e abaixe", (10, 32), 0.7, GREEN)
            sides = "+".join(n for n, here in zip(HAND_SLOT_NAMES, present) if here)
            put_text(frame, f"Mãos: {sum(present)} {f'({sides})' if sides else ''}   {fps_shown:.0f} fps", (10, 62), 0.6)
            if now < last_until:
                put_text(frame, last_result, (10, 98), 0.9, last_color)
            if audio_out is not None:
                status = "MUDO" if audio_out.muted else ("VB-Cable OK" if audio_out.cable is not None else "sem VB-Cable")
                put_text(frame, f"Áudio: {status}  |  {'frase' if args.modo == 'frase' else 'palavra'}", (10, 128), 0.5, GRAY, 1)

            if roteiro is not None:
                put_text(frame, roteiro.progress_text(), (10, 170), 1.0, YELLOW if not roteiro.done else GREEN)
                put_text(frame, "N pular palavra   D descartar tentativa", (10, 200), 0.55, GRAY, 1)
            draw_subtitle(frame, subtitle_words, subtitle_color, progress)
            put_text(frame, "Q sair  ENTER falar  BACKSPACE apagar  C limpar  M mudo  R calibrar",
                     (10, height - 12), 0.45, GRAY, 1)
            cv2.imshow(WINDOW_NAME, frame)

            # ---- teclado
            key = cv2.waitKeyEx(1)
            char = chr(key & 0xFF).lower() if 0 <= key < 256 else ""
            if char == "q":
                break
            if key in KEY_ENTER or key == KEY_SPACE:
                translator.speak(now)
            elif key == KEY_BACKSPACE:
                if log is not None and translator.phrase:
                    log.event("palavra_apagada", translator.phrase[-1])
                translator.undo()
            elif char == "c":
                translator.clear()
            elif char == "n" and roteiro is not None:
                roteiro.skip()
            elif char == "d" and roteiro is not None:
                roteiro.discard_last()
            elif char == "m" and audio_out is not None:
                audio_out.muted = not audio_out.muted
                if audio_out.muted:
                    audio_out.stop()
            elif char == "r":
                calibrating, calib_buffer, calib_start = True, [], None
                segmenter = SignSegmenter(segmenter.rest_line)
            elif key == KEY_UP:
                segmenter.rest_line = max(0.1, segmenter.rest_line - 0.02)
            elif key == KEY_DOWN:
                segmenter.rest_line = min(0.99, segmenter.rest_line + 0.02)

        # Fim do laço: fala o que sobrou e, no modo vídeo, espera o áudio terminar.
        translator.speak(time.monotonic())
        if audio_out is not None and args.video:
            time.sleep(0.3)
            while audio_out.busy():
                time.sleep(0.05)
            translator.tick(time.monotonic(), False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
