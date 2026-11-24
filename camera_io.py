import math
import sys
import threading
import time

import cv2
import numpy as np

CAMERA_SIZE = (640, 480)
FIRST_FRAME_TIMEOUT = 5.0  # segundos


def valid_fps(value: float, fallback: float = 30.0) -> float:
    return float(value) if math.isfinite(value) and value > 0 else fallback


def video_timestamp(seconds: float, previous: int) -> int:
    return max(int(seconds * 1000), previous + 1)


class CameraReader:

    def __init__(self, index: int):
        self.cap = None
        # No Windows, o backend DirectShow costuma abrir a câmera mais rápido e
        # aceitar o buffer de 1 frame. Se ele falhar, usamos o backend padrão.
        if sys.platform == "win32":
            candidate = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if candidate.isOpened():
                self.cap = candidate
            else:
                candidate.release()
        if self.cap is None:
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = valid_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self._frame = None
        self._time = 0.0
        self._count = 0  # número do último frame lido; read() compara com ele para saber se há frame novo
        self._failed = False
        self._new = threading.Condition()
        self._running = self.cap.isOpened()
        self._thread = threading.Thread(target=self._loop, name="librasfone-camera", daemon=True)
        if self._running:
            self._thread.start()
            # Se outro programa (Meet, Teams...) estiver usando a webcam, o Windows
            # às vezes "abre" a câmera mas nunca entrega imagem. Esperamos o
            # primeiro frame para detectar isso logo no início.
            with self._new:
                self._new.wait_for(lambda: self._count > 0 or self._failed, timeout=FIRST_FRAME_TIMEOUT)
            if self._count == 0:
                self.release()

    def isOpened(self) -> bool:
        return self._running and not self._failed

    def _loop(self) -> None:
        failures = 0
        try:
            while self._running:
                ok, frame = self.cap.read()
                captured_at = time.monotonic()
                if not ok:
                    # Falhas isoladas acontecem; muitas seguidas indicam câmera desconectada.
                    failures += 1
                    if failures > 50:
                        break
                    time.sleep(0.01)
                    continue
                failures = 0
                with self._new:
                    self._frame, self._time = frame, captured_at
                    self._count += 1
                    self._new.notify_all()
        except cv2.error:
            pass
        finally:
            with self._new:
                self._failed = True
                self._new.notify_all()  # acorda quem está esperando em read()

    def read(self, last_count: int) -> tuple[bool, np.ndarray | None, float, int]:
        with self._new:
            ready = self._new.wait_for(
                lambda: self._count > last_count or self._failed or not self._running,
                timeout=2.0,
            )
            if not ready or self._failed or not self._running or self._frame is None:
                return False, None, 0.0, last_count
            return True, self._frame, self._time, self._count

    def release(self) -> None:
        with self._new:
            self._running = False
            self._new.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        # Se a thread ainda estiver presa em cap.read(), liberar a câmera
        # faz essa leitura retornar; então esperamos a thread de novo.
        self.cap.release()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
