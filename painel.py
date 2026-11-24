"""Painel simples para iniciar e parar o LibrasFone sem usar a linha de comando.

Rode com: python painel.py
O painel monta os mesmos argumentos de librasfone.py e mostra a saída dele.
"""
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "librasfone.py"


def find_cameras(limit: int = 5) -> list[int]:
    import cv2
    found = []
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for index in range(limit):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            found.append(index)
        cap.release()
    return found


class Painel:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.process: subprocess.Popen | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        root.title("LibrasFone")
        root.resizable(False, False)
        frame = ttk.Frame(root, padding=12)
        frame.grid()

        self.camera = tk.StringVar(value="0")
        self.modo = tk.StringVar(value="frase")
        self.pausa = tk.DoubleVar(value=3.0)
        self.limiar = tk.DoubleVar(value=0.7)
        self.monitor = tk.BooleanVar(value=True)
        self.camera_virtual = tk.BooleanVar(value=False)
        self.registrar = tk.BooleanVar(value=False)
        self.roteiro = tk.BooleanVar(value=False)
        self.repeticoes = tk.IntVar(value=3)

        row = 0
        ttk.Label(frame, text="Webcam").grid(row=row, column=0, sticky="w")
        self.camera_box = ttk.Combobox(frame, textvariable=self.camera, values=["0"], width=6)
        self.camera_box.grid(row=row, column=1, sticky="w")
        ttk.Button(frame, text="Procurar", command=self.search_cameras).grid(row=row, column=2, sticky="w")
        row += 1
        ttk.Label(frame, text="Modo").grid(row=row, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Frase", variable=self.modo, value="frase").grid(row=row, column=1, sticky="w")
        ttk.Radiobutton(frame, text="Palavra", variable=self.modo, value="palavra").grid(row=row, column=2, sticky="w")
        row += 1
        ttk.Label(frame, text="Pausa para falar a frase (s)").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1.0, to=6.0, increment=0.5, textvariable=self.pausa, width=6).grid(
            row=row, column=1, sticky="w")
        row += 1
        ttk.Label(frame, text="Confiança mínima").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0.5, to=0.95, increment=0.05, textvariable=self.limiar, width=6).grid(
            row=row, column=1, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Ouvir a voz também no alto-falante", variable=self.monitor).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Enviar a imagem para a câmera virtual (OBS)", variable=self.camera_virtual).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Registrar a sessão para o teste (latência e CPU)", variable=self.registrar).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Modo roteiro: pede cada palavra e mede o acerto", variable=self.roteiro).grid(
            row=row, column=0, columnspan=2, sticky="w")
        ttk.Spinbox(frame, from_=1, to=10, increment=1, textvariable=self.repeticoes, width=4).grid(
            row=row, column=2, sticky="w")
        row += 1
        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=3, pady=(8, 4), sticky="w")
        self.start_button = ttk.Button(buttons, text="Iniciar", command=self.start)
        self.start_button.grid(row=0, column=0, padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="Parar", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 6))
        ttk.Button(buttons, text="Testar áudio", command=self.test_audio).grid(row=0, column=2)
        row += 1
        self.status = ttk.Label(frame, text="Parado")
        self.status.grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        self.output = tk.Text(frame, width=72, height=14, state="disabled")
        self.output.grid(row=row, column=0, columnspan=3, pady=(6, 0))

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.show_output)

    def search_cameras(self) -> None:
        self.status.config(text="Procurando webcams...")
        self.root.update_idletasks()
        cameras = [str(i) for i in find_cameras()]
        self.camera_box.config(values=cameras or ["0"])
        if cameras:
            self.camera.set(cameras[0])
        self.status.config(text=f"Webcams encontradas: {', '.join(cameras) if cameras else 'nenhuma'}")

    def arguments(self) -> list[str]:
        args = ["--camera", self.camera.get().strip() or "0", "--modo", self.modo.get(),
                "--pausa-frase", f"{self.pausa.get():.1f}", "--limiar", f"{self.limiar.get():.2f}"]
        if not self.monitor.get():
            args.append("--sem-monitor")
        if self.camera_virtual.get():
            args.append("--camera-virtual")
        if self.registrar.get():
            args.append("--registrar")
        if self.roteiro.get():
            args += ["--roteiro", "--repeticoes", str(int(self.repeticoes.get()))]
        return args

    def launch(self, args: list[str]) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(MAIN), "--painel", *args], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status.config(text="Rodando")

    def start(self) -> None:
        try:
            self.launch(self.arguments())
        except (tk.TclError, ValueError):
            self.status.config(text="Valor inválido em pausa ou confiança")

    def test_audio(self) -> None:
        self.launch(["--testar-audio"] + ([] if self.monitor.get() else ["--sem-monitor"]))

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            # Pede para o programa terminar sozinho, assim ele fala o que sobrou e salva o registro.
            self.process.stdin.write("parar\n")
            self.process.stdin.flush()
        except OSError:
            pass
        self.root.after(5000, self.force_stop, self.process)

    def force_stop(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.kill()

    def read_output(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            self.lines.put(line)
        process.wait()
        self.lines.put(None)

    def show_output(self) -> None:
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    self.process = None
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status.config(text="Parado")
                    continue
                self.output.config(state="normal")
                self.output.insert("end", line)
                self.output.see("end")
                self.output.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self.show_output)

    def close(self) -> None:
        process = self.process
        self.stop()
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Painel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
