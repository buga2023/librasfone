"""Registro de uma sessão de teste: eventos, latências e uso de CPU.

Usado com `python librasfone.py --registrar`. Cada sessão cria uma pasta em
testes/ com três arquivos:
  eventos.csv  um evento por linha (sinal reconhecido, frase enviada, voz iniciada...)
  cpu.csv      uma amostra por segundo do uso de CPU e memória
  resumo.txt   médias e máximos, gerado ao fechar o programa
"""
import csv
import os
import platform
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "testes"
CPU_INTERVAL = 1.0  # segundos entre as amostras de CPU

EVENT_FIELDS = ["hora", "t_s", "evento", "texto", "confianca", "duracao_sinal_s", "latencia_ms", "detalhe"]


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low, high = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def _stats(values: list[float]) -> str:
    if not values:
        return "sem dados"
    return (f"n={len(values)}  média={statistics.mean(values):.0f}  mediana={statistics.median(values):.0f}  "
            f"p95={_percentile(values, 95):.0f}  mín={min(values):.0f}  máx={max(values):.0f}")


class SessionLog:

    def __init__(self, args, labels_count: int):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("para medir a CPU instale o psutil: pip install psutil") from exc
        self._psutil = psutil
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder = TESTS_DIR / f"sessao_{stamp}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        self.fps = 0.0
        self.latencies: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}
        self.cpu_process: list[float] = []
        self.cpu_system: list[float] = []
        self.memory: list[float] = []
        self._lock = threading.Lock()

        self._events_file = open(self.folder / "eventos.csv", "w", newline="", encoding="utf-8-sig")
        self._events = csv.DictWriter(self._events_file, fieldnames=EVENT_FIELDS, delimiter=";")
        self._events.writeheader()
        self._cpu_file = open(self.folder / "cpu.csv", "w", newline="", encoding="utf-8-sig")
        self._cpu = csv.writer(self._cpu_file, delimiter=";")
        self._cpu.writerow(["hora", "t_s", "cpu_processo_pct", "cpu_sistema_pct", "memoria_mb", "fps"])

        self.header = [
            f"Sessão: {stamp}",
            f"Computador: {platform.node()} | {platform.platform()} | {platform.processor()}",
            f"Núcleos lógicos: {os.cpu_count()} | Python {platform.python_version()}",
            f"Modo: {args.modo} | pausa da frase: {args.pausa_frase}s | limiar: {args.limiar} | "
            f"entrada: {args.video or f'câmera {args.camera}'} | palavras no modelo: {labels_count}",
        ]
        self._process = psutil.Process()
        self._process.cpu_percent(None)  # a primeira leitura sempre vem zerada
        psutil.cpu_percent(None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_cpu, daemon=True)
        self._thread.start()
        print(f"[registro] gravando a sessão em {self.folder}")

    def _now(self) -> tuple[str, str]:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3], f"{time.monotonic() - self.start:.3f}"

    def _sample_cpu(self) -> None:
        cores = os.cpu_count() or 1
        while not self._stop.wait(CPU_INTERVAL):
            # cpu_percent do processo soma os núcleos (pode passar de 100%); dividimos
            # pelo número de núcleos para ficar na mesma escala do uso do sistema.
            process = self._process.cpu_percent(None) / cores
            system = self._psutil.cpu_percent(None)
            memory = self._process.memory_info().rss / 2**20
            with self._lock:
                self.cpu_process.append(process)
                self.cpu_system.append(system)
                self.memory.append(memory)
                hora, t = self._now()
                self._cpu.writerow([hora, t, f"{process:.1f}", f"{system:.1f}", f"{memory:.0f}", f"{self.fps:.1f}"])
                self._cpu_file.flush()

    def event(self, kind: str, text: str = "", confidence: float | None = None,
              duration: float | None = None, latency_ms: float | None = None, detail: str = "") -> None:
        with self._lock:
            hora, t = self._now()
            self._events.writerow({
                "hora": hora, "t_s": t, "evento": kind, "texto": text,
                "confianca": "" if confidence is None else f"{confidence:.3f}",
                "duracao_sinal_s": "" if duration is None else f"{duration:.2f}",
                "latencia_ms": "" if latency_ms is None else f"{latency_ms:.0f}",
                "detalhe": detail,
            })
            self._events_file.flush()
            self.counts[kind] = self.counts.get(kind, 0) + 1
            if latency_ms is not None:
                self.latencies.setdefault(kind, []).append(latency_ms)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        with self._lock:
            lines = self.header + [
                f"Duração: {time.monotonic() - self.start:.0f} s",
                "",
                "Eventos: " + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())),
                "",
                "Latências (ms):",
            ]
            for kind, values in sorted(self.latencies.items()):
                lines.append(f"  {kind}: {_stats(values)}")
            lines += [
                "",
                f"CPU do LibrasFone (% do total da máquina): {_stats(self.cpu_process)}",
                f"CPU do sistema inteiro (%): {_stats(self.cpu_system)}",
                f"Memória do LibrasFone (MB): {_stats(self.memory)}",
            ]
            (self.folder / "resumo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._events_file.close()
            self._cpu_file.close()
        print(f"[registro] resumo salvo em {self.folder / 'resumo.txt'}")
