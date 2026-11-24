"""Modo roteiro: o sistema pede cada sinal e anota o que foi reconhecido.

Usado com `python librasfone.py --roteiro` (ou pelo painel). A cada sinal
captado, a tentativa é registrada com a palavra pedida e a reconhecida, e o
roteiro passa para a próxima palavra. Teclas: N pula a palavra pedida e
D descarta a última tentativa (por exemplo, quando a mão subiu sem querer).

Ao fechar, grava na pasta da sessão:
  roteiro.csv          uma linha por tentativa ou palavra pulada
  roteiro_resumo.txt   acertos por palavra, palavras puladas e confusões
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ["ordem", "pedida", "reconhecida", "situacao", "confianca", "resultado"]


class Roteiro:

    def __init__(self, labels: list[str], falas: dict, repetitions: int, folder: Path, seed: int = 42):
        words = sorted(label for label in labels if not label.startswith("_"))
        rng = random.Random(seed)
        self.items: list[str] = []
        for _ in range(max(1, repetitions)):
            round_words = list(words)
            rng.shuffle(round_words)  # cada rodada em outra ordem, para não decorar a sequência
            self.items += round_words
        self.falas = falas
        self.folder = folder
        self.position = 0
        self.rows: list[dict] = []

    def name(self, label: str) -> str:
        return self.falas.get(label, label.replace("_", " "))

    @property
    def done(self) -> bool:
        return self.position >= len(self.items)

    def current(self) -> str | None:
        return None if self.done else self.items[self.position]

    def progress_text(self) -> str:
        if self.done:
            return "ROTEIRO CONCLUÍDO - aperte Q ou Parar"
        return f"FAÇA: {self.name(self.current()).upper()}   ({self.position + 1}/{len(self.items)})"

    def record(self, predicted: str, status: str, confidence: float) -> str:
        # status: "accepted" (falado), "uncertain" (abaixo do limiar) ou "unknown" (rejeitado)
        if self.done:
            return ""
        wanted = self.current()
        if status == "unknown":
            result = "rejeitado"
        elif status == "uncertain":
            result = "incerto"
        elif predicted == wanted:
            result = "acerto"
        else:
            result = "falado_errado"
        self.rows.append({
            "ordem": self.position + 1, "pedida": self.name(wanted),
            "reconhecida": "" if status == "unknown" else self.name(predicted),
            "situacao": status, "confianca": f"{confidence:.3f}", "resultado": result,
        })
        self.position += 1
        print(f"[roteiro] pedida: {self.name(wanted)} -> {result}")
        return result

    def skip(self) -> None:
        if self.done:
            return
        self.rows.append({"ordem": self.position + 1, "pedida": self.name(self.current()),
                          "reconhecida": "", "situacao": "", "confianca": "", "resultado": "pulado"})
        print(f"[roteiro] pulada: {self.name(self.current())}")
        self.position += 1

    def discard_last(self) -> None:
        # Volta para a palavra da última tentativa, que é apagada do registro.
        if not self.rows:
            return
        removed = self.rows.pop()
        self.position = removed["ordem"] - 1
        print(f"[roteiro] tentativa descartada: {removed['pedida']}")

    def close(self) -> None:
        with open(self.folder / "roteiro.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
            writer.writeheader()
            writer.writerows(self.rows)

        per_word: dict[str, Counter] = defaultdict(Counter)
        confusions = Counter()
        for row in self.rows:
            per_word[row["pedida"]][row["resultado"]] += 1
            if row["resultado"] == "falado_errado":
                confusions[(row["pedida"], row["reconhecida"])] += 1
        total = Counter(row["resultado"] for row in self.rows)
        attempts = sum(v for k, v in total.items() if k != "pulado")

        def pct(part: int, whole: int) -> str:
            return f"{100 * part / whole:.1f}%" if whole else "-"

        lines = [
            f"Palavras pedidas: {len(self.rows)} de {len(self.items)} previstas",
            f"Tentativas (sem contar puladas): {attempts}",
            f"  acertos: {total['acerto']} ({pct(total['acerto'], attempts)})",
            f"  falado errado: {total['falado_errado']} ({pct(total['falado_errado'], attempts)})",
            f"  incerto (não falado): {total['incerto']} ({pct(total['incerto'], attempts)})",
            f"  rejeitado (não falado): {total['rejeitado']} ({pct(total['rejeitado'], attempts)})",
            f"Puladas: {total['pulado']}",
            "",
            "Por palavra (acertos/tentativas | errado | incerto | rejeitado | puladas):",
        ]
        for word in sorted(per_word):
            c = per_word[word]
            tries = sum(v for k, v in c.items() if k != "pulado")
            lines.append(f"  {word:12s} {c['acerto']}/{tries}  | {c['falado_errado']} | {c['incerto']} | "
                         f"{c['rejeitado']} | {c['pulado']}")
        if confusions:
            lines += ["", "Confusões (pedida -> falada):"]
            lines += [f"  {a} -> {b}: {n}" for (a, b), n in confusions.most_common()]
        (self.folder / "roteiro_resumo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[roteiro] resultado salvo em {self.folder / 'roteiro_resumo.txt'}")
