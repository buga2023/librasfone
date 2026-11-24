from display import WHITE, YELLOW
from tts import join_phrase


class Translator:

    def __init__(self, args, falas, voices, audio_out, log=None):
        self.phrase_mode = args.modo == "frase"
        self.pause = args.pausa_frase
        self.falas = falas
        self.voices = voices
        self.audio_out = audio_out
        self.phrase: list[str] = []  # palavras ainda não faladas
        self.spoken: list[str] = []  # última frase falada (continua na legenda por alguns segundos)
        self.spoken_until = 0.0
        self.last_activity = 0.0
        # Falas enviadas para o áudio e ainda não iniciadas: servem para medir a latência.
        self.pending: list = []
        self.log = log  # registro da sessão (--registrar), ou None
        self.last_sign_end = 0.0

    def add_word(self, text: str, sign_end: float) -> None:
        self.last_sign_end = max(self.last_sign_end, sign_end)
        if self.phrase_mode:
            self.phrase.append(text)
            self.last_activity = max(self.last_activity, sign_end)
        else:
            self.spoken.append(text)
            self._play([text], sign_end, "fim do sinal -> voz")

    def speak(self, now: float) -> None:
        if not self.phrase:
            return
        words, self.phrase = self.phrase, []
        self.spoken = list(words)
        self.spoken_until = now + 4.0
        print(f">>> FRASE: {' '.join(words)}")
        if self.log is not None:
            self.log.event("frase_enviada", " ".join(words))
        self._play(words, now, "fim da pausa -> voz")

    def _play(self, words: list[str], reference: float, what: str) -> None:
        if self.audio_out is None:
            self._no_voice(words, "áudio desligado")
            return
        clip = join_phrase(words, self.voices)
        if clip is None:
            self._no_voice(words, "sem voz gerada para a palavra")
            return
        playback = self.audio_out.play(*clip)
        if playback is not None:
            self.pending.append((playback, reference, " ".join(words), what, self.last_sign_end))
        else:
            self._no_voice(words, "mudo ou sem dispositivo de saída")

    def _no_voice(self, words: list[str], reason: str) -> None:
        # Registra por que a voz não saiu, para o teste não ficar sem explicação.
        if self.log is not None:
            self.log.event("voz_nao_enviada", " ".join(words), detail=reason)

    def undo(self) -> None:
        if self.phrase:
            removed = self.phrase.pop()
            print(f"(apagado: {removed})")

    def clear(self) -> None:
        self.phrase.clear()
        self.spoken.clear()

    def tick(self, now: float, capturing: bool) -> float | None:
        # Mostra a latência das falas que já começaram a tocar; descarta as que
        # nunca começaram depois de 10 s (dispositivo de áudio travado, por exemplo).
        for item in list(self.pending):
            playback, reference, text, what, sign_end = item
            if playback.started is not None:
                print(f"    [latência] {what}: {1000 * (playback.started - reference):.0f} ms  ({text})")
                if self.log is not None:
                    self.log.event("voz_iniciou", text, latency_ms=1000 * (playback.started - reference), detail=what)
                    if self.phrase_mode:
                        # No modo frase também interessa o tempo desde o fim do último sinal.
                        self.log.event("voz_desde_ultimo_sinal", text,
                                       latency_ms=1000 * (playback.started - sign_end))
                self.pending.remove(item)
            elif now - playback.queued_at > 10:
                self.pending.remove(item)
                if self.log is not None:
                    self.log.event("voz_nao_iniciou", text, detail="o dispositivo não começou a tocar em 10 s")
        # Enquanto a pessoa está sinalizando, a pausa não conta.
        if capturing:
            self.last_activity = now
        if not self.phrase_mode or not self.phrase:
            return None
        waited = now - self.last_activity
        if waited >= self.pause:
            self.speak(now)
            return None
        return waited / self.pause

    def subtitle(self, now: float) -> tuple[list[str], tuple]:
        if self.phrase:
            return self.phrase, YELLOW
        if self.spoken and (not self.phrase_mode or now < self.spoken_until):
            return self.spoken, WHITE
        return [], WHITE
