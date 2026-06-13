# Zbuduj Pipeline Asystenta Głosowego — Capstone Fazy 6

> Wszystko z lekcji 01-11, zszyte razem. Zbuduj asystenta głosowego, który słucha, rozumuje i odpowiada. W 2026 to rozwiązany problem inżynieryjny, nie badawczy — ale szczegóły integracji decydują, czy wdrożenie się powiedzie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## Problem

Zbuduj asystenta end-to-end:

1. Przechwytuje wejście z mikrofonu (16 kHz mono).
2. Wykrywa początek/koniec mowy użytkownika.
3. Transkrybuje strumieniowo.
4. Przekazuje transkrypt do LLM, który może wywoływać narzędzia (timer, pogoda, kalendarz).
5. Strumieniuje tekst LLM do TTS.
6. Odtwarza audio z powrotem użytkownikowi.
7. Zatrzymuje się, jeśli użytkownik przerwie w trakcie odpowiedzi.

Cel opóźnienia: pierwszy bajt audio TTS w ciągu 800 ms od zakończenia wypowiedzi użytkownika na laptopie CPU. Cel jakości: żadnych brakujących słów, żadnych halucynacji napisów na ciszy, żadnego wycieku klonowania głosu, żadnego udanego wstrzyknięcia prompta.

## Koncepcja

![Pipeline asystenta głosowego: mikrofon → VAD → STT → LLM+narzędzia → TTS → głośnik](../assets/voice-assistant.svg)

### Siedem komponentów

1. **Przechwytywanie audio.** Mikrofon → 16 kHz mono → 20 ms fragmenty. Zazwyczaj `sounddevice` w Pythonie lub natywny AudioUnit/ALSA/WASAPI w produkcji.
2. **VAD (Lekcja 11).** Silero VAD @ próg 0.5, minimalna mowa 250 ms, zwłoka ciszy 500 ms. Sygnalizuje "początek" i "koniec."
3. **Strumieniowe STT (Lekcja 4-5).** Whisper-streaming, Parakeet-TDT lub Deepgram Nova-3 (API). Transkrypty częściowe + końcowe.
4. **LLM z wywoływaniem narzędzi.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. Schemat JSON dla narzędzi. Strumieniuj tokeny.
5. **Strumieniowy TTS (Lekcja 7).** Kokoro-82M (najszybszy otwarty) lub Cartesia Sonic (komercyjny). Rozpocznij TTS po 20 tokenach LLM.
6. **Odtwarzanie.** Wyjście na głośnik; opus-encode dla sieci o niskiej przepustowości.
7. **Handler przerwania.** Jeśli VAD uruchomi się podczas odtwarzania TTS, zatrzymaj odtwarzanie, anuluj LLM, uruchom ponownie STT.

### Trzy tryby awarii, które napotkasz

1. **Ucięcie pierwszego słowa.** VAD uruchamia się o ułamek sekundy za późno. Brakuje "hej" użytkownika. Ustaw próg na 0.3, nie 0.5.
2. **Zamieszanie przy przerwaniu w trakcie odpowiedzi.** LLM kontynuuje generowanie po przerwaniu przez użytkownika; asystent mówi nad użytkownikiem. Podłącz VAD → anuluj-LLM.
3. **Halucynacje ciszy.** Whisper produkuje "Thanks for watching" na cichych ramkach rozgrzewkowych. Zawsze stosuj bramę VAD.

### Produkcyjne stosy referencyjne 2026

| Stos | Opóźnienie | Licencja | Uwagi |
|------|-----------|----------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | komercyjne API | Domyślny standard branżowy 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | głównie otwarte | Przyjazny DIY |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Pojedynczy model; inna architektura, lekcja 15 |
| Vapi / Retell (zarządzane) | 300-500 ms | komercyjne | Najszybsze do uruchomienia; ograniczona personalizacja |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | otwarte | Prywatność / edge |

## Zbuduj To

### Krok 1: przechwytywanie mikrofonu z fragmentowaniem (pseudokod)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Krok 2: przechwytywanie tury z bramą VAD

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Krok 3: strumieniowe STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Krok 4: wywoływanie narzędzi w pętli LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Krok 5: obsługa przerwania

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Użyj Tego

Zobacz `code/main.py` dla uruchamialnej symulacji, która łączy wszystkie siedem komponentów z zastępczymi modelami, abyś mógł zobaczyć kształt pipeline'u nawet bez sprzętu. Do prawdziwej implementacji zamień zaślepki na:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` lub `openai-whisper`
- `openai` (`gpt-4o`) lub `anthropic`
- `kokoro` lub `cartesia`
- `sounddevice` dla I/O

## Pułapki

- **Logowanie PII na zawsze.** Pełna tura audio to PII w większości jurysdykcji. 30-dniowa retencja, szyfrowane w spoczynku.
- **Brak wtargnięcia (barge-in).** Użytkownicy będą przerywać. Twój asystent musi przestać mówić.
- **TTS, który blokuje.** Synchroniczny TTS blokuje pętlę zdarzeń. Użyj async lub osobnego wątku.
- **Brak obsługi błędów wywołań narzędzi.** Narzędzia zawodzą. LLM musi otrzymać błąd + spróbować ponownie raz, a następnie łagodnie degradować.
- **Zbyt gorliwe filtry halucynacji.** Za mocny filtr i asystent powtarza "Nie mogę w tym pomóc." Za słaby i mówi wszystko. Kalibruj na wstrzymanym zbiorze.
- **Brak opcji wywołania (wake word).** Zawsze nasłuchujący to zobowiązanie prywatności. Dodaj bramę wywołania (Porcupine lub openWakeWord).

## Wyślij To

Zapisz jako `outputs/skill-voice-assistant-architect.md`. Dając budżet + skalę + język + ograniczenia zgodności, wyprodukuj pełną specyfikację stosu.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Symuluje jedną pełną turę end-to-end z zastępczymi modułami i wypisuje opóźnienie na etap.
2. **Średnie.** Zastąp zaślepkę STT prawdziwym modelem Whisper na nagranym `.wav`. Zmierz WER i opóźnienie end-to-end.
3. **Trudne.** Dodaj wywoływanie narzędzi: zaimplementuj `get_weather` (dowolne API) i `set_timer`. Przekieruj LLM przez narzędzia i zweryfikuj, że gdy użytkownik powie "set a 5 minute timer", właściwa funkcja zostanie uruchomiona, a odpowiedź mówiona potwierdzi.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Turn | Podróż użytkownik + asystent w obie strony | Jedna mowa użytkownika ograniczona VAD + jedna odpowiedź LLM-TTS. |
| Barge-in | Przerwanie | Użytkownik mówi, gdy asystent mówi; asystent przestaje. |
| Wake word | "Hej asystencie" | Krótki detektor słów kluczowych; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Zakończenie tury | VAD + decyzja o minimalnej ciszy, że użytkownik skończył. |
| Pre-roll | Bufor przed-mową | Zachowaj 200-400 ms audio przed uruchomieniem VAD, aby uniknąć ucięcia pierwszego słowa. |
| Tool call | Wywołanie funkcji | LLM emituje JSON; środowisko wykonawcze wysyła; wynik wraca do pętli. |

## Dalsza Lektura

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — referencja produkcyjna.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — framework przyjazny DIY.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — zarządzana ścieżka natywna dla mowy.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — referencja full-duplex (Lekcja 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — bramka wywołania.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — wywoływanie funkcji LLM.