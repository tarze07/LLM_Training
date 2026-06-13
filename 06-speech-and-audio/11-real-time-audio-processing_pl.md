# Przetwarzanie Audio w Czasie Rzeczywistym

> Pipeline'y wsadowe przetwarzają plik. Pipeline'y czasu rzeczywistego przetwarzają następne 20 milisekund, zanim nadejdzie następne 20. Każda konwersacyjna AI, studio nadawcze i bot telefoniczny żyje lub umiera przez ten budżet opóźnienia.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Problem

Chcesz asystenta głosowego, który wydaje się żywy. Ludzkie opóźnienie zmiany tury w konwersacji wynosi ~230 ms (cisza-do-odpowiedzi). Wszystko powyżej 500 ms wydaje się robotyczne; powyżej 1500 ms wydaje się zepsute. Budżet na pełną pętlę **słyszeć → zrozumieć → odpowiedzieć → mówić** w 2026 wynosi:

| Etap | Budżet |
|------|--------|
| Mikrofon → bufor | 20 ms |
| VAD | 10 ms |
| ASR (strumieniowy) | 150 ms |
| LLM (pierwszy token) | 100 ms |
| TTS (pierwszy fragment) | 100 ms |
| Renderowanie → głośnik | 20 ms |
| **Razem** | **~400 ms** |

Moshi (Kyutai, 2024) osiągnął 200 ms full-duplex. GPT-4o-realtime (2024) osiąga ~320 ms. Kaskadowe pipeline'y w 2022 dostarczały 2500 ms. 10× poprawa pochodzi z trzech technik: (1) strumieniowanie wszędzie, (2) asynchroniczne pipelining z częściowymi wynikami, (3) przerywalna generacja.

## Koncepcja

![Strumieniowy pipeline audio z buforem pierścieniowym, bramą VAD, przerwaniem](../assets/real-time.svg)

**Ramka / fragment / okno.** Audio w czasie rzeczywistym płynie jako bloki o stałym rozmiarze. Typowy wybór: 20 ms (320 próbek przy 16 kHz). Wszystko poniżej musi nadążać za tym tempem.

**Bufor pierścieniowy.** Bufor cykliczny o stałym rozmiarze. Wątek producenta zapisuje nowe ramki, wątek konsumenta odczytuje. Zapobiega alokacjom na gorącej ścieżce. Rozmiar ≈ maksymalne-opóźnienie × częstotliwość próbkowania; 2-sekundowy pierścień 16 kHz = 32 000 próbek.

**VAD (Voice Activity Detection).** Blokuje dalsze przetwarzanie, gdy nikt nie mówi. Silero VAD 4.0 (2024) działa <1 ms na 30 ms ramkę na CPU. `webrtcvad` to starsza alternatywa.

**Strumieniowy ASR.** Modele emitujące częściowe transkrypty w miarę napływania audio. Parakeet-CTC-0.6B w trybie strumieniowym (NeMo, 2024) osiąga 2–5% WER przy 320 ms opóźnienia. Whisper-Streaming (Macháček et al., 2023) dzieli Whispera na fragmenty dla prawie-strumieniowania przy ~2 s opóźnienia.

**Przerwanie (interruption).** Gdy użytkownik mówi, podczas gdy asystent mówi, musisz (a) wykryć wtargnięcie, (b) zatrzymać TTS, (c) odrzucić pozostałe wyjście LLM. Wszystko w ciągu 100 ms, w przeciwnym razie użytkownik odbiera głuchego asystenta.

**Transport WebRTC Opus.** Ramki 20 ms, 48 kHz, adaptacyjna przepływność 8–128 kbps. Standard dla przeglądarki i urządzeń mobilnych. LiveKit, Daily.co, Pion to stosy 2026 do budowania aplikacji głosowych.

**Bufor jittera.** Pakiety sieciowe przychodzą w innej kolejności / późno. Bufor jittera porządkuje i wygładza; zbyt mały → słyszalne przerwy, zbyt duży → opóźnienie. Typowo 60–80 ms.

### Typowe problemy

- **Walka wątków.** GIL Pythona + ciężkie modele mogą zagłodzić wątek audio. Użyj biblioteki audio z callbackiem w C (sounddevice, PortAudio) i trzymaj Pythona z dala od gorącej ścieżki.
- **Opóźnienie konwersji częstotliwości próbkowania.** Resampling wewnątrz pipeline'u dodaje 5–20 ms. Resampluj z góry lub użyj resampera o zerowym opóźnieniu (PolyPhase, `soxr_hq`).
- **Rozgrzewanie TTS.** Nawet szybki TTS jak Kokoro ma 100–200 ms rozgrzewki przy pierwszym żądaniu. Cache'uj model + rozgrzej go fałszywym uruchomieniem przed pierwszą prawdziwą turą.
- **Eliminacja echa.** Bez AEC, wyjście TTS wraca do mikrofonu i wyzwala ASR na własnym głosie bota. WebRTC AEC3 to domyślne open-source.

```figure
nyquist-aliasing
```

## Zbuduj To

### Krok 1: bufor pierścieniowy

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Pojemność określa maksymalne opóźnienie buforowania. 32 000 próbek przy 16 kHz = 2 s.

### Krok 2: brama VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Zamień na Silero VAD w produkcji:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Krok 3: strumieniowy ASR

```python
# Parakeet-CTC-0.6B strumieniowo przez NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Krok 4: handler przerwania

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # wtargnięcie
        # następnie podaj do strumieniowego ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Opiera się na async I/O i możliwym do anulowania strumieniowaniu TTS. WebRTC peerconnection.stop() na ścieżce audio to kanoniczny sposób.

## Użyj Tego

Stos 2026:

| Warstwa | Wybór |
|--------|-------|
| Transport | LiveKit (WebRTC) lub Pion (Go) |
| VAD | Silero VAD 4.0 |
| Strumieniowy ASR | Parakeet-CTC-0.6B lub Whisper-Streaming |
| LLM pierwszy token | Groq, Cerebras, vLLM-streaming |
| Strumieniowy TTS | Kokoro lub ElevenLabs Turbo v2.5 |
| Eliminacja echa | WebRTC AEC3 |
| End-to-end natywny | OpenAI Realtime API lub Moshi |

## Pułapki

- **Buforowanie 500 ms dla bezpieczeństwa.** Bufor *jest* twoim minimalnym opóźnieniem. Zmniejsz go.
- **Nieprzypinanie wątków.** Callback audio na wątku o niższym priorytecie niż UI = zacinanie pod obciążeniem.
- **Zbyt małe fragmenty TTS.** Fragmenty poniżej 200 ms powodują słyszalne artefakty vocodera. 320 ms to słodki punkt.
- **Brak bufora jittera.** Prawdziwe sieci są jitterowe; bez wygładzania dostajesz trzaski.
- **Jednorazowa obsługa błędów.** Pipeline'y audio muszą być odporne na awarie. Jeden wyjątek zabija sesję.

## Wyślij To

Zapisz jako `outputs/skill-realtime-designer.md`. Zaprojektuj pipeline audio w czasie rzeczywistym z konkretnymi budżetami opóźnienia na etap.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Symuluje bufor pierścieniowy + energy VAD; wypisuje opóźnienia etapów dla fałszywego 10-sekundowego strumienia.
2. **Średnie.** Używając `sounddevice`, zbuduj pętlę passthrough, która przetwarza mikrofon w 20 ms ramkach i wypisuje stan VAD na każdej ramce.
3. **Trudne.** Zbuduj pełny test echa dupleksowego z `aiortc`: przeglądarka → WebRTC → Python → WebRTC → przeglądarka. Zmierz opóźnienie glass-to-glass z impulsem 1 kHz.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | Kolejka cykliczna | FIFO o stałym rozmiarze, lock-free (lub SPSC-locked) dla ramek audio. |
| VAD | Brama ciszy | Model lub heurystyka oznaczająca mowę vs nie-mowę. |
| Streaming ASR | STT w czasie rzeczywistym | Emituje częściowy tekst w miarę napływania audio; ograniczone wyprzedzenie. |
| Jitter buffer | Wygładzacz sieci | Kolejka porządkująca nieuporządkowane pakiety; typowo 60–80 ms. |
| AEC | Eliminacja echa | Odejmuje ścieżkę sprzężenia zwrotnego głośnik-mikrofon. |
| Barge-in | Przerwanie użytkownika | System wykrywa mowę użytkownika podczas TTS; musi anulować odtwarzanie. |
| Full duplex | Jednocześnie obie strony | Użytkownik i bot mogą mówić jednocześnie; Moshi jest full duplex. |

## Dalsza Lektura

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked near-streaming Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — produkcyjna orkiestracja agentów audio.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — VAD poniżej 1 ms, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — eliminacja echa w open source.