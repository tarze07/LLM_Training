# Detekcja Aktywności Głosowej i Zmiana Tury — Silero, Cobra i Sztuczka Flush

> Każdy agent głosowy żyje lub umiera na dwóch decyzjach: czy użytkownik mówi teraz i czy skończył? VAD odpowiada na pierwsze. Detekcja tury (VAD + zwłoka ciszy + model semantycznego punktu końcowego) odpowiada na drugie. Źle jedno, a twój asystent albo przerywa użytkownikom, albo nigdy nie milknie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## Problem

Trzy odrębne decyzje, które agent głosowy podejmuje na każdym 20 ms fragmencie:

1. **Czy ta ramka to mowa?** — VAD. Binarny, na ramkę.
2. **Czy użytkownik rozpoczął nową wypowiedź?** — detekcja początku.
3. **Czy użytkownik skończył?** — end-pointing (koniec tury).

Naiwna odpowiedź (próg energii) zawodzi na każdym szumie — ruch uliczny, klawiatura, gwar tłumu. Odpowiedź 2026: Silero VAD (otwarty, głęboko uczony) + model detekcji tury (semantyczne end-pointing) + kalibrowana zwłoka ciszy VAD.

## Koncepcja

![Kaskada VAD: energia → Silero → detektor tury → sztuczka flush](../assets/vad-turn-taking.svg)

### Trójpoziomowa kaskada VAD

**Poziom 1: brama energetyczna.** Najtańsza. Próg RMS przy -40 dBFS. Filtruje oczywistą ciszę, ale uruchamia się na każdym szumie powyżej progu.

**Poziom 2: Silero VAD** (2020-2026, MIT). 1M parametrów. Trenowany na 6000+ językach. Działa w ~1 ms na 30 ms ramkę na pojedynczym wątku CPU. 87.7% TPR przy 5% FPR. Otwarty domyślny wybór.

**Poziom 3: semantyczny detektor tury.** Model detekcji tury LiveKit (2024-2026) lub własny mały klasyfikator. Odróżnia "pauzę w środku zdania" od "skończyłem mówić." Używa kontekstu językowego (intonacja + ostatnie słowa), nie tylko ciszy.

### Kluczowe parametry i ich domyślne

- **Próg.** Silero produkuje prawdopodobieństwo; klasyfikuj mowę przy &gt; 0.5 (domyślnie) lub &gt; 0.3 (wrażliwy). Niższy próg = mniej uciętych pierwszych słów, więcej fałszywych alarmów.
- **Minimalny czas trwania mowy.** Odrzuć mowę krótszą niż 250 ms — zwykle kaszel lub odgłos krzesła.
- **Zwłoka ciszy (end-pointing).** Po powrocie VAD do 0, czekaj 500-800 ms przed ogłoszeniem końca tury. Zbyt krótko → przerywaj użytkownikowi. Zbyt długo → wydaje się ospały.
- **Bufor przed-mową (pre-roll).** Zachowaj 300-500 ms audio przed uruchomieniem VAD. Zapobiega ucięciu "hej."

### Sztuczka flush (Kyutai 2025)

Strumieniowe modele STT mają opóźnienie wyprzedzenia (500 ms dla Kyutai STT-1B, 2.5 s dla STT-2.6B). Normalnie czekałbyś tak długo po zakończeniu mowy na transkrypt. Sztuczka flush: gdy VAD uruchamia koniec mowy, **wyślij sygnał flush do STT**, który wymusza natychmiastowe wyjście. STT przetwarza przy ~4× czasu rzeczywistego, więc 500 ms bufor kończy się w ~125 ms.

End-to-end: 125 ms VAD + flush STT = opóźnienie konwersacyjne.

### Porównanie VAD 2026

| VAD | TPR @ 5% FPR | Opóźnienie | Licencja |
|-----|--------------|-----------|----------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | komercyjna |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero to właściwy domyślny. Cobra to ulepszenie zgodności / dokładności. VAD tylko na energii nie ma miejsca w produkcji 2026.

## Zbuduj To

### Krok 1: brama energetyczna

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Krok 2: Silero VAD w Pythonie

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Krok 3: maszyna stanów końca tury

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Krok 4: szkielet sztuczki flush

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI) musi obsługiwać flush, aby to działało. Strumieniowanie Whisper nie — jest blokowe i zawsze czeka na fragmenty.

## Użyj Tego

| Sytuacja | Wybór VAD |
|----------|----------|
| Otwarte, szybkie, ogólne | Silero VAD |
| Komercyjne call center | Cobra VAD |
| Na urządzeniu (telefon) | Silero VAD ONNX |
| Badania / diaryzacja | pyannote segmentation |
| Zastępczy bez zależności | WebRTC VAD (legacy) |
| Potrzeba jakości końca tury | Silero + LiveKit turn-detector layered |

Reguła kciuka: nigdy nie wdrażaj VAD tylko na energii, chyba że naprawdę nie masz innej opcji.

## Pułapki

- **Stały próg.** Działa w ciszy, zawodzi w hałasie. Kalibruj na urządzeniu lub przełącz na Silero.
- **Zbyt krótka zwłoka ciszy.** Agent przerywa w środku zdania. 500-800 ms to słodki punkt dla mowy konwersacyjnej.
- **Zbyt długa zwłoka.** Wydaje się ospała. Testuj A/B z docelowymi użytkownikami.
- **Brak bufora przed-mową.** Pierwsze 200-300 ms audio użytkownika tracone. Zawsze utrzymuj bufor przed-mową.
- **Ignorowanie semantycznego end-pointingu.** "Hmm, let me think..." zawiera długie pauzy. Użytkownicy nienawidzą być przerywani w trakcie myślenia. Użyj detektora tury LiveKit lub podobnego.

## Wyślij To

Zapisz jako `outputs/skill-vad-tuner.md`. Wybierz model VAD, próg, zwłokę, przed-mowę i strategię detekcji tury dla obciążenia.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Symuluje sekwencję mowa + cisza + mowa + kaszlnięcia i testuje trzy poziomy VAD.
2. **Średnie.** Zainstaluj `silero-vad`, przetwórz 5-minutowe nagranie, dostrój próg, aby zminimalizować zarówno ucięte pierwsze słowa, jak i fałszywe wyzwalacze. Raportuj precyzję/czułość.
3. **Trudne.** Zbuduj mini detektor tury: Silero VAD + 3-warstwowy MLP na osadzeniach ostatnich 10 słów (użyj sentence-transformers). Trenuj na ręcznie oznakowanym zbiorze końców tur. Pobij samego Silero o 10% F1.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| VAD | Detektor głosu | Binarny na ramkę: czy to mowa? |
| Turn detection | End-pointing | VAD + zwłoka ciszy + semantyczny endpoint. |
| Silence hangover | Czekaj-po-mowie | Czas do odczekania przed ogłoszeniem końca tury; 500-800 ms. |
| Pre-roll | Bufor przed-mową | Zachowaj 300-500 ms audio przed uruchomieniem VAD. |
| Flush trick | Sztuczka Kyutai | VAD → flush-STT → 125 ms zamiast 500 ms opóźnienia. |
| Semantic endpoint | "Czy chcieli skończyć?" | Klasyfikator ML patrzący na słowa, nie tylko ciszę. |
| TPR @ FPR 5% | Punkt ROC | Standardowy benchmark VAD; 87.7% dla Silero, 50% WebRTC. |

## Dalsza Lektura

- [Silero VAD](https://github.com/snakers4/silero-vad) — referencyjny otwarty VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — lider dokładności komercyjnej.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — sztuczka inżynieryjna poniżej 200 ms.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — semantyczne end-pointing w produkcji.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — legacy baseline.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — segmentacja klasy diaryzacji.