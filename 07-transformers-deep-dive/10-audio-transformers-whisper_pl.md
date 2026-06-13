# Audio Transformers — Architektura Whisper

> Audio to obraz częstotliwości w czasie. Whisper to ViT, który zjada spektrogramy mel i odpowiada mową.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 minutes

## Problem

Przed Whisper (OpenAI, Radford i in. 2022), najnowocześniejsze automatyczne rozpoznawanie mowy (ASR) oznaczało wav2vec 2.0 i HuBERT — samonadzorowane ekstraktory cech plus dostrojona głowa. Wysoka jakość, drogie potoki danych, kruchość domenowa. Wielojęzyczne rozpoznawanie mowy wymagało oddzielnych modeli na każdą rodzinę językową.

Whisper postawił trzy zakłady:

1. **Trenuj na wszystkim.** 680 000 godzin słabo oznakowanego audio zebranego z internetu w 97 językach. Żadnego czystego korpusu akademickiego. Żadnych etykiet fonemów.
2. **Pojedynczy model do wielu zadań.** Jeden dekoder trenowany wspólnie do transkrypcji, tłumaczenia, detekcji aktywności głosowej, identyfikacji języka i znaczników czasowych za pomocą tokenów zadań.
3. **Standardowy enkoder-dekoder transformer.** Enkoder konsumuje spektrogramy log-mel. Dekoder produkuje tokeny tekstu autoregresywnie. Żadnego wokodera, żadnego CTC, żadnego HMM.

Rezultat: Whisper large-v3 jest odporny na akcenty, szum i języki, które mają zero czystych oznakowanych danych. Jest domyślnym front-endem mowy dla każdego asystenta głosowego open-source i większości komercyjnych w 2026 roku.

## Koncepcja

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Krok 1 — resamplowanie + okno

Audio przy 16 kHz. Przytnij/wypełnij do 30 sekund. Oblicz spektrogram log-mel: 80 pasm mel, krok 10 ms → ~3,000 ramek × 80 cech. To jest "obraz wejściowy", który widzi Whisper.

### Krok 2 — splotowy pień

Dwie warstwy Conv1D o jądrze 3 i kroku 2 redukują 3,000 ramek do 1,500. Zmniejsza długość sekwencji o połowę bez dodawania wielu parametrów.

### Krok 3 — enkoder

24-warstwowy (dla dużego modelu) enkoder transformerów nad 1,500 krokami czasowymi. Sinusoidalne kodowanie pozycyjne, samouwaga, FFN z GELU. Produkuje 1,500 × 1,280 stanów ukrytych.

### Krok 4 — dekoder

24-warstwowy dekoder transformerów. Autoregresywnie produkuje tokeny ze słownika BPE, który jest nadzbiorem słownika GPT-2 z kilkoma specjalnymi tokenami specyficznymi dla audio.

### Krok 5 — tokeny zadań

Prompt dekodera zaczyna się od tokenów kontrolnych, które mówią modelowi, co ma robić:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

lub

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Model był trenowany na tej konwencji. Sterujesz zadaniem przez prefiks. Odpowiednik strojenia instrukcji z 2026 roku, ale zastosowany do mowy.

### Krok 6 — wyjście

Przeszukiwanie wiązkowe (szerokość 5) z progiem log-prawdopodobieństwa. Znaczniki czasowe są przewidywane co 0.02 sekundy audio, gdy brak tokena `<|notimestamps|>`.

### Rozmiary Whisper

| Model | Parametry | Warstwy | d_model | Głowy | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-warstwowy dekoder) |

Large-v3-turbo (2024) skrócił dekoder z 32 warstw do 4. 8× szybsze dekodowanie z regresją <1 punktu WER. To odblokowanie szybkości dekodowania jest powodem, dla którego Whisper-turbo jest domyślnym wyborem dla agentów głosowych czasu rzeczywistego w 2026 roku.

### Czego Whisper nie robi

- Brak diarizacji (kto mówi). Połącz z pyannote, aby to uzyskać.
- Brak natywnego strumieniowania w czasie rzeczywistym — okno 30-sekundowe jest stałe. Nowoczesne nakładki (`faster-whisper`, `WhisperX`) dodają strumieniowanie przez VAD + nakładanie.
- Brak kontekstu długiego formatu powyżej 30 s bez zewnętrznego dzielenia na części. Działa dobrze w praktyce, ponieważ ludzka mowa rzadko potrzebuje długiego kontekstu do transkrypcji.

### Krajobraz 2026

| Zadanie | Model | Uwagi |
|------|-------|-------|
| ASR angielski | Whisper-turbo, Moonshine | Moonshine jest 4× szybszy na urządzeniach brzegowych |
| ASR wielojęzyczny | Whisper-large-v3 | 97 języków |
| ASR strumieniowy | faster-whisper + VAD | Możliwe do osiągnięcia cele opóźnienia 150 ms |
| TTS | Piper, XTTS-v2, Kokoro | Wzorzec enkoder-dekoder, ale w kształcie Whisper |
| Audio + język | AudioLM, SeamlessM4T | Tokeny tekstu + tokeny audio w jednym transformerze |

## Zbuduj To

Zobacz `code/main.py`. Nie trenujemy Whisper — budujemy potok spektrogramu log-mel + formatowanie prompta z tokenami zadań. To są części, których faktycznie dotykasz w produkcji.

### Krok 1: syntetyzuj audio

Wygeneruj 1-sekundową falę sinusoidalną 440 Hz próbkowaną przy 16 kHz. 16,000 próbek.

### Krok 2: spektrogram log-mel (uproszczony)

Pełny spektrogram mel wymaga FFT. Robimy uproszczoną wersję z ramkowaniem + energią na ramkę, która pokazuje potok bez potrzeby używania `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Ramka = 25 ms, krok = 10 ms. Zgodne z okienkowaniem Whisper. Energia na ramkę zastępuje pasma mel dla celów pedagogicznych.

### Krok 3: wypełnij do 30 s

Whisper zawsze przetwarza 30-sekundowe fragmenty. Wypełnij (lub przytnij) spektrogram do 3,000 ramek.

### Krok 4: zbuduj tokeny prompta

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

To jest cała powierzchnia sterowania zadaniem. 4-tokenowy prefiks.

## Użyj Tego

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Szybsza, kompatybilna z OpenAI:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Kiedy wybrać Whisper w 2026:**

- Wielojęzyczny ASR z jednym modelem.
- Solidna transkrypcja zaszumionego, zróżnicowanego audio.
- Badania / prototypowanie ASR — najszybszy punkt startowy.

**Kiedy wybrać coś innego:**

- Strumieniowanie o bardzo niskim opóźnieniu na urządzeniach brzegowych — Moonshine bije Whisper przy porównywalnej jakości.
- Konwersacyjne AI czasu rzeczywistego wymagające <200 ms — dedykowane strumieniowe ASR.
- Diarizacja mówców — Whisper tego nie robi; dołącz pyannote.

## Dostarcz To

Zobacz `outputs/skill-asr-configurator.md`. Umiejętność wybiera model ASR, parametry dekodowania i potok przetwarzania wstępnego dla nowej aplikacji mowy.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Potwierdź, że liczba ramek dla 1-sekundowego sygnału przy 16 kHz z krokiem 10 ms wynosi ~100 ramek. Dla 30 sekund: ~3,000 ramek.
2. **Średnie.** Zbuduj pełny spektrogram log-mel używając `numpy.fft`. Zweryfikuj, że 80 pasm mel odpowiada `librosa.feature.melspectrogram(n_mels=80)` w granicach błędu numerycznego.
3. **Trudne.** Zaimplementuj wnioskowanie strumieniowe: podziel audio na 10-sekundowe okna z 2-sekundowym nakładaniem, uruchom Whisper na każdym fragmencie, scal transkrypty. Zmierz współczynnik błędów słów w porównaniu do jednoprzebiegowego przetwarzania na 5-minutowej próbce podcastu.

## Kluczowe Pojęcia

| Termin | Co ludzie mówią | Co to naprawdę znaczy |
|------|-----------------|-----------------------|
| Spektrogram mel | "Obraz audio" | Reprezentacja 2D: pasma częstotliwości na jednej osi, ramki czasowe na drugiej; logarytmiczna energia na komórkę. |
| Log-mel | "Co widzi Whisper" | Spektrogram mel przepuszczony przez log; przybliża ludzką percepcję głośności. |
| Ramka (frame) | "Jeden wycinek czasu" | 25-milisekundowe okno próbek; nakładające się co 10 ms. |
| Token zadania (task token) | "Prefiks prompta dla mowy" | Specjalne tokeny takie jak `<\|transcribe\|>` / `<\|translate\|>` w prompcie dekodera. |
| Detekcja aktywności głosowej (VAD) | "Znajdź mowę" | Brama usuwająca ciszę przed ASR; znacznie obniża koszty. |
| CTC | "Connectionist Temporal Classification" | Klasyczna strata ASR do trenowania bez alignowania; Whisper NIE używa tego. |
| Whisper-turbo | "Mały dekoder, pełny enkoder" | Enkoder large-v3 + 4-warstwowy dekoder; 8× szybsze dekodowanie. |
| Faster-whisper | "Produkcyjna nakładka" | Reimplementacja CTranslate2; kwantyzacja int8; 4× szybszy niż referencja OpenAI. |

## Dalsza Lektura

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — artykuł Whisper.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — kod referencyjny + wagi modelu. Przeczytaj `whisper/model.py`, aby zobaczyć pień Conv1D + enkoder + dekoder od góry do dołu w ~400 liniach.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — logika przeszukiwania wiązkowego + tokenów zadań opisana w Krokach 5–6; 500 linii, w pełni czytelna.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — prekursor; wciąż cechy SOTA w niektórych zastosowaniach.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — nakładka produkcyjna, 4× szybsza niż referencja.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — przyjazny dla urządzeń brzegowych ASR z 2024, w kształcie Whisper ale mniejszy.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — kanoniczny przepis na dostrajanie, obejmujący preprocesor spektrogramu mel i obsługę tokenów-czasu.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — pełna implementacja (enkoder, dekoder, uwaga krzyżowa, generacja) która odzwierciedla diagram architektury z tej lekcji.