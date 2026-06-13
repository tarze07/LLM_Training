# Neuralne Kodeki Audio — EnCodec, SNAC, Mimi, DAC i Podział Semantyczno-Akustyczny

> Generowanie audio w 2026 to prawie wszystko tokeny. EnCodec, SNAC, Mimi i DAC przekształcają ciągłe przebiegi w dyskretne sekwencje, które transformer może przewidzieć. Podział tokenów semantyczny-vs-akustyczny — pierwszy codebook jako semantyczny, reszta jako akustyczna — to najważniejsza zmiana architektoniczna od czasu Transformera dla audio.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## Problem

Modele językowe działają na dyskretnych tokenach. Audio jest ciągłe. Jeśli chcesz model w stylu LLM dla mowy / muzyki — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — najpierw potrzebujesz **neuralnego kodeka audio**: uczonego enkodera, który dyskretyzuje audio na małe słownictwo tokenów, i pasującego dekodera, który rekonstruuje przebieg.

Wyłoniły się dwie rodziny:

1. **Kodeki zorientowane na rekonstrukcję** — EnCodec, DAC. Optymalizują percepcyjną jakość audio. Tokeny są "akustyczne" — przechwytują wszystko, w tym tożsamość mówcy, barwę, szum tła.
2. **Kodeki zorientowane semantycznie** — Mimi (Kyutai), SpeechTokenizer. Wymuszają, aby pierwszy codebook kodował treść językową / fonetyczną (często przez destylację z WavLM). Kolejne codebooki to szczegóły akustyczne.

Wgląd 2024-2026: **czysty kodek rekonstrukcyjny daje rozmytą mowę, gdy próbujesz generować z tekstu.** LLM nad tokenami kodeka musi nauczyć się zarówno struktury języka, JAK i struktury akustycznej w tym samym codebooku, co nie skaluje się. Oddzielenie ich — semantyczny codebook 0, akustyczne codebooki 1-N — to to, co sprawia, że Moshi i Sesame CSM działają.

## Koncepcja

![Krajobraz czterech kodeków: EnCodec, DAC, SNAC (wieloskalowy), Mimi (semantyczny+akustyczny)](../assets/codec-comparison.svg)

### Główny trik: Residual Vector Quantization (RVQ)

Zamiast jednego dużego codebooka (który potrzebowałby milionów kodów dla dobrej jakości), wszystkie nowoczesne kodeki audio używają **RVQ**: kaskady małych codebooków. Pierwszy codebook kwantyzuje wyjście enkodera; drugi kwantyzuje resztę; itd. Każdy codebook to 1024 kody. 8 codebooków = efektywne słownictwo 1024^8 = 10^24.

Przy wnioskowaniu dekoder sumuje wszystkie wybrane kody na ramkę, aby zrekonstruować.

### Cztery kodeki, które mają znaczenie w 2026

**EnCodec (Meta, 2022).** Baseline. Enkoder-dekoder nad przebiegiem, wąskie gardło RVQ. 24 kHz, 32 codebooki możliwe, domyślnie 4 codebooki @ 1.5 kbps. Używa architektury `1D conv + transformer + 1D conv`. Używany przez MusicGen.

**DAC (Descript, 2023).** RVQ z codebookami znormalizowanymi L2, okresowymi funkcjami aktywacji, ulepszonymi stratami. Najwyższa wierność rekonstrukcji spośród otwartych kodeków — czasami nieodróżnialna od oryginalnej mowy z 12 codebookami. 44.1 kHz pełne pasmo.

**SNAC (Hubert Siuzdak, 2024).** Wieloskalowe RVQ — grube codebooki działają z niższą częstotliwością ramek niż drobne. Efektywnie modeluje audio hierarchicznie: gruby "szkic" przy ~12 Hz plus szczegóły przy 50 Hz. Używany przez Orpheus-3B, ponieważ hierarchiczna struktura dobrze mapuje się na generację opartą na LM.

**Mimi (Kyutai, 2024).** Przełom 2026. 12.5 Hz częstotliwość ramek (bardzo niska), 8 codebooków @ 4.4 kbps. Codebook 0 jest **destylowany z WavLM** — trenowany do przewidywania cech treści mowy WavLM. Codebooki 1-7 to reszty akustyczne. Ten podział napędza Moshi (Lekcja 15) i Sesame CSM.

### Częstotliwości ramek mają znaczenie dla modelowania językowego

Niższa częstotliwość ramek = krótsza sekwencja = szybszy LM.

| Kodek | Częstotliwość ramek | 1 s = N ramek | Dobry do |
|-------|--------------------|---------------|----------|
| EnCodec-24k | 75 Hz | 75 | muzyka, ogólne audio |
| DAC-44.1k | 86 Hz | 86 | wysokiej wierności muzyka |
| SNAC-24k (gruby) | ~12 Hz | 12 | wydajny AR-LM |
| Mimi | 12.5 Hz | 12.5 | strumieniowa mowa |

Przy 12.5 Hz, 10-sekundowa wypowiedź to tylko 125 ramek kodeka — transformer może je łatwo przewidzieć.

### Tokeny semantyczne vs akustyczne

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Token semantyczny (codebook 0 w Mimi).** Koduje co zostało powiedziane — fonemy, słowa, treść. Destylowany z WavLM przez pomocniczą stratę predykcji.
- **Tokeny akustyczne (codebooki 1-7).** Kodują barwę, tożsamość mówcy, prozodię, szum tła, drobne szczegóły.

AR LM przewiduje najpierw token semantyczny (warunkowany na tekście), a następnie przewiduje tokeny akustyczne (warunkowane na semantycznym + referencji mówcy). Ta faktoryzacja jest powodem, dla którego nowoczesny TTS może zerokrotnie klonować głosy: model semantyczny obsługuje treść; model akustyczny obsługuje barwę.

### Jakość rekonstrukcji 2026 (bity na sekundę, niższa przepływność jest lepsza)

| Kodek | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Tradycyjne kodeki jak Opus wciąż wygrywają na bit pod względem jakości percepcyjnej. Kodeki neuralne wygrywają na **dyskretnych tokenach** (których Opus nie produkuje) i **jakości modelu generatywnego** (co LM może zrobić z tymi tokenami).

## Zbuduj To

### Krok 1: koduj z EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` przy 6 kbps. Każdy kod to 0-1023 (10-bit).

### Krok 2: dekoduj i zmierz rekonstrukcję

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Krok 3: podział semantyczno-akustyczny (w stylu Mimi)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Codebook 0 semantyczny jest dopasowany do WavLM. Możesz trenować transformer tekst-do-semantyki — znacznie mniejsze słownictwo niż przejście bezpośrednio do audio. Następnie osobny dekoder akustyczny-do-przebiegu warunkuje na referencji mówcy.

### Krok 4: dlaczego AR LM nad tokenami kodeka działa

Dla 10 s klipu mowy przy 12.5 Hz × 8 codebooków Mimi:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokenów
```

1000 tokenów to trywialny kontekst dla transformera. Transformer 256M parametrów może wygenerować 10 sekund mowy w milisekundach na nowoczesnym GPU.

## Użyj Tego

Mapuj problem → kodek:

| Zadanie | Kodek |
|---------|-------|
| Ogólne generowanie muzyki | EnCodec-24k |
| Najwyższa wierność rekonstrukcji | DAC-44.1k |
| AR LM nad mową (TTS) | SNAC lub Mimi |
| Strumieniowa mowa full-duplex | Mimi (12.5 Hz) |
| Biblioteka efektów dźwiękowych z tekstem | EnCodec + T5 condition |
| Precyzyjna edycja audio | DAC + inpainting |

Reguła kciuka: **jeśli budujesz model generatywny, zacznij od Mimi lub SNAC. Jeśli budujesz pipeline kompresji, użyj Opusa.**

## Pułapki

- **Zbyt wiele codebooków.** Dodawanie codebooków zwiększa wierność liniowo, ale długość sekwencji LM też liniowo. Zatrzymaj się na 8-12.
- **Niedopasowanie częstotliwości ramek.** Trenowanie LM na 12.5 Hz Mimi, a następnie dostrajanie na 50 Hz EnCodec cicho zawodzi.
- **Zakładanie, że wszystkie codebooki są równe.** W Mimi, codebook 0 niesie treść; utrata go niszczy zrozumiałość. Utrata codebooka 7 jest ledwo zauważalna.
- **Używanie jakości rekonstrukcji jako jedynej metryki.** Kodek może mieć świetną rekonstrukcję, ale być bezużyteczny do generacji opartej na LM, jeśli struktura semantyczna jest zła.

## Wyślij To

Zapisz jako `outputs/skill-codec-picker.md`. Wybierz kodek dla danego zadania generatywnego lub kompresji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Implementuje zabawkowy skalarny + residualny kwantyzator i mierzy błąd rekonstrukcji podczas dodawania codebooków.
2. **Średnie.** Zainstaluj `encodec` i porównaj 1, 4, 8, 32 codebooki na wstrzymanym klipie mowy. Wykreśl PESQ lub MSE vs przepływność.
3. **Trudne.** Załaduj Mimi. Zakoduj klip. Zastąp codebook 0 losowymi liczbami całkowitymi; dekoduj. Następnie zastąp codebook 7 podobnie. Porównaj dwa uszkodzenia — uszkodzenie codebooka 0 powinno zniszczyć zrozumiałość; uszkodzenie codebooka 7 powinno ledwo cokolwiek zmienić.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RVQ | Kwantyzacja residualna | Kaskada małych codebooków; każdy kwantyzuje poprzednią resztę. |
| Frame rate | Szybkość kodeka | Liczba ramek tokenów na sekundę. Niższa = szybszy LM. |
| Semantic codebook | Codebook 0 (Mimi) | Codebook destylowany z cech SSL; koduje treść. |
| Acoustic codebooks | Cała reszta | Barwa, prozodia, szum, drobne szczegóły. |
| PESQ / ViSQOL | Jakość percepcyjna | Obiektywne metryki korelujące z MOS. |
| EnCodec | Kodek Meta | Baseline RVQ; używany przez MusicGen. |
| Mimi | Kodek Kyutai | 12.5 Hz częstotliwość ramek; podział semantyczno-akustyczny; napędza Moshi. |

## Dalsza Lektura

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — baseline RVQ.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — najwyższa wierność otwarta.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — wieloskalowe RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — podział semantyczno-akustyczny, destylacja WavLM.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — dwuetapowy paradygmat semantyczny/akustyczny.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — oryginalny strumieniowy kodek RVQ.