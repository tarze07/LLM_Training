# Rozpoznawanie Mowy (ASR) — CTC, RNN-T, Uwaga

> Rozpoznawanie mowy to klasyfikacja audio w każdym kroku czasowym, sklejona modelem sekwencyjnym, który zna angielski i ciszę. CTC, RNN-T i uwaga to trzy sposoby, aby to zrobić. Wybierz jeden i zrozum, dlaczego.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## Problem

Masz 10-sekundowy klip 16 kHz. Chcesz string: "turn on the kitchen lights". Wyzwanie jest strukturalne: ramki audio nie są mapowane jeden-do-jednego na znaki. Słowo "okay" może zająć 200 ms lub 1200 ms. Cisza oddziela wypowiedź. Niektóre fonemy są dłuższe niż inne. Liczba tokenów wyjściowych nie jest znana z góry.

Trzy sformułowania rozwiązują to:

1. **CTC (Connectionist Temporal Classification).** Emituj prawdopodobieństwa tokenów na ramkę, w tym specjalny *blank*. Połącz powtórzenia i usuń blanki podczas dekodowania. Nieautoregresyjny, szybki. Używany przez wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).** Wspólna sieć przewiduje następny token na podstawie ramki enkodera i poprzednich tokenów. Możliwy do strumieniowania. Używany przez ASR Google na urządzeniach, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Enkoder kompresuje audio do stanów ukrytych, dekoder z cross-attention generuje tokeny autoregresyjnie. Używany przez Whisper, SeamlessM4T.

W 2026 SOTA WER na LibriSpeech test-clean wynosi 1,4% (Parakeet-TDT-1.1B, NVIDIA) i 1,58% (Whisper-Large-v3-turbo). Różnice są niewielkie; różnice we wdrożeniu są ogromne.

## Koncepcja

![Trzy sformułowania ASR: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Intuicja CTC.** Pozwól enkoderowi wyjść z `T` rozkładami na ramkę nad `V+1` tokenami (V znaków + blank). Dla docelowego stringa `y` o długości `U < T`, każde dopasowanie ramek, które składa się do `y`, liczy się. Straty CTC sumują się przez wszystkie takie dopasowania. Wnioskowanie: argmax na ramkę, składanie powtórzeń, usuwanie blanków.

Zalety: nieautoregresyjny, możliwy do strumieniowania, zerowe wyprzedzenie. Wada: *założenie warunkowej niezależności* — każda predykcja ramki jest niezależna od innych, więc nie ma wewnętrznego modelu języka. Naprawa za pomocą zewnętrznego LM przez beam search lub shallow fusion.

**Intuicja RNN-T.** Dodaje sieć *predyktora*, która osadza historię tokenów i *łącznik*, który łączy stan predyktora z ramką enkodera w rozkład łączny nad `V+1` (`+1` to null / brak emisji). Wyraźnie modeluje zależność warunkową, którą CTC zignorował. Możliwy do strumieniowania, ponieważ każdy krok warunkuje tylko na przeszłych ramkach i przeszłych tokenach.

Zalety: możliwy do strumieniowania + wewnętrzny LM. Wada: trening jest bardziej złożony i pamięciożerny (3D loss lattice); jądra strat RNN-T to osobna kategoria biblioteczna.

**Attention encoder-decoder.** Enkoder (6-32 warstwy transformera) na ramkach log-mel. Dekoder (6-32 warstwy transformera) z cross-attention do wyjść enkodera, aby generować tokeny autoregresyjnie. Brak ograniczenia dopasowania — uwaga może patrzeć gdziekolwiek w audio. Niemożliwy do strumieniowania, chyba że ograniczysz uwagę (chunked Whisper-Streaming, 2024).

Zalety: najwyższa jakość na offline ASR, łatwy do trenowania ze standardowym seq2seq toolingiem. Wada: opóźnienie autoregresyjne jest proporcjonalne do długości wyjścia; nie można strumieniować bez inżynierii.

### WER: ta jedna liczba

**Word Error Rate** = `(S + D + I) / N`, gdzie S=substytucje, D=delecje, I=insercje, N=liczba słów w referencji. Dopasowuje odległość Levenshteina na poziomie słowa. Niższe znaczy lepiej. WER powyżej 20% jest generalnie bezużyteczny; poniżej 5% to poziom ludzki dla czytanej mowy. Liczby 2026 na standardowych benchmarkach:

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

Wszystkie te modele są oparte na encoder-dekoder lub RNN-T. Czyste systemy CTC (wav2vec 2.0) osiągają około 1,8–2,1% na test-clean.

## Zbuduj To

### Krok 1: zachłanne dekodowanie CTC

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: lista wektorów prawdopodobieństw na ramkę
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Dwie zasady: składaj kolejne powtórzenia, usuwaj blanki. Przykład: `a a _ _ a b b _ c` → `a a b c`.

### Krok 2: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Produkcja używa prefix tree beam search z fuzją LM; to jest szkic koncepcyjny.

### Krok 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Krok 4: wnioskowanie przez Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

Jednolinijkowiec dla najsilniejszego ogólnego ASR w 2026. Działa na 24 GB GPU przy ~20× czasu rzeczywistego.

### Krok 5: strumieniowanie z Parakeet lub wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Strumieniowy ASR wymaga chunked encoder attention i carryover state; użyj biblioteki, która to obsługuje (NeMo dla Parakeet, `transformers` pipeline z `chunk_length_s`).

## Użyj Tego

Stos 2026:

| Situation | Pick |
|-----------|------|
| Angielski, offline, maksymalna jakość | Whisper-large-v3-turbo |
| Wielojęzyczny, solidny | SeamlessM4T v2 |
| Strumieniowanie, niskie opóźnienie | Parakeet-TDT-1.1B lub Riva |
| Edge, mobilny, <500 ms opóźnienia | Whisper-Tiny skwantowany lub Moonshine (2024) |
| Długie formy | Whisper z chunkowaniem opartym na VAD (WhisperX) |
| Domena specyficzna (medyczna, prawna) | Dostrój wav2vec 2.0 + fuzja LM domeny |

## Pułapki, które wciąż występują w 2026

- **Brak VAD.** Uruchamianie Whispera na ciszy produkuje halucynacje ("Thanks for watching!"). Zawsze blokuj VAD.
- **WER: znak vs słowo vs pod-słowo.** Raportuj WER na poziomie słów *po* normalizacji (małe litery, usunięta interpunkcja).
- **Dryf identyfikacji języka.** Automatyczny LID Whispera może przekierować hałaśliwe klipy do japońskiego lub walijskiego; wymuś `language="en"`, gdy wiesz.
- **Długie klipy bez chunkowania.** Whisper ma okno 30-sekundowe. Użyj `chunk_length_s=30, stride=5` dla dłuższych.

## Wyślij To

Zapisz jako `outputs/skill-asr-picker.md`. Wybierz model, strategię dekodowania, chunkowanie i fuzję LM dla danego celu wdrożenia.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Zachłannie dekoduje ręcznie stworzone wyjście CTC i oblicza WER względem referencji.
2. **Średnie.** Zaimplementuj poprawnie prefix-tree beam search z Kroku 2 (uwzględniając regułę łączenia blanków). Porównaj z zachłannym na 10-przykładowym syntetycznym zbiorze.
3. **Trudne.** Użyj `whisper-large-v3-turbo` na [LibriSpeech test-clean](https://www.openslr.org/12). Oblicz WER na pierwszych 100 wypowiedziach. Porównaj z opublikowanymi liczbami.

## Kluczowe Terminy

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| CTC | Strata blank-token | Suma po wszystkich dopasowaniach ramka-do-tokenu; nie-AR. |
| RNN-T | Strata strumieniowa | CTC + predyktor następnego tokena; obsługuje kolejność słów. |
| Attention enc-dec | Styl Whispera | Enkoder + dekoder z cross-attention; najlepsza jakość offline. |
| WER | Liczba, którą raportujesz | `(S+D+I)/N` na poziomie słów. |
| Blank | Pustka | Specjalny token w CTC oznaczający "brak emisji w tej ramce". |
| LM fusion | Zewnętrzny model języka | Dodaj ważone log-prawdopodobieństwa LM podczas beam search. |
| VAD | Brama ciszy | Detektor aktywności głosowej; przycina nie-mowę. |

## Dalsza Lektura

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — artykuł o CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — artykuł o RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — kanoniczny artykuł 2022; rozszerzenie v3-turbo w 2024.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — lider Open ASR Leaderboard 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — żywy benchmark przez 25+ modeli.